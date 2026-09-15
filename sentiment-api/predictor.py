import json
import re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import GRUModel, BERTClassifier

ARTIFACTS_DIR = "artifacts"


# ---------------------------------------------------------------------------
# GRU preprocessing — must stay identical to what the GRU was trained on.
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)    # keep only letters
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)            # collapse whitespace
    return text


def encode(text: str, word2idx: dict, max_len: int) -> list:
    tokens = text.split()
    ids = [word2idx.get(t, 1) for t in tokens]  # 1 = <UNK>
    ids = ids[:max_len]                          # truncate
    ids = ids + [0] * (max_len - len(ids))       # pad
    return ids


class GRUPredictor:
    """
    Loads config, vocabulary, and GRU weights once at startup,
    then serves predictions via .predict(text).
    """

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        with open(f"{artifacts_dir}/config.json") as f:
            self.config = json.load(f)

        with open(f"{artifacts_dir}/word2idx.json") as f:
            self.word2idx = json.load(f)

        self.max_len = self.config["max_len"]

        self.model = GRUModel(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
        )

        state_dict = torch.load(
            f"{artifacts_dir}/gru_model.pt", map_location="cpu"
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)
        ids = encode(cleaned, self.word2idx, self.max_len)
        x = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()

        label = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
        return {"label": label, "probability": prob}


class BERTPredictor:
    """
    Loads bert_config.json, the saved HF tokenizer, and the fine-tuned
    DistilBERT weights once at startup, then serves predictions via
    .predict(text).
    """

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        with open(f"{artifacts_dir}/bert_config.json") as f:
            self.config = json.load(f)

        self.max_length = self.config["max_length"]

        # Tokenizer was saved locally during training (vocab.txt,
        # tokenizer_config.json, special_tokens_map.json) -> load from
        # that folder rather than re-downloading.
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"{artifacts_dir}/tokenizer"
        )

        self.model = BERTClassifier(
            checkpoint=self.config["checkpoint"],
            num_classes=self.config["num_classes"],
            dropout=self.config.get("dropout", 0.3),
        )

        state_dict = torch.load(
            f"{artifacts_dir}/bert_model.pt", map_location="cpu"
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, text: str) -> dict:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(encoded["input_ids"], encoded["attention_mask"])
            probs = F.softmax(logits, dim=1).squeeze(0)

        # Index 1 = POSITIVE, index 0 = NEGATIVE — this must match the
        # label encoding used during training (0/1 = neg/pos, same as
        # the GRU's df['label'] = (sentiment == 'positive')).
        prob_positive = probs[1].item()
        label = "POSITIVE" if prob_positive >= 0.5 else "NEGATIVE"
        return {"label": label, "probability": prob_positive}
