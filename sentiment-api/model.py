import torch.nn as nn
from transformers import AutoModel


class GRUModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                       padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)
        last_hidden = self.dropout(hidden[-1])
        return self.fc(last_hidden).squeeze(1)


class BERTClassifier(nn.Module):
    """
    DistilBERT backbone + a single linear classification head on top
    of the [CLS] token. Matches the state_dict saved as bert_model.pt:
    keys are "distilbert.*" (backbone) and "classifier.weight/bias"
    (head) — no pre_classifier layer, so pooling is done manually by
    taking the first token of last_hidden_state.
    """

    def __init__(self, checkpoint, num_classes, dropout=0.3):
        super().__init__()
        self.distilbert = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size,
                                     num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids,
                                   attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(self.dropout(cls_output))
