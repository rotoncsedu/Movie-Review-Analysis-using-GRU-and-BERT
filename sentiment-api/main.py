from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predictor import GRUPredictor, BERTPredictor

app = FastAPI(
    title="Sentiment Analysis Inference API",
    description="Serves GRU and BERT models trained on the IMDB movie review dataset.",
    version="2.0.0",
)

# Allow the standalone UI (served from a different port, or opened
# directly as a local file) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both predictors once at startup, not inside the predict function.
gru_predictor = GRUPredictor()
bert_predictor = BERTPredictor()


class ReviewInput(BaseModel):
    text: str = Field(..., min_length=1, description="Movie review text")


class ModelResult(BaseModel):
    label: str          # "POSITIVE" or "NEGATIVE"
    probability: float


class SentimentOutput(BaseModel):
    gru: ModelResult
    bert: ModelResult


@app.get("/")
def root():
    return {
        "name": "Sentiment Analysis Inference API",
        "description": (
            "A REST API serving GRU and BERT sentiment classifiers "
            "trained on the IMDB movie review dataset. "
            "POST a review to /predict to get both models' "
            "POSITIVE/NEGATIVE labels with probability scores."
        ),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SentimentOutput)
def predict(review: ReviewInput):
    try:
        gru_result = gru_predictor.predict(review.text)
        bert_result = bert_predictor.predict(review.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SentimentOutput(gru=gru_result, bert=bert_result)
