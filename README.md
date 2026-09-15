# Sentiment Analysis with FastAPI

A REST API that serves a GRU-based sentiment classifier trained on the IMDB movie review dataset. Send it a piece of review text and it returns a POSITIVE or NEGATIVE label along with a confidence score.

## Model

- **Architecture:** GRU (Gated Recurrent Unit) recurrent neural network, built in PyTorch (`model.py`)
- **Trained on:** IMDB movie review dataset
- **Task:** Binary sentiment classification
- **Predicts:** Given a raw text review, the model outputs a probability (0–1) that the review is positive. This is converted into:
  - `label`: `"POSITIVE"` if probability ≥ 0.5, otherwise `"NEGATIVE"`
  - `probability`: the raw sigmoid output from the model

At startup, the API loads three artifacts from the `artifacts/` folder:
| File | Purpose |
|---|---|
| `config.json` | Model hyperparameters (vocab size, embedding dim, hidden size, layers, max sequence length) |
| `word2idx.json` | Vocabulary mapping used to tokenize input text |
| `gru_model.pt` | Trained PyTorch model weights |

Incoming text is cleaned (HTML tags stripped, lowercased, non-letter characters removed), tokenized against the saved vocabulary, padded/truncated to a fixed length, and passed through the GRU model.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Returns basic API metadata (name and description) |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/predict` | Accepts a JSON body `{"text": "..."}` and returns `{"label": "POSITIVE" \| "NEGATIVE", "probability": float}` |

**Example request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely amazing!"}'
```

**Example response:**
```json
{
  "label": "POSITIVE",
  "probability": 0.9979891777038574
}
```

## Swagger UI (`/docs`)

FastAPI auto-generates interactive API docs at `/docs`, where you can try out each endpoint directly from the browser.

**Easy Positive prediction:**

![Swagger UI - positive prediction](screenshots/1.png)

**Easy Negative prediction:**

![Swagger UI - negative prediction](screenshots/2.png)

**Hard Example prediction:**
![Swagger UI - negative prediction](screenshots/3.png)

![Swagger UI - negative prediction](screenshots/4.png)


## Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/rotoncsedu/Sentiment-Analysis-with-FastAPI.git
cd Sentiment-Analysis-with-FastAPI/sentiment-api
```

**2. Create and activate a virtual environment** (optional but recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the API**
```bash
python -m uvicorn main:app --reload
```

The server starts at `http://127.0.0.1:8000`.

**5. Try it out**
- Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Or open `index.html` in a browser for a simple standalone UI that calls the API.

## Project Structure

```
sentiment-api/
├── main.py            # FastAPI app and route definitions
├── predictor.py        # Loads model/vocab and runs inference
├── model.py            # GRU model architecture
├── test_api.py
├── requirements.txt
└── artifacts/
    ├── config.json
    ├── word2idx.json
    └── gru_model.pt
```