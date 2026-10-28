
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "saved_distilbert"

tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    yield


app = FastAPI(
    title="Sentiment Prediction API",
    description="Predict positive or negative sentiment using fine-tuned DistilBERT.",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


@app.get("/")
def home():
    return {"message": "Sentiment API is running", "docs": "/docs"}


@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="Text must not be empty or whitespace.",
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_id].item())

    # Rotten Tomatoes labels: 0 = negative, 1 = positive
    label = model.config.id2label.get(
        predicted_id,
        {0: "NEGATIVE", 1: "POSITIVE"}[predicted_id],
    )

    return {
        "text": text,
        "label": label,
        "confidence": round(confidence, 4),
    }
