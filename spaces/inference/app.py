"""
CodeSheriff Inference Space

Minimal FastAPI server that loads the fine-tuned CodeBERT classifier
and exposes a POST /predict endpoint.  Called remotely by the Render backend.
"""

import torch
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

MODEL_ID = "jayansh21/codesheriff-bug-classifier"
NUM_LABELS = 5
MAX_LENGTH = 512
LABEL_NAMES = {
    0: "Clean",
    1: "Null Reference Risk",
    2: "Type Mismatch",
    3: "Security Vulnerability",
    4: "Logic Flaw",
}

app = FastAPI(title="CodeSheriff Inference")

print("Loading CodeSheriff classifier …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=NUM_LABELS
)
model.eval()
print("Model loaded ✅")


@app.post("/predict")
def predict(data: dict):
    """Classify a code snippet and return label, confidence, label_id."""
    code = data.get("code", "")
    if not code or not code.strip():
        return {"label": "Clean", "confidence": 0.0, "label_id": 0}

    encoding = tokenizer(
        code,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encoding)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    label_id = int(torch.argmax(probs).item())
    confidence = float(probs[label_id].item())

    return {
        "label": LABEL_NAMES.get(label_id, f"Unknown({label_id})"),
        "confidence": round(confidence, 4),
        "label_id": label_id,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
