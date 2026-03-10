"""
CodeSheriff — Inference Module

Provides a single function `predict_bug(code_snippet)` that returns
the predicted bug type, confidence, and label ID.

Usage (from project root):
    python -m ml.inference
"""

import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import MODELS_DIR, NUM_LABELS, LABEL_NAMES, MAX_TOKEN_LENGTH, MODEL_PATH
from utils.logger import get_logger

logger = get_logger("ml.inference")


# ---------------------------------------------------------------------------
# Singleton model holder (loaded once, reused)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_device = None


def _load_model(model_path: Optional[str] = None) -> None:
    """
    Load the fine-tuned model and tokenizer into module-level singletons.

    Supports both local paths and HuggingFace Hub repo IDs:
      - Local:  MODEL_PATH=models/codesheriff-model/final
      - Hub:    MODEL_PATH=your-username/codesheriff-bug-classifier
    """
    global _model, _tokenizer, _device

    if model_path is None:
        # Prefer the local final directory; fall back to MODEL_PATH env var
        # (which may be a Hub repo ID on Render)
        candidate = MODELS_DIR / "final"
        if candidate.exists():
            model_path = str(candidate)
        else:
            model_path = MODEL_PATH

    logger.info(f"Loading model from: {model_path}")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=NUM_LABELS
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("If using local path, ensure training is complete.")
        logger.error("If using HuggingFace Hub, ensure MODEL_PATH is set correctly.")
        raise

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)
    _model.eval()
    logger.info("Model loaded successfully ✅")


def predict_bug(code_snippet: str) -> dict:
    """
    Classify a code snippet and return prediction details.

    Returns
    -------
    dict with keys:
        label     – human-readable label name
        confidence – float 0..1
        label_id  – integer class id
    """
    # Guard: empty input
    if not code_snippet or not code_snippet.strip():
        logger.warning("Empty code snippet received — returning Clean with 0 confidence.")
        return {"label": LABEL_NAMES[0], "confidence": 0.0, "label_id": 0}

    try:
        # Lazy-load model on first call
        if _model is None:
            _load_model()

        encoding = _tokenizer(
            code_snippet,
            truncation=True,
            padding="max_length",
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(_device)
        attention_mask = encoding["attention_mask"].to(_device)

        with torch.no_grad():
            outputs = _model(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        label_id = int(torch.argmax(probs).item())
        confidence = float(probs[label_id].item())

        return {
            "label": LABEL_NAMES.get(label_id, f"Unknown({label_id})"),
            "confidence": round(confidence, 4),
            "label_id": label_id,
        }

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return {"label": "Error", "confidence": 0.0, "label_id": -1}


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_snippets = [
        'query = "SELECT * FROM users WHERE id = " + user_id',
        "result = conn.execute(query)\nreturn result.fetchone().name",
        "if discount = 100:\n    return 0",
        "for i in range(len(items) + 1):\n    total += items[i]",
    ]

    for snippet in sample_snippets:
        result = predict_bug(snippet)
        print(f"\nCode: {snippet!r}")
        print(f"  → {result}")
