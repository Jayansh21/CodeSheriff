"""
CodeSheriff — Inference Module

Provides a single function `predict_bug(code_snippet)` that returns
the predicted bug type, confidence, and label ID.

Inference modes (controlled by USE_LOCAL_MODEL env var):
  - Remote (default): calls a HuggingFace Space via gradio_client — zero RAM.
  - Local: loads the full model into memory (for dev / offline use).
    Set USE_LOCAL_MODEL=true in .env to enable.

Usage (from project root):
    python -m ml.inference
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import MODELS_DIR, NUM_LABELS, LABEL_NAMES, MAX_TOKEN_LENGTH, MODEL_PATH, HF_TOKEN
from utils.logger import get_logger

logger = get_logger("ml.inference")

# ---------------------------------------------------------------------------
# Inference mode toggle
# ---------------------------------------------------------------------------
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

# HuggingFace Space used for remote inference
INFERENCE_SPACE_ID = os.getenv(
    "INFERENCE_SPACE_ID", "jayansh21/codesheriff-inference"
)

# ---------------------------------------------------------------------------
# Singleton holders
# ---------------------------------------------------------------------------
_model = None       # Local mode: transformers model
_tokenizer = None   # Local mode: transformers tokenizer
_device = None      # Local mode: torch device
_hf_client = None   # Remote mode: cached requests.Session


# ===================================================================
# Remote inference via HuggingFace Space
# ===================================================================

def _predict_remote(code_snippet: str) -> dict:
    """Call the HuggingFace Space inference server via REST API."""
    global _hf_client
    import requests

    if _hf_client is None:
        _hf_client = requests.Session()
        _hf_client.headers.update({"Authorization": f"Bearer {HF_TOKEN}"})
        logger.info(f"Using inference Space: {INFERENCE_SPACE_ID}")

    space_host = INFERENCE_SPACE_ID.replace("/", "-")
    url = f"https://{space_host}.hf.space/predict"

    resp = _hf_client.post(url, json={"code": code_snippet}, timeout=120)

    if resp.status_code == 503:
        # Space might be waking up — retry once
        import time
        logger.info("Inference Space is waking up, retrying in 30s …")
        time.sleep(30)
        resp = _hf_client.post(url, json={"code": code_snippet}, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(f"Inference Space returned {resp.status_code}: {resp.text}")

    result = resp.json()
    if isinstance(result, dict) and "label" in result:
        return result

    raise RuntimeError(f"Unexpected inference Space response: {result}")


# ===================================================================
# Local inference (opt-in via USE_LOCAL_MODEL=true)
# ===================================================================

def _load_model(model_path=None):
    """Load the fine-tuned model into memory (local mode only)."""
    global _model, _tokenizer, _device
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if model_path is None:
        candidate = MODELS_DIR / "final"
        model_path = str(candidate) if candidate.exists() else MODEL_PATH

    logger.info(f"Loading local model from: {model_path}")
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=NUM_LABELS
    )
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)
    _model.eval()
    logger.info("Local model loaded ✅")


def _predict_local(code_snippet: str) -> dict:
    """Run inference using the locally loaded model."""
    import torch

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


# ===================================================================
# Public API — same interface regardless of mode
# ===================================================================

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
    if not code_snippet or not code_snippet.strip():
        logger.warning("Empty code snippet received — returning Clean with 0 confidence.")
        return {"label": LABEL_NAMES[0], "confidence": 0.0, "label_id": 0}

    try:
        if USE_LOCAL_MODEL:
            logger.debug("Using local model inference")
            return _predict_local(code_snippet)
        else:
            logger.debug("Using HuggingFace Space inference")
            return _predict_remote(code_snippet)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return {"label": "Error", "confidence": 0.0, "label_id": -1}


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mode = "LOCAL" if USE_LOCAL_MODEL else f"HF Space ({INFERENCE_SPACE_ID})"
    print(f"Inference mode: {mode}\n")

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
