"""
Push the trained CodeSheriff model to HuggingFace Hub.

Prerequisites:
  1. Run: huggingface-cli login
  2. Ensure MODEL_PATH points to your local trained model
  3. Set HF_REPO_NAME env var or edit the default below

Usage:
  python scripts/push_model_to_hub.py
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.config import HF_TOKEN
from utils.logger import get_logger

logger = get_logger("scripts.push_model_to_hub")

LOCAL_MODEL_PATH = os.getenv("MODEL_PATH", "models/codesheriff-model/final")
HF_REPO_NAME = os.getenv("HF_REPO_NAME", "jayansh21/codesheriff-bug-classifier")


def push_model():
    if not HF_TOKEN or HF_TOKEN == "your_huggingface_token_here":
        logger.error("HF_TOKEN is not set. Add your HuggingFace write token to .env")
        logger.error("Get one at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    logger.info(f"Loading model from {LOCAL_MODEL_PATH}")

    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.error(f"Model not found at {LOCAL_MODEL_PATH}")
        logger.error("Run `python -m ml.train` first to train the model.")
        sys.exit(1)

    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

    logger.info(f"Pushing to HuggingFace Hub: {HF_REPO_NAME}")
    model.push_to_hub(HF_REPO_NAME, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_NAME, token=HF_TOKEN)

    logger.info("Model pushed successfully!")
    logger.info(f"Update MODEL_PATH in your .env to: {HF_REPO_NAME}")
    logger.info(f"Update MODEL_PATH in Render dashboard to: {HF_REPO_NAME}")


if __name__ == "__main__":
    push_model()
