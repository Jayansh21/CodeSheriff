"""
CodeSheriff — Model Evaluation Module

Loads the fine-tuned model and prints a full classification report,
per-class precision/recall/F1, and generates a confusion matrix.

Usage (from project root):
    python -m ml.evaluate
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import (
    SEED,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    NUM_LABELS,
    LABEL_NAMES,
    MAX_TOKEN_LENGTH,
    BATCH_SIZE,
)
from utils.logger import get_logger

logger = get_logger("ml.evaluate")

torch.manual_seed(SEED)
np.random.seed(SEED)


def evaluate() -> str:
    """Run evaluation on the validation set and return a report string."""

    model_dir = MODELS_DIR / "final"

    if not model_dir.exists():
        logger.error(f"Model not found at {model_dir}. Run training first.")
        sys.exit(1)

    # ---- load data ---------------------------------------------------------
    test_csv = DATA_PROCESSED_DIR / "test.csv"

    if test_csv.exists():
        # Prefer dedicated test split from balanced pipeline
        test_df = pd.read_csv(test_csv)
        val_codes = test_df["code"].astype(str).tolist()
        val_labels = test_df["label"].astype(int).tolist()
        logger.info(f"Loaded {len(val_codes):,} test samples from {test_csv}")
    else:
        csv_path = DATA_PROCESSED_DIR / "labeled_dataset.csv"
        if not csv_path.exists():
            logger.error(f"Dataset not found at {csv_path}.")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        codes = df["code"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()
        _, val_codes, _, val_labels = train_test_split(
            codes, labels, test_size=0.2, random_state=SEED, stratify=labels
        )
        logger.info(f"Loaded {len(val_codes):,} val samples from {csv_path} (80/20 split)")

    # ---- load model --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=NUM_LABELS
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ---- inference on val set ---------------------------------------------
    all_preds = []
    batch_size = BATCH_SIZE

    for i in range(0, len(val_codes), batch_size):
        batch_codes = val_codes[i : i + batch_size]
        encoding = tokenizer(
            batch_codes,
            truncation=True,
            padding="max_length",
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)

    # ---- metrics -----------------------------------------------------------
    target_names = [LABEL_NAMES[i] for i in range(NUM_LABELS)]
    report = classification_report(
        val_labels, all_preds, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(val_labels, all_preds)

    full_report = (
        "=" * 60 + "\n"
        "CodeSheriff — Evaluation Report\n"
        "=" * 60 + "\n\n"
        f"Samples evaluated: {len(val_labels)}\n\n"
        "Classification Report:\n"
        f"{report}\n\n"
        "Confusion Matrix:\n"
        f"{cm}\n"
        "=" * 60 + "\n"
    )

    logger.info("\n" + full_report)

    # ---- persist report ----------------------------------------------------
    report_path = MODELS_DIR / "evaluation_report.txt"
    report_path.write_text(full_report, encoding="utf-8")
    logger.info(f"Report saved to {report_path}")

    return full_report


if __name__ == "__main__":
    evaluate()
