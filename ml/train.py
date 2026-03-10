"""
CodeSheriff — Model Training Module

Fine-tunes microsoft/codebert-base for 5-class sequence classification
(bug-type detection) on the prepared dataset.

Supports:
    - Automatic GPU/CPU device selection
    - Mixed precision (fp16) on CUDA
    - Gradient accumulation
    - Class-weighted loss for imbalanced labels
    - Linear warmup + decay LR schedule
    - Gradient clipping
    - Early stopping on validation F1
    - Automatic batch-size reduction on CUDA OOM

Usage (from project root):
    python -m ml.train
"""

import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import (
    SEED,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    NUM_LABELS,
    MAX_TOKEN_LENGTH,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_EPOCHS,
    LEARNING_RATE,
)
from utils.logger import get_logger

logger = get_logger("ml.train")

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class CodeDataset(Dataset):
    """Simple dataset for tokenised code snippets."""

    def __init__(self, codes, labels, tokenizer, max_length=MAX_TOKEN_LENGTH):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.codes[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    """Select GPU if available, else CPU.  Log the chosen device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_mb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024 ** 2)
        logger.info(f"Device: cuda ({gpu_name}, {vram_mb:.0f} MB VRAM)")
    else:
        device = torch.device("cpu")
        logger.info("Device: cpu  (CUDA not available — training will be slower)")
    return device


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train() -> None:
    """Run the full training pipeline."""

    # ---- load data --------------------------------------------------------
    train_csv = DATA_PROCESSED_DIR / "train.csv"
    val_csv = DATA_PROCESSED_DIR / "val.csv"

    if train_csv.exists() and val_csv.exists():
        # Prefer pre-split files produced by `python -m ml.dataset --balanced`
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        train_codes = train_df["code"].astype(str).tolist()
        train_labels = train_df["label"].astype(int).tolist()
        val_codes = val_df["code"].astype(str).tolist()
        val_labels = val_df["label"].astype(int).tolist()
        logger.info(f"Loaded pre-split data  →  Train: {len(train_codes):,}  |  Val: {len(val_codes):,}")
    else:
        csv_path = DATA_PROCESSED_DIR / "labeled_dataset.csv"
        if not csv_path.exists():
            logger.error(f"Dataset not found. Run `python -m ml.dataset` first.")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df):,} samples from {csv_path}")
        codes = df["code"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()
        train_codes, val_codes, train_labels, val_labels = train_test_split(
            codes, labels, test_size=0.2, random_state=SEED, stratify=labels
        )
        logger.info(f"Train: {len(train_codes):,}  |  Val: {len(val_codes):,}")

    # ---- tokenizer & model -----------------------------------------------
    model_name = "microsoft/codebert-base"
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    )

    device = _select_device()
    model.to(device)

    use_fp16 = device.type == "cuda"
    if use_fp16:
        logger.info("Mixed precision (fp16) enabled")

    # ---- dataloaders ------------------------------------------------------
    batch_size = BATCH_SIZE
    accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    effective_batch = batch_size * accumulation_steps
    logger.info(
        f"Batch size: {batch_size}  |  Gradient accumulation: {accumulation_steps}  |  "
        f"Effective batch size: {effective_batch}"
    )

    train_ds = CodeDataset(train_codes, train_labels, tokenizer)
    val_ds = CodeDataset(val_codes, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---- class weights (handle imbalanced labels) -------------------------
    label_counts = Counter(train_labels)
    total_train = len(train_labels)
    class_weights = torch.tensor(
        [total_train / (NUM_LABELS * label_counts.get(i, 1)) for i in range(NUM_LABELS)],
        dtype=torch.float,
    ).to(device)
    logger.info(f"Class weights: {[round(w, 3) for w in class_weights.tolist()]}")

    # ---- optimiser --------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ---- learning rate scheduler (linear warmup + decay) ------------------
    # Account for gradient accumulation in total optimizer steps
    steps_per_epoch = len(train_loader) // accumulation_steps
    total_optim_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = total_optim_steps // 10  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps
    )
    logger.info(
        f"Scheduler: {warmup_steps} warmup steps, {total_optim_steps} total optimizer steps"
    )

    # ---- mixed-precision scaler ------------------------------------------
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # ---- training ---------------------------------------------------------
    if device.type == "cpu":
        logger.warning(
            "⚠️  Training on CPU will be slow (3-6 hours).  "
            "Press Ctrl+C to stop early (last checkpoint will be preserved)."
        )

    logger.info("Training started")
    logger.info("=" * 70)

    best_val_f1 = 0.0
    patience = 2
    epochs_without_improvement = 0
    total_samples_all_epochs = 0
    total_steps_all_epochs = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        all_preds, all_labels_epoch = [], []
        start = time.time()
        optimizer.zero_grad()
        epoch_samples_processed = 0
        epoch_steps_ok = 0
        epoch_steps_failed = 0
        total_steps_in_epoch = len(train_loader)
        total_samples_in_epoch = len(train_ds)

        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} — {total_steps_in_epoch} steps, {total_samples_in_epoch} samples")
        logger.info("-" * 70)

        for step, batch in enumerate(train_loader, 1):
            step_start = time.time()
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["labels"].to(device)
                current_batch_size = input_ids.size(0)

                with torch.amp.autocast("cuda", enabled=use_fp16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, batch_labels)
                    loss = loss / accumulation_steps  # scale for accumulation

                scaler.scale(loss).backward()

                # Optimizer step every accumulation_steps (or at end of epoch)
                if step % accumulation_steps == 0 or step == total_steps_in_epoch:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                # Track metrics (un-scaled loss)
                step_loss = loss.item() * accumulation_steps
                epoch_loss += step_loss
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_epoch.extend(batch_labels.cpu().numpy())

                epoch_samples_processed += current_batch_size
                epoch_steps_ok += 1
                total_samples_all_epochs += current_batch_size
                total_steps_all_epochs += 1

                # Progress log every 10 steps
                if step % 10 == 0 or step == total_steps_in_epoch:
                    pct = epoch_samples_processed / total_samples_in_epoch * 100
                    elapsed = time.time() - start
                    avg_step_time = elapsed / step
                    remaining_steps = total_steps_in_epoch - step
                    eta_sec = remaining_steps * avg_step_time
                    eta_min = eta_sec / 60
                    running_loss = epoch_loss / step
                    correct = sum(p == l for p, l in zip(all_preds, all_labels_epoch))
                    running_acc = correct / len(all_preds) * 100

                    logger.info(
                        f"  Step {step:>4}/{total_steps_in_epoch} | "
                        f"Processed: {epoch_samples_processed:>5}/{total_samples_in_epoch} ({pct:5.1f}%) | "
                        f"Loss: {step_loss:.4f} (avg: {running_loss:.4f}) | "
                        f"Acc: {running_acc:5.1f}% | "
                        f"ETA: {eta_min:.1f}min"
                    )

            except torch.cuda.OutOfMemoryError:
                # --- CUDA OOM recovery: halve batch size and rebuild loader ---
                epoch_steps_failed += 1
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    f"CUDA OOM at epoch {epoch}, step {step}. "
                    f"Reducing batch size to {batch_size} and restarting epoch."
                )
                logger.info(
                    f"  Epoch stats before OOM — Processed: {epoch_samples_processed} | "
                    f"OK: {epoch_steps_ok} | Failed: {epoch_steps_failed}"
                )
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

                # Recompute scheduler for new loader length
                steps_per_epoch = len(train_loader) // accumulation_steps
                remaining_epochs = NUM_EPOCHS - epoch + 1
                remaining_steps = steps_per_epoch * remaining_epochs
                warmup_steps = remaining_steps // 10
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=remaining_steps,
                )
                break  # restart the epoch with the smaller batch size
        else:
            # for/else: only runs if the inner loop completed without break
            avg_loss = epoch_loss / total_steps_in_epoch
            acc = accuracy_score(all_labels_epoch, all_preds)
            f1 = f1_score(all_labels_epoch, all_preds, average="weighted", zero_division=0)
            elapsed = time.time() - start

            logger.info("-" * 70)
            logger.info(
                f"Epoch {epoch} COMPLETE | Samples: {epoch_samples_processed} | "
                f"Steps OK: {epoch_steps_ok} | Failed: {epoch_steps_failed}"
            )
            logger.info(
                f"  Train Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | "
                f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)"
            )

            # Checkpoint
            ckpt_dir = MODELS_DIR / f"checkpoint-epoch-{epoch}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"  Checkpoint saved → {ckpt_dir}")

            # ---- validation -----------------------------------------------
            logger.info(f"  Running validation on {len(val_ds)} samples …")
            model.eval()
            val_preds, val_labels_all = [], []
            val_loss_total = 0.0
            val_steps = len(val_loader)
            with torch.no_grad():
                for vi, batch in enumerate(val_loader, 1):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    batch_labels = batch["labels"].to(device)

                    with torch.amp.autocast("cuda", enabled=use_fp16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        v_loss = loss_fn(outputs.logits, batch_labels)

                    val_loss_total += v_loss.item()
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels_all.extend(batch_labels.cpu().numpy())

                    if vi % 20 == 0 or vi == val_steps:
                        vpct = vi / val_steps * 100
                        logger.info(
                            f"    Val step {vi:>4}/{val_steps} ({vpct:5.1f}%) | "
                            f"Samples: {len(val_preds)}/{len(val_ds)}"
                        )

            avg_val_loss = val_loss_total / val_steps
            val_acc = accuracy_score(val_labels_all, val_preds)
            val_f1 = f1_score(val_labels_all, val_preds, average="weighted", zero_division=0)
            logger.info(
                f"  Validation — Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
            )

            # Save best model based on VALIDATION F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                best_dir = MODELS_DIR / "final"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                logger.info(f"  ✅ NEW BEST MODEL (val F1={best_val_f1:.4f}) saved → {best_dir}")
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"  ⚠️  No improvement for {epochs_without_improvement} epoch(s) "
                    f"(best val F1={best_val_f1:.4f})"
                )
                if epochs_without_improvement >= patience:
                    logger.info(
                        f"  🛑 Early stopping triggered after {epoch} epochs "
                        f"(patience={patience})."
                    )
                    break

            logger.info("=" * 70)
            continue
        # OOM break landed here — the epoch will be re-attempted by the outer loop
        continue

    # ---- final summary ----------------------------------------------------
    logger.info("=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info(f"  Total samples processed: {total_samples_all_epochs:,}")
    logger.info(f"  Total steps completed:   {total_steps_all_epochs:,}")
    logger.info(f"  Best validation F1:      {best_val_f1:.4f}")
    logger.info(f"  Model saved to:          {MODELS_DIR / 'final'}")
    logger.info("Training complete ✅")
    logger.info("=" * 70)


if __name__ == "__main__":
    train()
