"""
Fine-tune LayoutLMv3 for token classification on NDA/contract layout detection.

Runs with hardcoded config. Automatically resumes from checkpoint_last if it exists.

Label set:
    0=paragraph  1=heading  2=list_item  3=table  4=caption  5=other
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------
LABEL2ID = {
    "paragraph": 0,
    "heading":   1,
    "list_item": 2,
    "table":     3,
    "caption":   4,
    "other":     5,
}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_dataset_from_disk(dataset_dir: str):
    """Load a datasets.DatasetDict from disk."""
    from datasets import load_from_disk
    dsd = load_from_disk(dataset_dir)
    logger.info(f"Loaded dataset from {dataset_dir}")
    logger.info(f"  Train: {len(dsd['train'])}  Val: {len(dsd['validation'])}")
    return dsd


def tokenize_and_align(examples, processor, max_length: int = 512):
    """
    Tokenise words+boxes and propagate word-level labels to sub-word tokens.

    LayoutLMv3Processor handles the image encoding as well.
    Labels for sub-word tokens that are *not* the first sub-word of a word
    are set to -100 so they are ignored by the cross-entropy loss.
    """
    images = examples["image"]
    words  = examples["words"]
    boxes  = examples["boxes"]
    labels = examples["labels"]

    encoding = processor(
        images,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,  # tells us which word each token came from
    )

    # word_ids() maps each token position → word index (or None for special tokens)
    aligned_labels = []
    for batch_idx in range(len(labels)):
        word_ids   = encoding.word_ids(batch_index=batch_idx)
        word_labels = labels[batch_idx]
        token_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                token_labels.append(-100)           # special token
            elif word_id != prev_word_id:
                # First sub-word of this word → use the real label
                label_val = word_labels[word_id] if word_id < len(word_labels) else -100
                token_labels.append(label_val)
            else:
                token_labels.append(-100)           # subsequent sub-words → ignore
            prev_word_id = word_id
        aligned_labels.append(token_labels)

    encoding["labels"] = aligned_labels
    encoding.pop("offset_mapping", None)   # not needed by the model
    return encoding


def prepare_datasets(dataset_dir: str, processor, max_length: int = 512):
    """Load, tokenise, and return train/val datasets as PyTorch-ready objects."""
    raw = load_dataset_from_disk(dataset_dir)

    tokenised = raw.map(
        lambda ex: tokenize_and_align(ex, processor, max_length),
        batched=True,
        batch_size=8,
        remove_columns=raw["train"].column_names,
        desc="Tokenising",
    )
    tokenised.set_format("torch")
    return tokenised


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions_and_labels, id2label=ID2LABEL):
    """
    Compute token-level precision, recall, F1, accuracy.
    Ignores positions where the label is -100.
    """
    predictions, labels = predictions_and_labels
    # predictions shape: (N, seq_len, num_labels)
    preds_flat  = np.argmax(predictions, axis=2).flatten()
    labels_flat = labels.flatten()

    mask        = labels_flat != -100
    preds_flat  = preds_flat[mask]
    labels_flat = labels_flat[mask]

    # Per-class counts
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, true in zip(preds_flat, labels_flat):
        if pred == true:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    # Macro F1
    f1s = []
    for label_id in range(NUM_LABELS):
        p = tp[label_id] / (tp[label_id] + fp[label_id] + 1e-9)
        r = tp[label_id] / (tp[label_id] + fn[label_id] + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        f1s.append(f1)

    accuracy = (preds_flat == labels_flat).mean()
    macro_f1 = float(np.mean(f1s))

    # Per-class report
    per_class = {
        id2label[i]: round(f1s[i], 4) for i in range(NUM_LABELS)
    }

    return {
        "accuracy":  round(float(accuracy), 4),
        "macro_f1":  round(macro_f1, 4),
        "per_class_f1": per_class,
    }


# ---------------------------------------------------------------------------
# Training configuration (hardcoded)
# ---------------------------------------------------------------------------

BASE_MODEL = "microsoft/layoutlmv3-base"
DATASET_DIR = "data/layoutlm_dataset"
OUTPUT_DIR = "models/layoutlmv3-nda"
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MAX_LENGTH = 384
DEVICE = "auto"  # "auto", "cpu", "cuda", "mps"

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    from transformers import (
        LayoutLMv3Processor,
        LayoutLMv3ForTokenClassification,
        get_linear_schedule_with_warmup,
    )

    # ---- Device ----
    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    # ---- Processor ----
    logger.info(f"Loading processor: {BASE_MODEL}")
    processor = LayoutLMv3Processor.from_pretrained(
        BASE_MODEL,
        apply_ocr=False,
    )

    # ---- Dataset ----
    tokenised = prepare_datasets(DATASET_DIR, processor, max_length=MAX_LENGTH)

    train_loader = DataLoader(
        tokenised["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        tokenised["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ---- Model ----
    logger.info("Initialising LayoutLMv3ForTokenClassification")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # ← silences the classifier MISSING warning
    )
    model.to(device)

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = max(1, total_steps // 10)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ---- Output dir ----
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1    = 0.0
    best_epoch = 0

    # ---- Resume (automatic) ----
    start_epoch = 0
    checkpoint_path = output_dir / "checkpoint_last"
    if checkpoint_path.exists():
        start_epoch_file = checkpoint_path / "epoch.txt"
        if start_epoch_file.exists():
            completed_epoch = int(start_epoch_file.read_text().strip())
            start_epoch = completed_epoch + 1
            if start_epoch >= EPOCHS:
                logger.info(f"Training already complete (last epoch: {completed_epoch}). Restarting from epoch 0.")
                start_epoch = 0
            else:
                logger.info(f"Resuming from {checkpoint_path} (last epoch: {completed_epoch})")
                model = LayoutLMv3ForTokenClassification.from_pretrained(str(checkpoint_path))
                model.to(device)
        else:
            logger.info(f"Resuming from {checkpoint_path} (epoch info not found)")
            model = LayoutLMv3ForTokenClassification.from_pretrained(str(checkpoint_path))
            model.to(device)
            start_epoch = 0

    logger.info(
        f"Training: epochs={EPOCHS}  batch={BATCH_SIZE}  "
        f"lr={LEARNING_RATE}  warmup={warmup_steps}  total_steps={total_steps}  start_epoch={start_epoch}"
    )

    history = []

    for epoch in trange(start_epoch, EPOCHS, desc="Epochs"):
        # ---- Train epoch ----
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss    = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ---- Validation ----
        model.eval()
        all_preds  = []
        all_labels = []
        val_loss   = 0.0
        n_val      = 0

        with torch.no_grad():
            for batch in val_loader:
                batch   = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss   += outputs.loss.item()
                n_val      += 1

                logits = outputs.logits.cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.append(logits)
                all_labels.append(labels)

        avg_val_loss = val_loss / max(n_val, 1)
        preds_concat  = np.concatenate(all_preds,  axis=0)
        labels_concat = np.concatenate(all_labels, axis=0)
        metrics = compute_metrics((preds_concat, labels_concat))

        logger.info(
            f"Epoch {epoch+1}/{EPOCHS}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"accuracy={metrics['accuracy']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
        for cls, f1 in metrics["per_class_f1"].items():
            logger.info(f"    {cls:12s}  f1={f1:.4f}")

        row = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 5),
            "val_loss":   round(avg_val_loss, 5),
            **metrics,
        }
        history.append(row)

        # ---- Save last checkpoint ----
        # model.save_pretrained(str(checkpoint_path))
        # processor.save_pretrained(str(checkpoint_path))
        # (checkpoint_path / "epoch.txt").write_text(str(epoch))

        # ---- Save best checkpoint ----
        if metrics["macro_f1"] >= best_f1:
            best_f1    = metrics["macro_f1"]
            best_epoch = epoch + 1
            best_path  = output_dir / "checkpoint_best"
            model.save_pretrained(str(best_path))
            processor.save_pretrained(str(best_path))
            logger.info(f"  ✓ New best model saved (f1={best_f1:.4f})")

    # ---- Save training history ----
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history → {history_path}")

    logger.info(
        f"\nTraining complete. Best epoch: {best_epoch}  Best macro F1: {best_f1:.4f}"
    )
    logger.info(f"Best model: {output_dir / 'checkpoint_best'}")
    return history


if __name__ == "__main__":
    train()