"""
finetune/lora_layoutlmv3_trainer.py
====================================
Fine-tunes LayoutLMv3ForTokenClassification using LoRA (PEFT) adapters.

Key differences from layoutlmv3_trainer.py
-------------------------------------------
- Base model weights are FROZEN; only adapter matrices are trained.
- LLRD is removed (its rationale — protecting lower layers — does not apply
  when those layers are frozen). A flat AdamW with cosine schedule is used.
- Each adapter group is trained independently and saved to its own subdir
  under `output_root / group_name /`.
- Adapter checkpoints are tiny (~8-15 MB each vs ~500 MB for a full model).

LoRA configuration
------------------
- target_modules : ["query", "value"]  — standard for encoder-only transformers
- r=16, lora_alpha=32                  — good default; increase r for Group 3
  (reasoning-heavy) if macro_f1 plateaus
- lora_dropout=0.1
- task_type=TOKEN_CLS

Requirements
------------
    pip install peft
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from config import ID2LABEL, LABEL2ID, NUM_LABELS
from metrics import assign_labels_by_containment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA config factory — Group 3 gets a larger rank for complex reasoning tasks
# ---------------------------------------------------------------------------

def _make_lora_config(group_name: str):
    """Return a PEFT LoraConfig tuned for the adapter group."""
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        raise RuntimeError("peft not installed. Run: pip install peft")

    # Group 3 (reasoning / NDA) benefits from a slightly larger rank
    r = 32 if group_name == "group_3" else 16
    return LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=r,
        lora_alpha=r * 2,          # standard: alpha = 2 * r
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
    )


# ---------------------------------------------------------------------------
# Token-classification preprocessing  (shared with layoutlmv3_trainer.py)
# ---------------------------------------------------------------------------

def _encode_example(example: Dict, processor, max_length: int) -> Dict:
    image    = example["image"]
    segments = example.get("segments", [])

    try:
        enc = processor(
            image,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    except Exception as exc:
        logger.debug("Processor failed: %s", exc)
        return {
            "input_ids":      torch.zeros(max_length, dtype=torch.long),
            "attention_mask": torch.zeros(max_length, dtype=torch.long),
            "bbox":           torch.zeros((max_length, 4), dtype=torch.long),
            "pixel_values":   torch.zeros((3, 224, 224)),
            "labels":         torch.full((max_length,), -100, dtype=torch.long),
        }

    word_ids   = enc.word_ids(batch_index=0)
    raw_bboxes = enc["bbox"].squeeze(0).tolist()

    word_to_bbox: Dict[int, List[int]] = {}
    for tok_idx, wid in enumerate(word_ids):
        if wid is not None and wid not in word_to_bbox:
            word_to_bbox[wid] = raw_bboxes[tok_idx]

    num_words = max(word_to_bbox.keys(), default=-1) + 1
    if num_words > 0 and segments:
        ordered_bboxes = [word_to_bbox.get(i, [0, 0, 0, 0]) for i in range(num_words)]
        word_label_ids = assign_labels_by_containment(
            [tuple(b) for b in ordered_bboxes], segments
        )
    else:
        word_label_ids = [LABEL2ID["paragraph"]] * num_words

    token_labels: List[int] = []
    seen: set = set()
    for wid in word_ids:
        if wid is None:
            token_labels.append(-100)
        elif wid in seen:
            token_labels.append(-100)
        else:
            seen.add(wid)
            token_labels.append(
                word_label_ids[wid] if wid < len(word_label_ids) else LABEL2ID["other"]
            )

    token_labels = token_labels[:max_length]
    token_labels += [-100] * (max_length - len(token_labels))

    return {
        "input_ids":      enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "bbox":           enc["bbox"].squeeze(0),
        "pixel_values":   enc["pixel_values"].squeeze(0),
        "labels":         torch.tensor(token_labels, dtype=torch.long),
    }


def _preprocess_dataset(dataset, processor, max_length: int):
    encoded = dataset.map(
        lambda ex: _encode_example(ex, processor, max_length),
        batched=False,
        remove_columns=dataset.column_names,
        desc="Encode LayoutLMv3 (LoRA) inputs",
    )
    encoded.set_format("torch")
    return encoded


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_lora_layoutlmv3(
    train_dataset,
    val_dataset,
    output_dir: Path,
    group_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> Dict:
    """
    Fine-tune LayoutLMv3 with LoRA for one adapter group.

    Parameters
    ----------
    train_dataset, val_dataset : HuggingFace datasets
    output_dir   : where to save the adapter (e.g. models/adapters/group_1)
    group_name   : "group_1" | "group_2" | "group_3"
    """
    try:
        from peft import get_peft_model, PeftModel
    except ImportError:
        raise RuntimeError("peft not installed. Run: pip install peft")

    from sklearn.metrics import f1_score
    from transformers import (
        EarlyStoppingCallback,
        LayoutLMv3ForTokenClassification,
        LayoutLMv3Processor,
        Trainer,
        TrainingArguments,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load base model + wrap with LoRA
    # ------------------------------------------------------------------
    logger.info("[LoRA-%s] Loading base model microsoft/layoutlmv3-base", group_name)
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=True
    )
    base_model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    lora_cfg = _make_lora_config(group_name)
    model    = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()   # logs e.g. "trainable params: 2,097,152 || ..."

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    logger.info("[LoRA-%s] Preprocessing train split …", group_name)
    encoded_train = _preprocess_dataset(train_dataset, processor, max_length)
    logger.info("[LoRA-%s] Preprocessing val split …", group_name)
    encoded_val   = _preprocess_dataset(val_dataset,   processor, max_length)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds  = np.argmax(logits, axis=-1).flatten()
        labels = labels.flatten()
        mask   = labels != -100
        preds, labels = preds[mask], labels[mask]
        return {
            "macro_f1": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
            "accuracy": round(float((preds == labels).mean()), 4),
        }

    # ------------------------------------------------------------------
    # Training args  — flat AdamW, no LLRD needed (base is frozen)
    # ------------------------------------------------------------------
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        num_train_epochs=epochs,
        bf16=torch.cuda.is_available(),
        fp16=False,
        weight_decay=0.01,
        max_grad_norm=1.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        remove_unused_columns=False,
        logging_steps=25,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("[LoRA-%s] Starting training …", group_name)
    trainer.train()

    # ------------------------------------------------------------------
    # Save adapter only (NOT the full model — keeps checkpoint tiny)
    # ------------------------------------------------------------------
    model.save_pretrained(str(output_dir))          # saves adapter_config.json + adapter weights
    processor.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "label_map.json").write_text(
        json.dumps(ID2LABEL, indent=2), encoding="utf-8"
    )
    # Record which group this adapter was trained for
    (output_dir / "adapter_meta.json").write_text(
        json.dumps({"group": group_name, "base_model": "microsoft/layoutlmv3-base"}, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "[LoRA-%s] Done — macro_f1=%.4f  accuracy=%.4f  saved to %s",
        group_name,
        metrics.get("eval_macro_f1", 0),
        metrics.get("eval_accuracy", 0),
        output_dir,
    )
    return metrics