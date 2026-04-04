"""
finetune/lora_layoutlmv3_trainer.py
===================================
Fine-tunes LayoutLMv3ForTokenClassification using LoRA (PEFT) adapters.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from metrics import assign_labels_by_containment
from metrics_logger import EpochCSVLogger, generate_training_plots

logger = logging.getLogger(__name__)


def _make_lora_config(group_name: str):
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        raise RuntimeError("peft not installed. Run: pip install peft")

    r = 32 if group_name == "group_3" else 16
    return LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=r,
        lora_alpha=r * 2,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
    )


def _default_label_id(label2id: Dict[str, int]) -> int:
    for candidate in ("O", "other", "paragraph"):
        if candidate in label2id:
            return label2id[candidate]
    return next(iter(label2id.values()))


def _encode_example(example: Dict, processor, max_length: int, label2id: Dict[str, int]) -> Dict:
    image = example["image"]
    segments = example.get("segments", [])
    default_id = _default_label_id(label2id)

    try:
        enc = processor(
            image,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    except Exception as exc:
        if "tesseract" in str(exc).lower():
            raise RuntimeError(
                "LayoutLMv3 preprocessing failed because OCR could not run. "
                "Install Tesseract OCR and ensure it is on PATH."
            ) from exc
        logger.debug("Processor failed: %s", exc)
        fallback_labels = torch.full((max_length,), -100, dtype=torch.long)
        fallback_labels[0] = default_id
        return {
            "input_ids": torch.zeros(max_length, dtype=torch.long),
            "attention_mask": torch.zeros(max_length, dtype=torch.long),
            "bbox": torch.zeros((max_length, 4), dtype=torch.long),
            "pixel_values": torch.zeros((3, 224, 224)),
            "labels": fallback_labels,
        }

    word_ids = enc.word_ids(batch_index=0)
    raw_bboxes = enc["bbox"].squeeze(0).tolist()

    word_to_bbox: Dict[int, List[int]] = {}
    for tok_idx, wid in enumerate(word_ids):
        if wid is not None and wid not in word_to_bbox:
            word_to_bbox[wid] = raw_bboxes[tok_idx]

    num_words = max(word_to_bbox.keys(), default=-1) + 1
    if num_words > 0 and segments:
        ordered_bboxes = [word_to_bbox.get(i, [0, 0, 0, 0]) for i in range(num_words)]
        word_label_ids = assign_labels_by_containment(
            [tuple(b) for b in ordered_bboxes],
            segments,
            label2id=label2id,
        )
    else:
        word_label_ids = [default_id] * num_words

    token_labels: List[int] = []
    seen: set[int] = set()
    for wid in word_ids:
        if wid is None:
            token_labels.append(-100)
        elif wid in seen:
            token_labels.append(-100)
        else:
            seen.add(wid)
            token_labels.append(word_label_ids[wid] if wid < len(word_label_ids) else default_id)

    token_labels = token_labels[:max_length]
    token_labels += [-100] * (max_length - len(token_labels))
    if all(label == -100 for label in token_labels):
        token_labels[0] = default_id

    return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "bbox": enc["bbox"].squeeze(0),
        "pixel_values": enc["pixel_values"].squeeze(0),
        "labels": torch.tensor(token_labels, dtype=torch.long),
    }


def _preprocess_dataset(dataset, processor, max_length: int, label2id: Dict[str, int]):
    encoded = dataset.map(
        lambda ex: _encode_example(ex, processor, max_length, label2id),
        batched=False,
        remove_columns=dataset.column_names,
        desc="Encode LayoutLMv3 (LoRA) inputs",
    )
    encoded.set_format("torch")
    return encoded


def _log_supervision_stats(encoded_dataset, split_name: str, group_name: str, id2label: Dict[int, str]) -> None:
    label_counts: Counter = Counter()
    total_tokens = 0
    supervised_tokens = 0
    for row in encoded_dataset["labels"]:
        values = row.tolist() if hasattr(row, "tolist") else row
        for label in values:
            total_tokens += 1
            if label != -100:
                supervised_tokens += 1
                label_counts[int(label)] += 1

    ratio = supervised_tokens / max(total_tokens, 1)
    logger.info(
        "[LoRA-%s] %s supervised tokens: %d/%d (%.2f%%) | class_counts=%s",
        group_name,
        split_name,
        supervised_tokens,
        total_tokens,
        ratio * 100.0,
        dict(sorted(label_counts.items())),
    )
    if not label_counts:
        logger.warning(
            "[LoRA-%s] %s has no supervised tokens (all labels are -100).",
            group_name,
            split_name,
        )
    elif len(label_counts) == 1:
        only_label_id = next(iter(label_counts))
        label_name = id2label.get(only_label_id, str(only_label_id))
        logger.warning(
            "[LoRA-%s] %s is single-class only (%s). Metrics may look artificially perfect.",
            group_name,
            split_name,
            label_name,
        )


def train_lora_layoutlmv3(
    train_dataset,
    val_dataset,
    output_dir: Path,
    group_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> Dict:
    try:
        from peft import get_peft_model
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

    logger.info("[LoRA-%s] Loading base model microsoft/layoutlmv3-base", group_name)
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=True, use_fast=False
    )
    model_base = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    lora_cfg = _make_lora_config(group_name)
    model = get_peft_model(model_base, lora_cfg)
    model.print_trainable_parameters()

    logger.info("[LoRA-%s] Preprocessing train split ...", group_name)
    encoded_train = _preprocess_dataset(train_dataset, processor, max_length, label2id)
    _log_supervision_stats(encoded_train, "train", group_name, id2label)
    logger.info("[LoRA-%s] Preprocessing val split ...", group_name)
    encoded_val = _preprocess_dataset(val_dataset, processor, max_length, label2id)
    _log_supervision_stats(encoded_val, "validation", group_name, id2label)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1).flatten()
        labels_flat = labels.flatten()
        mask = labels_flat != -100
        preds, labels_active = preds[mask], labels_flat[mask]
        if labels_active.size == 0:
            return {"macro_f1": 0.0, "accuracy": 0.0}
        return {
            "macro_f1": round(f1_score(labels_active, preds, average="macro", zero_division=0), 4),
            "accuracy": round(float((preds == labels_active).mean()), 4),
        }

    csv_logger = EpochCSVLogger(output_dir / "training_log.csv")
    grad_accum = 8
    steps_per_epoch = max(1, int(np.ceil(len(encoded_train) / max(batch_size * grad_accum, 1))))
    warmup_steps = max(1, int(steps_per_epoch * epochs * 0.06))

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        num_train_epochs=epochs,
        bf16=torch.cuda.is_available(),
        fp16=False,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="epoch",
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
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), csv_logger],
    )

    logger.info("[LoRA-%s] Starting training ...", group_name)
    trainer.train()

    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "label_map.json").write_text(
        json.dumps({str(idx): label for idx, label in sorted(id2label.items())}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "adapter_meta.json").write_text(
        json.dumps(
            {
                "group": group_name,
                "base_model": "microsoft/layoutlmv3-base",
                "num_labels": len(label2id),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        generate_training_plots(
            output_dir,
            primary_metric="eval_macro_f1",
            higher_is_better=True,
            model_label=f"LayoutLMv3 LoRA {group_name}",
        )
    except Exception as exc:
        logger.warning("[LoRA-%s] Plot generation failed: %s", group_name, exc)

    logger.info(
        "[LoRA-%s] Done - macro_f1=%.4f  accuracy=%.4f  saved to %s",
        group_name,
        metrics.get("eval_macro_f1", 0),
        metrics.get("eval_accuracy", 0),
        output_dir,
    )
    return metrics