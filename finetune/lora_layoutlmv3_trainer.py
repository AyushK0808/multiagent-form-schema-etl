"""
finetune/lora_layoutlmv3_trainer.py
===================================
Fine-tunes LayoutLMv3ForTokenClassification using LoRA (PEFT) adapters.

Two encoding paths:
  - Default  : segment-containment assignment from our normalised dataset.
  - HF-native: uses pre-tokenized HF datasets (nielsr/funsd-layoutlmv3,
               nielsr/cord-layoutlmv3) that already carry words/bboxes/ner_tags,
               bypassing OCR and containment assignment entirely.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from dataset_mappings.funsd import FUNSD_BIO_LABELS
from metrics import assign_labels_by_containment
from metrics_logger import EpochCSVLogger, generate_training_plots

try:
    from transformers import Trainer
except ImportError:
    Trainer = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


_WORD_LEVEL_LABEL_SETS = {
    frozenset(FUNSD_BIO_LABELS),
}


def _is_word_level(segments: list) -> bool:
    """Return True when segments carry per-word BIO labels (one segment = one word)."""
    if not segments:
        return False
    labels = {seg.get("label", "") for seg in segments[:20]}
    return any(labels <= s for s in _WORD_LEVEL_LABEL_SETS) or any(
        str(lbl).startswith(("B-", "I-")) for lbl in labels
    )


# HF repo IDs for the pre-tokenized LayoutLMv3 datasets
_HF_NATIVE_REPOS: Dict[str, str] = {
    "CORD": "nielsr/cord-layoutlmv3",
}

# nielsr/cord-layoutlmv3 uses CORD BIO entity tags.
# Map them to our 6-class layout label space.
_CORD_NER_TO_LAYOUT: Dict[str, str] = {
    # Store / header info
    "B-STORE_INFO.STORE_NAME":  "heading",
    "I-STORE_INFO.STORE_NAME":  "heading",
    "B-STORE_INFO.STORE_ADDR":  "heading",
    "I-STORE_INFO.STORE_ADDR":  "heading",
    "B-STORE_INFO.BRANCH_NM":   "heading",
    "I-STORE_INFO.BRANCH_NM":   "heading",
    "B-STORE_INFO.STORE_TEL":   "heading",
    "I-STORE_INFO.STORE_TEL":   "heading",
    "B-STORE_INFO.STORE_EMAIL": "heading",
    "I-STORE_INFO.STORE_EMAIL": "heading",
    "B-STORE_INFO.STORE_ETC":   "heading",
    "I-STORE_INFO.STORE_ETC":   "heading",
    # Menu line items
    "B-MENU.NM":           "list_item",
    "I-MENU.NM":           "list_item",
    "B-MENU.NUM":          "list_item",
    "I-MENU.NUM":          "list_item",
    "B-MENU.UNITPRICE":    "list_item",
    "I-MENU.UNITPRICE":    "list_item",
    "B-MENU.CNT":          "list_item",
    "I-MENU.CNT":          "list_item",
    "B-MENU.DISCOUNTPRICE": "list_item",
    "I-MENU.DISCOUNTPRICE": "list_item",
    "B-MENU.PRICE":        "list_item",
    "I-MENU.PRICE":        "list_item",
    "B-MENU.ITEMSUBTOTAL": "list_item",
    "I-MENU.ITEMSUBTOTAL": "list_item",
    "B-MENU.VATYN":        "list_item",
    "I-MENU.VATYN":        "list_item",
    "B-MENU.ETC":          "list_item",
    "I-MENU.ETC":          "list_item",
    "B-MENU.SUB_NM":       "list_item",
    "I-MENU.SUB_NM":       "list_item",
    # Sub-totals
    "B-SUB_TOTAL.SUBTOTAL_PRICE":  "list_item",
    "I-SUB_TOTAL.SUBTOTAL_PRICE":  "list_item",
    "B-SUB_TOTAL.DISCOUNT_PRICE":  "list_item",
    "I-SUB_TOTAL.DISCOUNT_PRICE":  "list_item",
    "B-SUB_TOTAL.SERVICE_PRICE":   "list_item",
    "I-SUB_TOTAL.SERVICE_PRICE":   "list_item",
    "B-SUB_TOTAL.TAX_PRICE":       "list_item",
    "I-SUB_TOTAL.TAX_PRICE":       "list_item",
    "B-SUB_TOTAL.ETC":             "list_item",
    "I-SUB_TOTAL.ETC":             "list_item",
    # Total
    "B-TOTAL.TOTAL_PRICE":         "list_item",
    "I-TOTAL.TOTAL_PRICE":         "list_item",
    "B-TOTAL.TOTAL_ETC":           "list_item",
    "I-TOTAL.TOTAL_ETC":           "list_item",
    "B-TOTAL.CASHPRICE":           "list_item",
    "I-TOTAL.CASHPRICE":           "list_item",
    "B-TOTAL.CHANGEPRICE":         "list_item",
    "I-TOTAL.CHANGEPRICE":         "list_item",
    "B-TOTAL.CREDITCARDPRICE":     "list_item",
    "I-TOTAL.CREDITCARDPRICE":     "list_item",
    "B-TOTAL.EMONEYPRICE":         "list_item",
    "I-TOTAL.EMONEYPRICE":         "list_item",
    # Payment / meta
    "B-PAYMENT_INFO.CARD_COMPANY": "other",
    "I-PAYMENT_INFO.CARD_COMPANY": "other",
    "B-PAYMENT_INFO.CARD_NUMBER":  "other",
    "I-PAYMENT_INFO.CARD_NUMBER":  "other",
    "O": "paragraph",
}


def _cord_ner_to_layout(tag_name: str) -> str:
    """Map a CORD BIO NER tag to our 6-class layout label.

    Handles exact matches first, then falls back to prefix matching on
    the entity group (STORE_INFO → heading, MENU/SUB_TOTAL/TOTAL → list_item,
    PAYMENT_INFO → other) so unknown sub-fields degrade gracefully.
    """
    if tag_name in _CORD_NER_TO_LAYOUT:
        return _CORD_NER_TO_LAYOUT[tag_name]
    upper = tag_name.upper()
    if "STORE_INFO" in upper:
        return "heading"
    if any(g in upper for g in ("MENU", "SUB_TOTAL", "TOTAL")):
        return "list_item"
    if "PAYMENT" in upper:
        return "other"
    return "paragraph"


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
        if _is_word_level(segments):
            # Word-level datasets (FUNSD): segment index == word index.
            # Skip spatial containment — assign label directly by position.
            word_label_ids = [
                label2id.get(segments[i]["label"], default_id) if i < len(segments) else default_id
                for i in range(num_words)
            ]
        else:
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


# ---------------------------------------------------------------------------
# HF-native path (nielsr/funsd-layoutlmv3, nielsr/cord-layoutlmv3)
# ---------------------------------------------------------------------------

def _load_hf_native(dataset_name: str, max_train: Optional[int], max_val: Optional[int]):
    """Load a pre-tokenized LayoutLMv3 dataset from HuggingFace."""
    from datasets import load_dataset
    repo = _HF_NATIVE_REPOS.get(dataset_name)
    if repo is None:
        raise ValueError(f"No HF-native repo registered for '{dataset_name}'. "
                         f"Available: {list(_HF_NATIVE_REPOS)}")
    ds = load_dataset(repo)
    train_split = ds.get("train") or ds[next(iter(ds))]
    val_split   = ds.get("test") or ds.get("validation") or train_split
    if max_train:
        train_split = train_split.select(range(min(max_train, len(train_split))))
    if max_val:
        val_split = val_split.select(range(min(max_val, len(val_split))))
    return train_split, val_split


def _label_list_from_hf_native(dataset) -> List[str]:
    """Extract the ordered label list from a ClassLabel feature or from data."""
    from datasets import ClassLabel, Sequence
    features = dataset.features
    # Try ner_tags first, then ner_tag, then tags
    for col in ("ner_tags", "ner_tag", "tags"):
        if col not in features:
            continue
        feat = features[col]
        if isinstance(feat, Sequence) and isinstance(feat.feature, ClassLabel):
            return feat.feature.names
    # Fallback: collect unique string labels from the data
    seen = set()
    for row in dataset["ner_tags"]:
        seen.update(str(t) for t in row)
    return sorted(seen)


def _encode_hf_native_example(
    example: Dict,
    processor,
    max_length: int,
    label2id: Dict[str, int],
    label_list: List[str],
    dataset_name: str = "",
) -> Dict:
    """Encode one example from a pre-tokenized HF dataset.

    For CORD: maps BIO NER tags → 6-class layout labels via _cord_ner_to_layout.
    For others: maps tag ids → label_list names → label2id.
    """
    default_id = _default_label_id(label2id)

    image    = example["image"]
    words    = example.get("words") or example.get("tokens") or []
    boxes    = example.get("bboxes") or example.get("bbox") or []
    raw_tags = example.get("ner_tags") or example.get("ner_tag") or example.get("tags") or []

    word_labels: List[int] = []
    for tag in raw_tags:
        if dataset_name == "CORD":
            # tag is an int index into label_list of BIO NER names
            tag_name = label_list[tag] if isinstance(tag, int) and tag < len(label_list) else str(tag)
            layout_label = _cord_ner_to_layout(tag_name)
            word_labels.append(label2id.get(layout_label, default_id))
        else:
            if isinstance(tag, int):
                name = label_list[tag] if tag < len(label_list) else "O"
            else:
                name = str(tag)
            word_labels.append(label2id.get(name, default_id))

    if not words or not boxes:
        fallback = torch.full((max_length,), -100, dtype=torch.long)
        fallback[0] = default_id
        return {
            "input_ids":      torch.zeros(max_length, dtype=torch.long),
            "attention_mask": torch.zeros(max_length, dtype=torch.long),
            "bbox":           torch.zeros((max_length, 4), dtype=torch.long),
            "pixel_values":   torch.zeros((3, 224, 224)),
            "labels":         fallback,
        }

    try:
        enc = processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    except Exception as exc:
        logger.debug("HF-native processor failed: %s", exc)
        fallback = torch.full((max_length,), -100, dtype=torch.long)
        fallback[0] = default_id
        return {
            "input_ids":      torch.zeros(max_length, dtype=torch.long),
            "attention_mask": torch.zeros(max_length, dtype=torch.long),
            "bbox":           torch.zeros((max_length, 4), dtype=torch.long),
            "pixel_values":   torch.zeros((3, 224, 224)),
            "labels":         fallback,
        }

    return {
        "input_ids":      enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "bbox":           enc["bbox"].squeeze(0),
        "pixel_values":   enc["pixel_values"].squeeze(0),
        "labels":         enc["labels"].squeeze(0),
    }


def _preprocess_hf_native(
    dataset,
    processor,
    max_length: int,
    label2id: Dict[str, int],
    label_list: List[str],
    dataset_name: str = "",
):
    encoded = dataset.map(
        lambda ex: _encode_hf_native_example(ex, processor, max_length, label2id, label_list, dataset_name),
        batched=False,
        remove_columns=dataset.column_names,
        desc=f"Encode LayoutLMv3 HF-native inputs ({dataset_name})",
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


def _class_weights(encoded_train, num_labels: int, id2label: Dict[int, str]) -> torch.Tensor:
    counts: Counter = Counter()
    for row in encoded_train["labels"]:
        for lbl in (row.tolist() if hasattr(row, "tolist") else row):
            if lbl != -100:
                counts[int(lbl)] += 1
    total = sum(counts.values()) or 1
    weights = []
    for i in range(num_labels):
        c = counts.get(i, 0)
        weights.append(min(total / (num_labels * c), 4.0) if c > 0 else 1.0)
    logger.info("[class_weights] %s", {id2label.get(i, i): round(w, 3) for i, w in enumerate(weights)})
    return torch.tensor(weights, dtype=torch.float)


class _WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self._class_weights.to(logits.device)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            weight=weight,
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


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
    hf_native_datasets: Optional[List[str]] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
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
        "microsoft/layoutlmv3-base", apply_ocr=True, use_fast=True
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

    # ------------------------------------------------------------------
    # Encoding: HF-native path or segment-containment path
    # ------------------------------------------------------------------
    if hf_native_datasets:
        from datasets import concatenate_datasets
        train_parts, val_parts = [], []
        # Accumulate label names across all native datasets so label2id is consistent
        all_label_names: List[str] = list(label2id.keys())

        for ds_name in hf_native_datasets:
            logger.info("[LoRA-%s] Loading HF-native dataset: %s", group_name, ds_name)
            tr, va = _load_hf_native(ds_name, max_train_samples, max_val_samples)
            # Extend label2id with any new labels from this dataset
            native_labels = _label_list_from_hf_native(tr)
            for lbl in native_labels:
                if lbl not in label2id:
                    new_id = len(label2id)
                    label2id[lbl] = new_id
                    id2label[new_id] = lbl
                    all_label_names.append(lbl)
            logger.info("[LoRA-%s] Preprocessing HF-native train (%s) ...", group_name, ds_name)
            train_parts.append(_preprocess_hf_native(tr, processor, max_length, label2id, native_labels, ds_name))
            logger.info("[LoRA-%s] Preprocessing HF-native val (%s) ...", group_name, ds_name)
            val_parts.append(_preprocess_hf_native(va, processor, max_length, label2id, native_labels, ds_name))

        encoded_train = concatenate_datasets(train_parts)
        encoded_val   = concatenate_datasets(val_parts)
    else:
        logger.info("[LoRA-%s] Preprocessing train split ...", group_name)
        encoded_train = _preprocess_dataset(train_dataset, processor, max_length, label2id)
        logger.info("[LoRA-%s] Preprocessing val split ...", group_name)
        encoded_val   = _preprocess_dataset(val_dataset, processor, max_length, label2id)

    _log_supervision_stats(encoded_train, "train", group_name, id2label)
    _log_supervision_stats(encoded_val,   "validation", group_name, id2label)

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
    grad_accum = 2
    steps_per_epoch = max(1, int(np.ceil(len(encoded_train) / max(batch_size * grad_accum, 1))))
    warmup_steps = max(1, steps_per_epoch)  # 1 epoch warmup

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine_with_restarts",
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

    cw = _class_weights(encoded_train, len(label2id), id2label)
    trainer = _WeightedTrainer(
        model=model,
        args=args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), csv_logger],
        class_weights=cw,
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