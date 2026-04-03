"""
finetune/donut_trainer.py
=========================
Fine-tunes Donut (VisionEncoderDecoderModel) for schema recognition.

Given a page image it outputs JSON like {"schema": "cord_receipt"} which
tells the downstream pipeline which field extraction schema to apply.

Key design decisions
--------------------
- predict_with_generate=True so compute_metrics receives decoded token sequences
- Metric: CER (character error rate) + exact schema match
- Curriculum ordering is handled upstream in data_loader.build_combined_dataset
- EarlyStoppingCallback(patience=3) on CER (lower = better)
- Per-epoch CSV log  → <output_dir>/training_log.csv
- Matplotlib plots   → <output_dir>/plots/
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from config import DONUT_TASK_PROMPT
from metrics import compute_cer
from metrics_logger import EpochCSVLogger, generate_training_plots

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_dataset(dataset, processor, max_length: int):
    def encode(example: Dict) -> Dict:
        target       = json.dumps({"schema": example["label_text"]}, ensure_ascii=True)
        pixel_values = processor(example["image"], return_tensors="pt").pixel_values.squeeze(0)
        decoder_text = f"{DONUT_TASK_PROMPT}{target}{processor.tokenizer.eos_token}"
        labels = processor.tokenizer(
            decoder_text,
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

    encoded = dataset.map(
        encode,
        batched=False,
        remove_columns=dataset.column_names,
        desc="Encode Donut inputs",
    )
    encoded.set_format("torch")
    return encoded


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_donut(
    train_dataset,
    val_dataset,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> Dict:
    from transformers import (
        DonutProcessor,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        VisionEncoderDecoderModel,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    added     = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [DONUT_TASK_PROMPT]}
    )
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    if added:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))

    prompt_id = processor.tokenizer.convert_tokens_to_ids(DONUT_TASK_PROMPT)
    model.config.decoder_start_token_id = prompt_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.eos_token_id
    model.config.max_length             = max_length

    logger.info("[Donut] Preprocessing train split …")
    encoded_train = _preprocess_dataset(train_dataset, processor, max_length)
    logger.info("[Donut] Preprocessing val split …")
    encoded_val   = _preprocess_dataset(val_dataset, processor, max_length)

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred
        label_ids   = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        preds_str   = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        refs_str    = processor.batch_decode(label_ids, skip_special_tokens=True)
        cer_score   = compute_cer(preds_str, refs_str)

        exact_matches = 0
        for pred_str, ref_str in zip(preds_str, refs_str):
            try:
                if json.loads(pred_str).get("schema") == json.loads(ref_str).get("schema"):
                    exact_matches += 1
            except json.JSONDecodeError:
                pass

        return {
            "cer":         round(cer_score, 4),
            "exact_match": round(exact_matches / max(len(preds_str), 1), 4),
        }

    def collate(features: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels":       torch.stack([f["labels"]       for f in features]),
        }

    # ── CSV + plot callback ───────────────────────────────────────────────
    csv_logger = EpochCSVLogger(output_dir / "training_log.csv")

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        num_train_epochs=epochs,
        bf16=torch.cuda.is_available(),
        weight_decay=0.01,
        max_grad_norm=1.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=max_length,
        remove_unused_columns=False,
        logging_steps=25,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        data_collator=collate,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), csv_logger],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    metrics = trainer.evaluate()

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "task_prompt.txt").write_text(DONUT_TASK_PROMPT, encoding="utf-8")

    # ── Generate plots ────────────────────────────────────────────────────
    try:
        generate_training_plots(
            output_dir,
            primary_metric="eval_cer",
            higher_is_better=False,   # lower CER = better
            model_label="Donut",
        )
    except Exception as exc:
        logger.warning("[Donut] Plot generation failed: %s", exc)

    logger.info(
        "[Donut] Done — CER=%.4f  exact_match=%.4f",
        metrics.get("eval_cer", 1.0), metrics.get("eval_exact_match", 0.0),
    )
    return metrics