"""
Fine-tune LayoutLMv3 and Donut for schema recognition across document datasets.

Supported datasets:
* FUNSD -> forms
* CORD -> receipts
* SROIE -> invoices
* DocVQA -> reasoning-heavy document QA pages
* RVL-CDIP -> diverse document classes
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

_DONUT_TASK_PROMPT = "<s_schema_recognition>"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    schema_name: Optional[str]
    description: str


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "FUNSD": DatasetSpec(
        name="FUNSD",
        repo_id="nielsr/funsd",
        schema_name="funsd_form",
        description="Forms",
    ),
    "CORD": DatasetSpec(
        name="CORD",
        repo_id="naver-clova-ix/cord-v2",
        schema_name="cord_receipt",
        description="Receipts",
    ),
    "SROIE": DatasetSpec(
        name="SROIE",
        repo_id="rth/sroie-2019-v2",
        schema_name="sroie_invoice",
        description="Invoices",
    ),
    "DOCVQA": DatasetSpec(
        name="DOCVQA",
        repo_id="lmms-lab/DocVQA",
        schema_name="docvqa_reasoning",
        description="Document reasoning",
    ),
    "RVL-CDIP": DatasetSpec(
        name="RVL-CDIP",
        repo_id="aharley/rvl_cdip",
        schema_name=None,
        description="Document diversity",
    ),
}


def _open_image_from_value(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            import io

            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    raise ValueError("Unsupported image payload")


def _extract_image(example: Dict) -> Image.Image:
    for key in ("image", "img", "document", "page_image"):
        if key in example and example[key] is not None:
            return _open_image_from_value(example[key])
    raise KeyError(f"Could not find image key in example with keys={list(example.keys())}")


def _extract_rvl_label(example: Dict, label_names: Optional[List[str]]) -> str:
    for key in ("label", "labels", "class", "category"):
        if key not in example:
            continue
        value = example[key]
        if isinstance(value, str):
            raw = value
        elif label_names is not None and isinstance(value, (int, np.integer)):
            raw = label_names[int(value)]
        else:
            raw = str(value)
        safe = raw.lower().replace(" ", "_").replace("-", "_")
        return f"rvl_cdip_{safe}"
    raise KeyError("RVL-CDIP example is missing a label field")


def _load_single_dataset(
    spec: DatasetSpec,
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
):
    from datasets import load_dataset

    logger.info("Loading dataset %s from %s", spec.name, spec.repo_id)
    ds = load_dataset(spec.repo_id)

    if "train" not in ds and len(ds) == 1:
        only_split = next(iter(ds.keys()))
        split = ds[only_split].train_test_split(test_size=0.1, seed=42)
        ds = {"train": split["train"], "validation": split["test"]}
    else:
        train_split_name = "train" if "train" in ds else next(iter(ds.keys()))
        val_split_name = next((name for name in ("validation", "val", "dev", "test") if name in ds), None)
        if val_split_name is None:
            split = ds[train_split_name].train_test_split(test_size=0.1, seed=42)
            ds = {"train": split["train"], "validation": split["test"]}
        else:
            ds = {"train": ds[train_split_name], "validation": ds[val_split_name]}

    label_names = None
    train_features = getattr(ds["train"], "features", None)
    if train_features and "label" in train_features and hasattr(train_features["label"], "names"):
        label_names = list(train_features["label"].names)

    def normalize(example: Dict) -> Dict:
        image = _extract_image(example)
        label_text = spec.schema_name or _extract_rvl_label(example, label_names)
        return {"image": image, "label_text": label_text, "dataset_name": spec.name}

    train_ds = ds["train"]
    val_ds = ds["validation"]

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_val_samples:
        val_ds = val_ds.select(range(min(max_val_samples, len(val_ds))))

    train_ds = train_ds.map(normalize, remove_columns=train_ds.column_names, desc=f"Normalize {spec.name} train")
    val_ds = val_ds.map(normalize, remove_columns=val_ds.column_names, desc=f"Normalize {spec.name} val")
    return train_ds, val_ds


def build_combined_dataset(
    dataset_names: List[str],
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
):
    from datasets import concatenate_datasets

    train_parts = []
    val_parts = []
    manifest = []

    for dataset_name in dataset_names:
        spec = DATASET_SPECS[dataset_name]
        train_ds, val_ds = _load_single_dataset(spec, max_train_samples, max_val_samples)
        train_parts.append(train_ds)
        val_parts.append(val_ds)
        manifest.append(
            {
                **asdict(spec),
                "train_examples": len(train_ds),
                "validation_examples": len(val_ds),
            }
        )

    train_dataset = concatenate_datasets(train_parts)
    val_dataset = concatenate_datasets(val_parts)
    label_texts = sorted(set(train_dataset["label_text"]) | set(val_dataset["label_text"]))
    label2id = {label: idx for idx, label in enumerate(label_texts)}
    id2label = {idx: label for label, idx in label2id.items()}
    return train_dataset, val_dataset, label2id, id2label, manifest


def _layoutlm_preprocess(dataset, processor, label2id: Dict[str, int], max_length: int):
    def encode_batch(batch: Dict) -> Dict:
        enc = processor(
            images=batch["image"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc["labels"] = [label2id[label] for label in batch["label_text"]]
        return enc

    encoded = dataset.map(
        encode_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Encode LayoutLMv3 inputs",
    )
    encoded.set_format("torch")
    return encoded


def train_layoutlmv3(
    train_dataset,
    val_dataset,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
):
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        LayoutLMv3ForSequenceClassification,
        LayoutLMv3Processor,
        Trainer,
        TrainingArguments,
    )

    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
    encoded_train = _layoutlm_preprocess(train_dataset, processor, label2id, max_length)
    encoded_val = _layoutlm_preprocess(val_dataset, processor, label2id, max_length)

    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
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
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    metrics = trainer.evaluate()
    (output_dir / "label_map.json").write_text(json.dumps(id2label, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _donut_preprocess(dataset, processor, max_length: int):
    def encode_example(example: Dict) -> Dict:
        target = json.dumps({"schema": example["label_text"]}, ensure_ascii=True)
        pixel_values = processor(example["image"], return_tensors="pt").pixel_values.squeeze(0)
        decoder_text = f"{_DONUT_TASK_PROMPT}{target}{processor.tokenizer.eos_token}"
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
        encode_example,
        remove_columns=dataset.column_names,
        desc="Encode Donut inputs",
    )
    encoded.set_format("torch")
    return encoded


def train_donut(
    train_dataset,
    val_dataset,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
):
    from transformers import (
        DonutProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        VisionEncoderDecoderModel,
    )

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    added_tokens = processor.tokenizer.add_special_tokens({"additional_special_tokens": [_DONUT_TASK_PROMPT]})
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    if added_tokens:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))

    prompt_id = processor.tokenizer.convert_tokens_to_ids(_DONUT_TASK_PROMPT)
    model.config.decoder_start_token_id = prompt_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = max_length

    encoded_train = _donut_preprocess(train_dataset, processor, max_length)
    encoded_val = _donut_preprocess(val_dataset, processor, max_length)

    def collate(features: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([feature["pixel_values"] for feature in features]),
            "labels": torch.stack([feature["labels"] for feature in features]),
        }

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
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
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    metrics = trainer.evaluate()
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "task_prompt.txt").write_text(_DONUT_TASK_PROMPT, encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schema recognition fine-tuning")
    parser.add_argument(
        "--model",
        choices=("layoutlmv3", "donut", "both"),
        default="both",
        help="Which recognizer model to fine-tune",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["FUNSD", "CORD", "SROIE", "DOCVQA", "RVL-CDIP"],
        choices=list(DATASET_SPECS.keys()),
        help="Datasets to merge for training",
    )
    parser.add_argument("--output-root", type=Path, default=Path("models") / "schema_recognition")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-samples-per-dataset", type=int)
    parser.add_argument("--max-val-samples-per-dataset", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, label2id, id2label, manifest = build_combined_dataset(
        dataset_names=args.datasets,
        max_train_samples=args.max_train_samples_per_dataset,
        max_val_samples=args.max_val_samples_per_dataset,
    )

    manifest_payload = {
        "datasets": manifest,
        "labels": id2label,
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
    }
    (args.output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )

    if args.model in ("layoutlmv3", "both"):
        layout_dir = args.output_root / "layoutlmv3"
        layout_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Training LayoutLMv3 schema recognizer")
        train_layoutlmv3(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            label2id=label2id,
            id2label=id2label,
            output_dir=layout_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
        )

    if args.model in ("donut", "both"):
        donut_dir = args.output_root / "donut"
        donut_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Training Donut schema recognizer")
        train_donut(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=donut_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
        )


if __name__ == "__main__":
    main()
