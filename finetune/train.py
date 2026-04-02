"""
finetune/train.py
=================
CLI entry point — orchestrates dataset loading and model training.

Usage examples
--------------
# Train both models on the default dataset set
python train.py

# LayoutLMv3 only, all datasets, with curriculum ordering
python train.py --model layoutlmv3 --all-datasets --curriculum

# Quick smoke-test (100 samples per dataset, no augmentation)
python train.py --max-train-samples-per-dataset 100 --no-augment

# Donut only on receipts + invoices
python train.py --model donut --datasets CORD SROIE
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from augmentation import augmentation_available
from config import DATASET_SPECS
from data_loader import build_combined_dataset
from donut_trainer import train_donut
from layoutlmv3_trainer import train_layoutlmv3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schema recognition fine-tuning")
    parser.add_argument(
        "--model",
        choices=("layoutlmv3", "donut", "both"),
        default="both",
        help="Which model to fine-tune",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "FUNSD", "CORD", "SROIE", "DOCVQA", "RVL-CDIP",
            "PUBLAYNET", "DOCLAYNET", "KLEISTER_NDA",
        ],
        choices=list(DATASET_SPECS.keys()),
        help="Datasets to train on",
    )
    parser.add_argument(
        "--all-datasets", action="store_true",
        help="Use every dataset in DATASET_SPECS (overrides --datasets)",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("models") / "schema_recognition",
    )
    parser.add_argument(
        "--normalized-cache-root",
        type=Path,
        default=Path("data") / "intermediate" / "finetune_normalized",
        help="Directory for persisted normalized dataset splits reused across runs",
    )
    parser.add_argument(
        "--refresh-normalized-cache",
        action="store_true",
        help="Rebuild persisted normalized dataset splits instead of reusing them",
    )
    parser.add_argument(
        "--augmented-cache-root",
        type=Path,
        default=Path("data") / "intermediate" / "finetune_augmented",
        help="Directory for persisted augmented training splits reused across runs",
    )
    parser.add_argument(
        "--refresh-augmented-cache",
        action="store_true",
        help="Rebuild persisted augmented training splits instead of reusing them",
    )
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch-size",    type=int,   default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length",    type=int,   default=512)
    parser.add_argument(
        "--llrd-factor", type=float, default=0.9,
        help="Per-layer LR decay factor for LayoutLMv3 (0 < factor ≤ 1)",
    )
    parser.add_argument(
        "--max-train-samples-per-dataset", type=int,
        help="Cap each dataset's training split (useful for debugging)",
    )
    parser.add_argument("--max-val-samples-per-dataset", type=int)
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable Albumentations image augmentation",
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Order datasets by curriculum_order before concatenation",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.normalized_cache_root.mkdir(parents=True, exist_ok=True)
    args.augmented_cache_root.mkdir(parents=True, exist_ok=True)

    dataset_names = list(DATASET_SPECS.keys()) if args.all_datasets else args.datasets
    augment       = not args.no_augment

    if augment and not augmentation_available:
        logger.warning(
            "Augmentation requested but albumentations is not installed — skipping. "
            "Install with: pip install albumentations"
        )
        augment = False

    logger.info(
        "Config — datasets=%s  augment=%s  curriculum=%s  bf16=%s",
        dataset_names, augment, args.curriculum, torch.cuda.is_available(),
    )

    train_dataset, val_dataset, label2id, id2label, manifest = build_combined_dataset(
        dataset_names=dataset_names,
        max_train_samples=args.max_train_samples_per_dataset,
        max_val_samples=args.max_val_samples_per_dataset,
        normalized_cache_root=args.normalized_cache_root,
        refresh_normalized_cache=args.refresh_normalized_cache,
        augmented_cache_root=args.augmented_cache_root,
        refresh_augmented_cache=args.refresh_augmented_cache,
        augment_train=augment,
        curriculum=args.curriculum,
    )

    # Write dataset manifest so training runs are reproducible
    (args.output_root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "datasets":             manifest,
                "labels":               id2label,
                "train_examples":       len(train_dataset),
                "validation_examples":  len(val_dataset),
                "augmentation_enabled": augment,
                "curriculum_enabled":   args.curriculum,
                "normalized_cache_root": str(args.normalized_cache_root.resolve()),
                "normalized_cache_refreshed": args.refresh_normalized_cache,
                "augmented_cache_root": str(args.augmented_cache_root.resolve()),
                "augmented_cache_refreshed": args.refresh_augmented_cache,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.model in ("layoutlmv3", "both"):
        layout_dir = args.output_root / "layoutlmv3"
        layout_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "=== LayoutLMv3ForTokenClassification  (6-label, LLRD=%.2f, bf16=%s) ===",
            args.llrd_factor, torch.cuda.is_available(),
        )
        train_layoutlmv3(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=layout_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            llrd_factor=args.llrd_factor,
        )

    if args.model in ("donut", "both"):
        donut_dir = args.output_root / "donut"
        donut_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "=== Donut seq2seq  (CER metric, curriculum=%s, bf16=%s) ===",
            args.curriculum, torch.cuda.is_available(),
        )
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
