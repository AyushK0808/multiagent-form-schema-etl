"""
finetune/train_lora.py
======================
CLI entry point for LoRA adapter training.

Trains one LoRA adapter per group (or a specific subset) using the
three-group curriculum defined in adapter_groups.py.

Usage examples
--------------
# Train all three adapter groups
python train_lora.py

# Train only group_3 (reasoning/NDA) — useful after adding new datasets
python train_lora.py --groups group_3

# Quick smoke-test
python train_lora.py --max-train-samples 100 --no-augment

# LayoutLMv3 adapters only (skip Donut)
python train_lora.py --model layoutlmv3

# Restrict to specific datasets across all selected groups
python train_lora.py --groups group_1 group_2 --datasets CORD FUNSD SROIE
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from adapter_groups import ADAPTER_GROUPS, AdapterGroupSpec
from augmentation import augmentation_available
from config import DATASET_SPECS
from data_loader import build_combined_dataset
from label_spaces import build_group_label_space, log_dataset_label_distribution
from lora_layoutlmv3_trainer import train_lora_layoutlmv3

# Donut adapter training reuses the same Seq2Seq trainer but with LoRA
from lora_donut_trainer import train_lora_donut

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA adapter training (3-group curriculum)")
    parser.add_argument(
        "--model",
        choices=("layoutlmv3", "donut", "both"),
        default="both",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["group_1", "group_2", "group_3"],
        default=["group_1", "group_2", "group_3"],
        help="Which adapter groups to train",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET_NAME",
        help=(
            "Restrict training to specific datasets. "
            "Only datasets that also belong to the selected group(s) are used. "
            "If omitted, all datasets in each group are used. "
            "Example: --datasets CORD FUNSD KLEISTER_NDA"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("models") / "adapters",
        help="Root dir; each group saved under <root>/<group_name>/",
    )
    parser.add_argument(
        "--normalized-cache-root", type=Path,
        default=Path("data") / "intermediate" / "finetune_normalized",
    )
    parser.add_argument("--refresh-normalized-cache", action="store_true")
    parser.add_argument(
        "--augmented-cache-root", type=Path,
        default=Path("data") / "intermediate" / "finetune_augmented",
    )
    parser.add_argument("--refresh-augmented-cache", action="store_true")
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch-size",    type=int,   default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Higher than full fine-tune is fine; only adapters update")
    parser.add_argument("--max-length",    type=int,   default=512)
    parser.add_argument("--max-train-samples", type=int,
                        help="Cap per-dataset training samples (debug)")
    parser.add_argument("--max-val-samples",   type=int)
    parser.add_argument("--no-augment",    action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-group training helper
# ---------------------------------------------------------------------------

def _train_group(
    grp: AdapterGroupSpec,
    args: argparse.Namespace,
    augment: bool,
) -> None:
    logger.info("=" * 60)
    logger.info("  Adapter group: %s  (%s)", grp.name, grp.label)
    logger.info("  Group datasets: %s", grp.datasets)
    logger.info("=" * 60)

    # Start with datasets that belong to this group and exist in DATASET_SPECS
    available = [d for d in grp.datasets if d in DATASET_SPECS]

    # Apply --datasets filter if provided
    if args.datasets:
        filtered = [d for d in available if d in args.datasets]
        if not filtered:
            logger.warning(
                "[%s] --datasets filter excluded all datasets for this group "
                "(group datasets: %s, filter: %s) — skipping",
                grp.name, available, args.datasets,
            )
            return
        if filtered != available:
            logger.info(
                "[%s] --datasets filter: using %s (excluded: %s)",
                grp.name, filtered,
                [d for d in available if d not in filtered],
            )
        available = filtered

    if not available:
        logger.warning("[%s] No available datasets — skipping", grp.name)
        return

    logger.info("[%s] Training on: %s", grp.name, available)

    # Build combined dataset for this group (curriculum order within the group)
    train_ds, val_ds, _, _, manifest = build_combined_dataset(
        dataset_names=available,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        normalized_cache_root=args.normalized_cache_root / grp.name,
        refresh_normalized_cache=args.refresh_normalized_cache,
        augmented_cache_root=args.augmented_cache_root / grp.name,
        refresh_augmented_cache=args.refresh_augmented_cache,
        augment_train=augment,
        curriculum=True,
    )
    label2id, id2label = build_group_label_space(available, train_ds, val_ds)
    log_dataset_label_distribution(train_ds, "train", grp.name)
    log_dataset_label_distribution(val_ds, "validation", grp.name)
    logger.info("[%s] Token label space: %s", grp.name, label2id)

    group_output = args.output_root / grp.name

    # Write dataset manifest for reproducibility
    group_output.mkdir(parents=True, exist_ok=True)
    (group_output / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "group":       grp.name,
                "label":       grp.label,
                "datasets":    manifest,
                "datasets_filter": args.datasets,
                "train_n":     len(train_ds),
                "val_n":       len(val_ds),
                "augmented":   augment,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # LayoutLMv3 adapter
    if args.model in ("layoutlmv3", "both"):
        lm_out = group_output / "layoutlmv3"
        logger.info("[%s] Training LayoutLMv3 LoRA adapter → %s", grp.name, lm_out)
        train_lora_layoutlmv3(
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=lm_out,
            group_name=grp.name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            label2id=label2id,
            id2label=id2label,
        )

    # Donut adapter
    if args.model in ("donut", "both"):
        donut_out = group_output / "donut"
        logger.info("[%s] Training Donut LoRA adapter → %s", grp.name, donut_out)
        train_lora_donut(
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=donut_out,
            group_name=grp.name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.normalized_cache_root.mkdir(parents=True, exist_ok=True)
    args.augmented_cache_root.mkdir(parents=True, exist_ok=True)

    augment = not args.no_augment
    if augment and not augmentation_available:
        logger.warning("Augmentation requested but albumentations not installed — skipping")
        augment = False

    logger.info(
        "LoRA training — groups=%s  model=%s  datasets_filter=%s  augment=%s  bf16=%s",
        args.groups, args.model, args.datasets, augment, torch.cuda.is_available(),
    )

    target_groups = [g for g in ADAPTER_GROUPS if g.name in args.groups]
    for grp in target_groups:
        _train_group(grp, args, augment)

    logger.info("All requested adapter groups trained.")
    logger.info("Adapter root: %s", args.output_root.resolve())


if __name__ == "__main__":
    main()