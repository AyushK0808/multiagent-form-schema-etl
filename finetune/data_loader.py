"""
finetune/data_loader.py
=======================
Loads, normalises, augments, and concatenates datasets into the unified
intermediate format consumed by both trainers.

Public API
----------
build_combined_dataset(dataset_names, ...) → (train, val, label2id, id2label, manifest)
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from PIL import Image

from augmentation import augment_image
from config import DATASET_SPECS, DatasetSpec
from normalizers import NORMALIZERS, normalize_generic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-dataset loader
# ---------------------------------------------------------------------------

def _load_single_dataset(
    spec: DatasetSpec,
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    augment_train: bool = True,
) -> Tuple[Optional[object], Optional[object]]:
    from datasets import load_dataset

    logger.info("[DataLoader] Loading %s from %s", spec.name, spec.repo_id)
    try:
        ds = load_dataset(spec.repo_id)
    except Exception as exc:
        logger.warning("[DataLoader] Could not load %s: %s — skipping", spec.name, exc)
        return None, None

    # Standardise splits: ensure "train" and "validation" always exist
    if "train" not in ds:
        only = next(iter(ds.keys()))
        sp = ds[only].train_test_split(test_size=0.1, seed=42)
        ds = {"train": sp["train"], "validation": sp["test"]}
    else:
        val_key = next(
            (n for n in ("validation", "val", "dev", "test") if n in ds), None
        )
        if val_key is None:
            sp = ds["train"].train_test_split(test_size=0.1, seed=42)
            ds = {"train": sp["train"], "validation": sp["test"]}
        else:
            ds = {"train": ds["train"], "validation": ds[val_key]}

    norm_fn = NORMALIZERS.get(spec.name)

    def normalize(example: Dict, is_train: bool = False) -> Dict:
        try:
            result = norm_fn(example) if norm_fn else normalize_generic(example, spec)
        except Exception as exc:
            logger.debug("[DataLoader] Normalize error (%s): %s", spec.name, exc)
            result = {
                "image":        Image.new("RGB", (224, 224)),
                "segments":     [],
                "label_text":   spec.schema_name or spec.name.lower(),
                "dataset_name": spec.name,
            }
        if is_train and augment_train:
            result["image"] = augment_image(result["image"])
        return result

    train_ds = ds["train"]
    val_ds   = ds["validation"]

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_val_samples:
        val_ds = val_ds.select(range(min(max_val_samples, len(val_ds))))

    train_ds = train_ds.map(
        lambda ex: normalize(ex, is_train=True),
        remove_columns=train_ds.column_names,
        desc=f"Normalize {spec.name} train",
    )
    val_ds = val_ds.map(
        lambda ex: normalize(ex, is_train=False),
        remove_columns=val_ds.column_names,
        desc=f"Normalize {spec.name} val",
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def build_combined_dataset(
    dataset_names: List[str],
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    augment_train: bool = True,
    curriculum: bool = False,
):
    """
    Load and concatenate multiple datasets.

    If curriculum=True, datasets are ordered by DatasetSpec.curriculum_order
    (simpler/cleaner data first) before concatenation, which helps Donut
    build up schema recognition progressively.

    Returns
    -------
    train_dataset, val_dataset, label2id, id2label, manifest
    """
    from datasets import concatenate_datasets

    specs = [DATASET_SPECS[n] for n in dataset_names]
    if curriculum:
        specs = sorted(specs, key=lambda s: s.curriculum_order)

    train_parts, val_parts, manifest = [], [], []
    for spec in specs:
        train_ds, val_ds = _load_single_dataset(
            spec, max_train_samples, max_val_samples, augment_train=augment_train
        )
        if train_ds is None:
            continue
        train_parts.append(train_ds)
        val_parts.append(val_ds)
        manifest.append({
            **asdict(spec),
            "train_examples":      len(train_ds),
            "validation_examples": len(val_ds),
        })

    if not train_parts:
        raise RuntimeError(
            "No datasets could be loaded — check dataset IDs and network access."
        )

    train_dataset = concatenate_datasets(train_parts)
    val_dataset   = concatenate_datasets(val_parts)

    label_texts = sorted(
        set(train_dataset["label_text"]) | set(val_dataset["label_text"])
    )
    label2id = {l: i for i, l in enumerate(label_texts)}
    id2label = {i: l for l, i in label2id.items()}

    return train_dataset, val_dataset, label2id, id2label, manifest