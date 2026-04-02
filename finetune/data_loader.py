"""
finetune/data_loader.py
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from augmentation import augment_image
from config import DATASET_SPECS, DatasetSpec
from normalizers import NORMALIZERS, normalize_generic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HuggingFace authentication — load .env from project root if present
# ---------------------------------------------------------------------------

def _hf_login() -> None:
    """Authenticate with the HF Hub using HF_TOKEN from the environment or .env."""
    # Walk up from finetune/ to find .env at the project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            logger.info("[HF] Loaded .env from %s", env_path)
        except ImportError:
            logger.warning("[HF] python-dotenv not installed; set HF_TOKEN manually")

    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning(
            "[HF] HF_TOKEN not set — gated datasets will return 401. "
            "Add HF_TOKEN=<your_token> to .env or the environment."
        )
        return

    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        logger.info("[HF] Authenticated with HuggingFace Hub")
    except Exception as exc:
        logger.warning("[HF] Hub login failed: %s", exc)


_hf_login()


# ---------------------------------------------------------------------------
# Single-dataset loader
# ---------------------------------------------------------------------------

def _load_single_dataset(
    spec: DatasetSpec,
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    normalized_cache_root: Optional[Path] = None,
    refresh_normalized_cache: bool = False,
    augmented_cache_root: Optional[Path] = None,
    refresh_augmented_cache: bool = False,
    augment_train: bool = True,
) -> Tuple[Optional[object], Optional[object]]:
    from datasets import load_dataset, load_from_disk

    logger.info("[DataLoader] Loading %s from %s", spec.name, spec.repo_id)

    cache_base = None
    train_cache_dir = None
    val_cache_dir = None
    augmented_cache_base = None
    augmented_train_cache_dir = None
    if normalized_cache_root is not None:
        cache_base = normalized_cache_root / spec.name.lower()
        train_cache_dir = cache_base / "train"
        val_cache_dir = cache_base / "validation"
    if augmented_cache_root is not None:
        augmented_cache_base = augmented_cache_root / spec.name.lower()
        augmented_train_cache_dir = augmented_cache_base / "train"
    if (
        augment_train
        and augmented_train_cache_dir is not None
        and not refresh_augmented_cache
        and augmented_train_cache_dir.exists()
        and val_cache_dir is not None
        and val_cache_dir.exists()
    ):
        logger.info("[DataLoader] Reusing augmented cache for %s", spec.name)
        train_ds = load_from_disk(str(augmented_train_cache_dir))
        val_ds = load_from_disk(str(val_cache_dir))
        if max_train_samples:
            train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
        if max_val_samples:
            val_ds = val_ds.select(range(min(max_val_samples, len(val_ds))))
        return train_ds, val_ds
    if (
        not refresh_normalized_cache
        and train_cache_dir is not None
        and val_cache_dir is not None
        and train_cache_dir.exists()
        and val_cache_dir.exists()
    ):
        logger.info("[DataLoader] Reusing normalized cache for %s", spec.name)
        train_ds = load_from_disk(str(train_cache_dir))
        val_ds = load_from_disk(str(val_cache_dir))
        if augment_train:
            train_ds = _maybe_augment_and_cache(
                train_ds=train_ds,
                spec=spec,
                augmented_cache_base=augmented_cache_base,
                augmented_train_cache_dir=augmented_train_cache_dir,
                refresh_augmented_cache=refresh_augmented_cache,
            )
        if max_train_samples:
            train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
        if max_val_samples:
            val_ds = val_ds.select(range(min(max_val_samples, len(val_ds))))
        return train_ds, val_ds

    load_kwargs: Dict = {}
    if spec.config_name:
        load_kwargs["name"] = spec.config_name
    if spec.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    try:
        ds = load_dataset(spec.repo_id, **load_kwargs)
    except Exception as exc:
        logger.warning("[DataLoader] Could not load %s: %s — skipping", spec.name, exc)
        return None, None

    # Standardise splits
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

    def normalize(example: Dict) -> Dict:
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
        return result

    train_ds = ds["train"]
    val_ds   = ds["validation"]

    train_ds = train_ds.map(
        normalize,
        remove_columns=train_ds.column_names,
        desc=f"Normalize {spec.name} train",
    )
    val_ds = val_ds.map(
        normalize,
        remove_columns=val_ds.column_names,
        desc=f"Normalize {spec.name} val",
    )

    if cache_base is not None:
        cache_base.mkdir(parents=True, exist_ok=True)
        logger.info("[DataLoader] Saving normalized cache for %s to %s", spec.name, cache_base)
        train_ds.save_to_disk(str(train_cache_dir))
        val_ds.save_to_disk(str(val_cache_dir))
        (cache_base / "manifest.json").write_text(
            json.dumps(
                {
                    **asdict(spec),
                    "train_examples": len(train_ds),
                    "validation_examples": len(val_ds),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if augment_train:
        train_ds = _maybe_augment_and_cache(
            train_ds=train_ds,
            spec=spec,
            augmented_cache_base=augmented_cache_base,
            augmented_train_cache_dir=augmented_train_cache_dir,
            refresh_augmented_cache=refresh_augmented_cache,
        )
    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_val_samples:
        val_ds = val_ds.select(range(min(max_val_samples, len(val_ds))))
    return train_ds, val_ds


def _augment_normalized_example(example: Dict) -> Dict:
    example["image"] = augment_image(example["image"])
    return example


def _maybe_augment_and_cache(
    train_ds,
    spec: DatasetSpec,
    augmented_cache_base: Optional[Path],
    augmented_train_cache_dir: Optional[Path],
    refresh_augmented_cache: bool,
):
    from datasets import load_from_disk

    if (
        augmented_train_cache_dir is not None
        and not refresh_augmented_cache
        and augmented_train_cache_dir.exists()
    ):
        logger.info("[DataLoader] Reusing augmented cache for %s", spec.name)
        return load_from_disk(str(augmented_train_cache_dir))

    augmented_train_ds = train_ds.map(
        _augment_normalized_example,
        batched=False,
        desc=f"Augment {spec.name} train",
    )
    if augmented_cache_base is not None and augmented_train_cache_dir is not None:
        augmented_cache_base.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[DataLoader] Saving augmented cache for %s to %s",
            spec.name,
            augmented_cache_base,
        )
        augmented_train_ds.save_to_disk(str(augmented_train_cache_dir))
        (augmented_cache_base / "manifest.json").write_text(
            json.dumps(
                {
                    **asdict(spec),
                    "train_examples": len(augmented_train_ds),
                    "source": "normalized_train_split",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return augmented_train_ds


# ---------------------------------------------------------------------------
# Combined loader  (unchanged from original)
# ---------------------------------------------------------------------------

def build_combined_dataset(
    dataset_names: List[str],
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    normalized_cache_root: Optional[Path] = None,
    refresh_normalized_cache: bool = False,
    augmented_cache_root: Optional[Path] = None,
    refresh_augmented_cache: bool = False,
    augment_train: bool = True,
    curriculum: bool = False,
):
    from datasets import concatenate_datasets

    specs = [DATASET_SPECS[n] for n in dataset_names]
    if curriculum:
        specs = sorted(specs, key=lambda s: s.curriculum_order)

    train_parts, val_parts, manifest = [], [], []
    for spec in specs:
        train_ds, val_ds = _load_single_dataset(
            spec,
            max_train_samples,
            max_val_samples,
            normalized_cache_root=normalized_cache_root,
            refresh_normalized_cache=refresh_normalized_cache,
            augmented_cache_root=augmented_cache_root,
            refresh_augmented_cache=refresh_augmented_cache,
            augment_train=augment_train,
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
