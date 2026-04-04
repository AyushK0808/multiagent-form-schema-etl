"""
Helpers for selecting and auditing token label spaces.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from dataset_mappings.funsd import FUNSD_BIO_LABELS

logger = logging.getLogger(__name__)

LAYOUT_LABELS: List[str] = [
    "paragraph",
    "heading",
    "list_item",
    "table",
    "caption",
    "other",
]


def _iter_segment_labels(dataset) -> Iterable[str]:
    for segments in dataset["segments"]:
        if not segments:
            continue
        for seg in segments:
            label = seg.get("label")
            if label:
                yield str(label)


def _make_maps(labels: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def build_group_label_space(dataset_names: Sequence[str], train_dataset, val_dataset) -> Tuple[Dict[str, int], Dict[int, str]]:
    selected = sorted(set(dataset_names))

    # For FUNSD-only runs use the canonical FUNSD BIO tagset.
    if selected == ["FUNSD"]:
        return _make_maps(FUNSD_BIO_LABELS)

    labels = set(_iter_segment_labels(train_dataset)) | set(_iter_segment_labels(val_dataset))
    if not labels:
        labels = set(LAYOUT_LABELS)

    ordered: List[str] = [label for label in LAYOUT_LABELS if label in labels]
    extra = sorted(labels - set(ordered))
    ordered.extend(extra)
    if not ordered:
        ordered = list(LAYOUT_LABELS)

    return _make_maps(ordered)


def log_dataset_label_distribution(dataset, split_name: str, group_name: str) -> None:
    per_dataset_counts: Dict[str, Counter] = {}
    for ds_name, segments in zip(dataset["dataset_name"], dataset["segments"]):
        counter = per_dataset_counts.setdefault(ds_name, Counter())
        for seg in segments or []:
            counter[str(seg.get("label", ""))] += 1

    for ds_name in sorted(per_dataset_counts):
        counts = per_dataset_counts[ds_name]
        if not counts:
            logger.warning(
                "[LoRA-%s] %s/%s has no segment labels after normalization.",
                group_name,
                split_name,
                ds_name,
            )
            continue
        logger.info(
            "[LoRA-%s] %s/%s segment label counts: %s",
            group_name,
            split_name,
            ds_name,
            dict(sorted(counts.items())),
        )
        if len(counts) == 1:
            logger.warning(
                "[LoRA-%s] %s/%s is single-class (%s) before tokenization.",
                group_name,
                split_name,
                ds_name,
                next(iter(counts)),
            )