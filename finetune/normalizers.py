"""
finetune/normalizers.py
=======================
Per-dataset normalizers that convert raw HuggingFace dataset examples into a
unified intermediate dict:

    {
        "image"        : PIL.Image,
        "segments"     : [{"bbox": [x0,y0,x1,y1] in 0-1000, "label": str}],
        "label_text"   : str,   # Donut schema target
        "dataset_name" : str,
    }

Word-level datasets (FUNSD, DocBank, Kleister-NDA) express each annotated
word as a tiny 1-word segment so the same containment algorithm in metrics.py
works for both word-level and region-level annotations.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Dict

from PIL import Image

from config import DatasetSpec
from dataset_maps import DOCLAYNET_MAP, DOCBANK_MAP, FUNSD_MAP, PUBLAYNET_MAP
from metrics import norm_bbox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _open_image(value: Any) -> Image.Image:
    if value is None:
        raise ValueError("image value is None")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    raise ValueError(f"Unsupported image payload: {type(value)}")


# ---------------------------------------------------------------------------
# Per-dataset normalizers
# ---------------------------------------------------------------------------

def normalize_funsd(example: Dict) -> Dict:
    image = _open_image(example.get("image") or example.get("img"))
    w, h = image.size
    segments = []
    for entity in example.get("form", []):
        mapped = FUNSD_MAP.get(entity.get("label", "other"), "other")
        for word in entity.get("words", []):
            bbox = word.get("box", [0, 0, w, h])
            segments.append({"bbox": norm_bbox(bbox, w, h), "label": mapped})
    return {"image": image, "segments": segments,
            "label_text": "funsd_form", "dataset_name": "FUNSD"}


def normalize_publaynet(example: Dict) -> Dict:
    image = _open_image(example.get("image"))
    w, h = image.size
    objects = example.get("objects", {})
    cat_ids = objects.get("category", [])
    bboxes  = objects.get("bbox", [])   # COCO [x, y, width, height]
    segments = []
    for cat_id, bbox in zip(cat_ids, bboxes):
        label = PUBLAYNET_MAP.get(int(cat_id), "other")
        x, y, bw, bh = bbox
        segments.append({"bbox": norm_bbox([x, y, x + bw, y + bh], w, h), "label": label})
    return {"image": image, "segments": segments,
            "label_text": "publaynet_layout", "dataset_name": "PUBLAYNET"}


def normalize_doclaynet(example: Dict) -> Dict:
    image = _open_image(example.get("image"))
    w, h = image.size
    segments = []
    for ann in example.get("annotations", []):
        cat_name = ann.get("category_name", "Text")
        label    = DOCLAYNET_MAP.get(cat_name, "other")
        bbox     = ann.get("bbox", [0, 0, w, h])
        x, y, bw, bh = bbox[:4]
        segments.append({"bbox": norm_bbox([x, y, x + bw, y + bh], w, h), "label": label})
    return {"image": image, "segments": segments,
            "label_text": "doclaynet_layout", "dataset_name": "DOCLAYNET"}


def normalize_docbank(example: Dict) -> Dict:
    image = _open_image(example.get("image"))
    w, h = image.size
    tokens     = example.get("tokens", [])
    raw_labels = example.get("labels", ["paragraph"] * len(tokens))
    bboxes     = example.get("bboxes", [[0, 0, w, h]] * len(tokens))
    segments   = []
    for _, raw_label, bbox in zip(tokens, raw_labels, bboxes):
        label = DOCBANK_MAP.get(str(raw_label).lower(), "other")
        segments.append({"bbox": norm_bbox(bbox, w, h), "label": label})
    return {"image": image, "segments": segments,
            "label_text": "docbank_layout", "dataset_name": "DOCBANK"}


def normalize_kleister_nda(example: Dict) -> Dict:
    """
    Kleister-NDA has token-level NER tags but no layout bboxes.
    Empty segments → containment falls back to 'paragraph' for all tokens,
    which is correct (NDA clauses are body text; heading detection happens
    at inference via the production heuristic analyser).
    """
    image = _open_image(example.get("image") or example.get("img"))
    return {"image": image, "segments": [],
            "label_text": "kleister_nda", "dataset_name": "KLEISTER_NDA"}


def normalize_generic(example: Dict, spec: DatasetSpec) -> Dict:
    """Fallback for datasets with no layout annotations (RVL-CDIP, QA datasets)."""
    for key in ("image", "img", "document", "page_image"):
        if example.get(key) is not None:
            image = _open_image(example[key])
            break
    else:
        raise KeyError(f"No image field found. Keys: {list(example.keys())}")
    label_text = spec.schema_name or spec.name.lower()
    return {"image": image, "segments": [],
            "label_text": label_text, "dataset_name": spec.name}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

NORMALIZERS = {
    "FUNSD":        normalize_funsd,
    "PUBLAYNET":    normalize_publaynet,
    "DOCLAYNET":    normalize_doclaynet,
    "DOCBANK":      normalize_docbank,
    "KLEISTER_NDA": normalize_kleister_nda,
}