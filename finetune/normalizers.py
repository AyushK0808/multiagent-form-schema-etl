"""
finetune/normalizers.py
=======================
Per-dataset normalizers that convert raw HuggingFace dataset examples into a
unified intermediate dict:

    {
        "image"        : PIL.Image,
        "segments"     : [{"bbox": [x0,y0,x1,y1] in 0-1000, "label": str}],
        "label_text"   : str,
        "dataset_name" : str,
    }
"""
from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, Iterable, List

from PIL import Image

from config import DatasetSpec
from dataset_maps import DOCLAYNET_MAP, DOCBANK_MAP, PUBLAYNET_MAP
from dataset_mappings.cord import cord_layout_label, cord_parse_ground_truth, cord_word_bbox
from dataset_mappings.doclaynet import doclaynet_category_name
from dataset_mappings.funsd import funsd_ner_tag_to_label, funsd_word_bio_label
from dataset_mappings.rvl_cdip import rvl_cdip_label_name
from metrics import norm_bbox

logger = logging.getLogger(__name__)

NORMALIZATION_VERSION = 9


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


def _coerce_bbox(bbox: Any, width: int, height: int) -> List[int]:
    if not bbox:
        return [0, 0, width, height]
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x0", "y0", "x1", "y1")):
            return [bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]]
        if all(k in bbox for k in ("left", "top", "right", "bottom")):
            return [bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]]
        if all(k in bbox for k in ("x", "y", "w", "h")):
            return [bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]]
        if all(k in bbox for k in ("x", "y", "width", "height")):
            return [bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]]
    if isinstance(bbox, (list, tuple)):
        vals = list(bbox)
        if len(vals) >= 4:
            x0, y0, x1, y1 = vals[:4]
            if x1 <= x0 or y1 <= y0:
                return [x0, y0, x0 + x1, y0 + y1]
            return [x0, y0, x1, y1]
    return [0, 0, width, height]


def _extract_segments_from_words(words: Iterable[Any], width: int, height: int, label: str) -> List[Dict]:
    segments: List[Dict] = []
    for word in words:
        bbox = None
        if isinstance(word, dict):
            bbox = word.get("bbox") or word.get("box") or word.get("bounding_box")
        elif isinstance(word, (list, tuple)) and len(word) >= 4:
            bbox = word
        if bbox is None:
            continue
        segments.append({
            "bbox": norm_bbox(_coerce_bbox(bbox, width, height), width, height),
            "label": label,
        })
    return segments


def normalize_funsd(example: Dict) -> Dict:
    image = _open_image(example.get("image") or example.get("img"))
    w, h = image.size
    segments = []

    # Current HF parquet format (nielsr/funsd): token-level fields.
    # keys: words, bboxes, ner_tags, image
    bboxes = example.get("bboxes") or []
    ner_tags = example.get("ner_tags") or []
    if bboxes and ner_tags:
        for bbox, tag in zip(bboxes, ner_tags):
            segments.append(
                {
                    "bbox": norm_bbox(_coerce_bbox(bbox, w, h), w, h),
                    "label": funsd_ner_tag_to_label(tag),
                }
            )
        return {"image": image, "segments": segments, "label_text": "funsd_form", "dataset_name": "FUNSD"}

    # Legacy FUNSD-style format fallback (form -> words -> box + entity label)
    for entity in example.get("form", []):
        raw_label = entity.get("label", "other")
        for word_index, word in enumerate(entity.get("words", [])):
            bbox = word.get("box", [0, 0, w, h])
            segments.append(
                {
                    "bbox": norm_bbox(_coerce_bbox(bbox, w, h), w, h),
                    "label": funsd_word_bio_label(raw_label, word_index),
                }
            )
    return {"image": image, "segments": segments, "label_text": "funsd_form", "dataset_name": "FUNSD"}


def normalize_cord(example: Dict) -> Dict:
    image = _open_image(example.get("image") or example.get("img"))
    w, h = image.size
    segments = []
    gt_parse = cord_parse_ground_truth(example.get("ground_truth"))

    # Format A: gt_parse contains valid_line[{category, words:[{quad,text}]}]
    valid_lines = gt_parse.get("valid_line") or []
    if valid_lines:
        for line in valid_lines:
            line_label = cord_layout_label([line.get("category", "")])
            for word in line.get("words") or []:
                bbox = cord_word_bbox(word)
                if bbox is None:
                    continue
                segments.append({
                    "bbox": norm_bbox(_coerce_bbox(bbox, w, h), w, h),
                    "label": line_label,
                })
        if segments:
            return {"image": image, "segments": segments, "label_text": "cord_receipt", "dataset_name": "CORD"}

    # Format B: semantic-only gt_parse (nm/cnt/price under menu, plus
    # store_info / sub_total / payment_info sections).
    # Walk each TOP-LEVEL section separately so the section key — not the
    # leaf key — drives the label, then assign a canonical vertical zone
    # for that section so the model sees realistic spatial distribution.
    #
    # Typical receipt vertical layout (normalised 0-1000):
    #   store header  0  – 150
    #   menu items  150  – 700
    #   sub-totals  700  – 850
    #   payment     850  – 950
    #   meta/date   950  – 1000
    SECTION_ZONES: Dict[str, tuple] = {
        "store_info":    ("heading",   0,   150),
        "menu":          ("list_item", 150, 700),
        "sub_total":     ("list_item", 700, 850),
        "payment_info":  ("other",     850, 950),
        "void_menu":     ("other",     850, 950),
    }
    DEFAULT_ZONE = ("paragraph", 0, 1000)

    for section_key, section_val in gt_parse.items():
        label, zone_y0, zone_y1 = SECTION_ZONES.get(section_key, DEFAULT_ZONE)

        # Collect every leaf string in this section
        def _leaves(node: Any) -> List[str]:
            if isinstance(node, dict):
                return [v for sub in node.values() for v in _leaves(sub)]
            if isinstance(node, list):
                return [v for item in node for v in _leaves(item)]
            return [str(node)] if node is not None else []

        leaf_values = _leaves(section_val)
        if not leaf_values:
            continue

        # Distribute leaves evenly within the section's vertical zone
        zone_h = max(zone_y1 - zone_y0, 1)
        step = zone_h / len(leaf_values)
        for i, _ in enumerate(leaf_values):
            y0 = int(zone_y0 + i * step)
            y1 = int(zone_y0 + (i + 1) * step)
            segments.append({
                "bbox": norm_bbox([0, y0, w, y1], w, h),
                "label": label,
            })

    return {"image": image, "segments": segments, "label_text": "cord_receipt", "dataset_name": "CORD"}


def normalize_sroie(example: Dict) -> Dict:
    image = _open_image(example.get("image") or example.get("img"))
    w, h = image.size
    segments = []

    words = example.get("words") or []
    bboxes = example.get("bboxes") or []
    entities: Dict = example.get("entities") or {}

    # Build a set of entity value substrings for fast label lookup
    company_val = str(entities.get("company", "")).lower()
    date_val    = str(entities.get("date", "")).lower()
    address_val = str(entities.get("address", "")).lower()
    total_val   = str(entities.get("total", "")).lower()

    for word, bbox in zip(words, bboxes):
        text = str(word).lower()
        if company_val and company_val in text:
            label = "heading"
        elif total_val and total_val in text:
            label = "list_item"
        elif date_val and date_val in text:
            label = "other"
        elif address_val and any(part in text for part in address_val.split() if len(part) > 3):
            label = "paragraph"
        else:
            label = "paragraph"
        segments.append({"bbox": norm_bbox(_coerce_bbox(bbox, w, h), w, h), "label": label})

    return {"image": image, "segments": segments, "label_text": "sroie_invoice", "dataset_name": "SROIE"}


def normalize_publaynet(example: Dict) -> Dict:
    image = _open_image(example.get("image"))
    w, h = image.size
    objects = example.get("objects", {})
    cat_ids = objects.get("category", [])
    bboxes = objects.get("bbox", [])
    segments = []
    for cat_id, bbox in zip(cat_ids, bboxes):
        label = PUBLAYNET_MAP.get(int(cat_id), "other")
        x, y, bw, bh = bbox
        segments.append({"bbox": norm_bbox([x, y, x + bw, y + bh], w, h), "label": label})
    return {"image": image, "segments": segments, "label_text": "publaynet_layout", "dataset_name": "PUBLAYNET"}


def normalize_doclaynet(example: Dict) -> Dict:
    image = _open_image(example.get("image"))
    w, h = image.size
    segments = []

    # Format A: annotation objects
    annotations = example.get("annotations", [])
    if annotations:
        for ann in annotations:
            cat_name = ann.get("category_name", "Text")
            label = DOCLAYNET_MAP.get(cat_name, "other")
            bbox = ann.get("bbox", [0, 0, w, h])
            x, y, bw, bh = bbox[:4]
            segments.append({"bbox": norm_bbox([x, y, x + bw, y + bh], w, h), "label": label})
        return {"image": image, "segments": segments, "label_text": "doclaynet_layout", "dataset_name": "DOCLAYNET"}

    # Format B: parallel arrays from docling-project/DocLayNet-v1.2
    category_ids = example.get("category_id", [])
    bboxes = example.get("bboxes", [])
    for cat_id, bbox in zip(category_ids, bboxes):
        cat_name = doclaynet_category_name(cat_id)
        label = DOCLAYNET_MAP.get(cat_name, "other")
        x0, y0, x1, y1 = _coerce_bbox(bbox, w, h)
        segments.append({"bbox": norm_bbox([x0, y0, x1, y1], w, h), "label": label})
    return {"image": image, "segments": segments, "label_text": "doclaynet_layout", "dataset_name": "DOCLAYNET"}


def normalize_docbank(example: Dict) -> Dict:
    image = _open_image(example.get("image"))
    w, h = image.size
    raw_labels = example.get("labels", [])
    bboxes = example.get("bboxes", [[0, 0, w, h]] * len(raw_labels))
    segments = []
    for raw_label, bbox in zip(raw_labels, bboxes):
        label = DOCBANK_MAP.get(str(raw_label).lower(), "other")
        segments.append({"bbox": norm_bbox(bbox, w, h), "label": label})
    return {"image": image, "segments": segments, "label_text": "docbank_layout", "dataset_name": "DOCBANK"}


def normalize_docvqa(example: Dict) -> Dict:
    image = _open_image(example.get("image") or example.get("img") or example.get("document") or example.get("page_image"))
    w, h = image.size
    segments = []
    for token in example.get("ocr_info") or example.get("ocr_tokens") or []:
        if not isinstance(token, dict):
            continue
        bbox = token.get("bbox") or token.get("box")
        if bbox is None:
            continue
        segments.append({"bbox": norm_bbox(_coerce_bbox(bbox, w, h), w, h), "label": "paragraph"})
    return {"image": image, "segments": segments, "label_text": "docvqa_reasoning", "dataset_name": "DOCVQA"}


def normalize_kleister_nda(example: Dict) -> Dict:
    image = _open_image(example.get("image") or example.get("img"))
    return {"image": image, "segments": [], "label_text": "kleister_nda", "dataset_name": "KLEISTER_NDA"}


def normalize_rvl_cdip(example: Dict) -> Dict:
    image = _open_image(example.get("image") or example.get("img"))
    w, h = image.size
    raw_label = example.get("label", example.get("labels"))
    class_name = rvl_cdip_label_name(raw_label)
    # RVL-CDIP is document-level classification, so assign a full-page segment.
    segments = [{"bbox": norm_bbox([0, 0, w, h], w, h), "label": class_name}]
    return {"image": image, "segments": segments, "label_text": class_name, "dataset_name": "RVL-CDIP"}


def normalize_infographicvqa(example: Dict) -> Dict:
    image = _open_image(example.get("image") or example.get("img") or example.get("document") or example.get("page_image"))
    w, h = image.size
    segments = []
    for token in example.get("ocr_tokens") or example.get("ocr_info") or example.get("words") or []:
        if not isinstance(token, dict):
            continue
        bbox = token.get("bbox") or token.get("box")
        if bbox is None:
            continue
        segments.append({"bbox": norm_bbox(_coerce_bbox(bbox, w, h), w, h), "label": "paragraph"})
    return {"image": image, "segments": segments, "label_text": "infographic_vqa", "dataset_name": "INFOGRAPHICVQA"}


def normalize_synthdog_en(example: Dict) -> Dict:
    image = _open_image(example.get("image") or example.get("img"))
    w, h = image.size
    segments = []
    for token in example.get("words") or example.get("ocr_tokens") or example.get("tokens") or example.get("annotations") or []:
        if not isinstance(token, dict):
            continue
        bbox = token.get("bbox") or token.get("box") or token.get("bounding_box")
        if bbox is None:
            continue
        raw_label = str(token.get("label") or token.get("type") or "paragraph").lower()
        label = "paragraph"
        if any(mark in raw_label for mark in ("title", "header", "heading")):
            label = "heading"
        elif any(mark in raw_label for mark in ("table", "cell")):
            label = "table"
        elif any(mark in raw_label for mark in ("list", "item", "bullet")):
            label = "list_item"
        segments.append({"bbox": norm_bbox(_coerce_bbox(bbox, w, h), w, h), "label": label})
    return {"image": image, "segments": segments, "label_text": "synthdog_ocr", "dataset_name": "SYNTHDOG_EN"}


def normalize_generic(example: Dict, spec: DatasetSpec) -> Dict:
    for key in ("image", "img", "document", "page_image"):
        if example.get(key) is not None:
            image = _open_image(example[key])
            break
    else:
        raise KeyError(f"No image field found. Keys: {list(example.keys())}")
    label_text = spec.schema_name or spec.name.lower()
    return {"image": image, "segments": [], "label_text": label_text, "dataset_name": spec.name}


NORMALIZERS = {
    "CORD": normalize_cord,
    "DOCBANK": normalize_docbank,
    "DOCLAYNET": normalize_doclaynet,
    "DOCVQA": normalize_docvqa,
    "FUNSD": normalize_funsd,
    "INFOGRAPHICVQA": normalize_infographicvqa,
    "KLEISTER_NDA": normalize_kleister_nda,
    "PUBLAYNET": normalize_publaynet,
    "RVL-CDIP": normalize_rvl_cdip,
    "SROIE": normalize_sroie,
    "SYNTHDOG_EN": normalize_synthdog_en,
}