"""
CORD-specific parsing and label mapping helpers.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


def cord_parse_ground_truth(raw_ground_truth: Any) -> Dict[str, Any]:
    parsed: Any = raw_ground_truth
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed.get("gt_parse", parsed)


def cord_layout_label(path_tokens: Iterable[str]) -> str:
    joined = " ".join(str(token).lower() for token in path_tokens if token)

    if any(k in joined for k in ("table", "tabular", "grid")):
        return "table"
    if any(k in joined for k in ("menu", "item", "qty", "quantity", "price", "subtotal", "sub_total", "total", "tax")):
        return "list_item"
    if any(k in joined for k in ("store", "seller", "address", "phone", "business", "company", "header", "title", "date", "time")):
        return "heading"
    return "paragraph"


def _bbox_from_quad(quad: List[Dict[str, Any]]) -> List[float] | None:
    xs: List[float] = []
    ys: List[float] = []
    for point in quad:
        if not isinstance(point, dict):
            continue
        if "x" in point and "y" in point:
            xs.append(float(point["x"]))
            ys.append(float(point["y"]))
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def cord_word_bbox(word: Any) -> List[float] | None:
    if isinstance(word, dict):
        if isinstance(word.get("quad"), list):
            quad_bbox = _bbox_from_quad(word["quad"])
            if quad_bbox is not None:
                return quad_bbox
        for key in ("bbox", "box", "bounding_box"):
            bbox = word.get(key)
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    elif isinstance(word, (list, tuple)) and len(word) >= 4:
        return [float(word[0]), float(word[1]), float(word[2]), float(word[3])]
    return None

