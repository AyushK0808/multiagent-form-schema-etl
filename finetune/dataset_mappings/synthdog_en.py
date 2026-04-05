"""
SynthDog-EN specific parsing helpers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


SYNTHDOG_LABEL_MAP: Dict[str, str] = {
    "title":   "heading",
    "header":  "heading",
    "heading": "heading",
    "table":   "table",
    "cell":    "table",
    "list":    "list_item",
    "item":    "list_item",
    "bullet":  "list_item",
    "caption": "caption",
    "figure":  "caption",
}


def synthdog_token_label(raw_label: Any) -> str:
    key = str(raw_label or "").strip().lower()
    for marker, label in SYNTHDOG_LABEL_MAP.items():
        if marker in key:
            return label
    return "paragraph"


def synthdog_bbox(token: Dict[str, Any], width: int, height: int) -> Optional[List[float]]:
    for key in ("bbox", "box", "bounding_box"):
        bbox = token.get(key)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    return None


def synthdog_parse_ground_truth(gt: Any) -> List[Dict]:
    """Extract word-level annotations from SynthDog ground_truth JSON."""
    import json
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except json.JSONDecodeError:
            return []
    if not isinstance(gt, dict):
        return []
    words = []
    for region in gt.get("valid_line", []) or gt.get("lines", []) or []:
        for word in (region.get("words") or region.get("tokens") or []):
            if isinstance(word, dict):
                words.append(word)
    return words
