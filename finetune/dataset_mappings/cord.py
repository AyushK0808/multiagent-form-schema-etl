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


# Maps every known CORD-v2 gt_parse key to a layout label.
# Leaf keys take priority over parent keys when both appear in a path.
_CORD_KEY_LABEL: Dict[str, str] = {
    # store / header info  (parent: store_info)
    "store_info":        "heading",
    "store_name":        "heading",
    "store_addr":        "heading",
    "biz_nm":            "heading",
    "branch_nm":         "heading",
    "tel":               "heading",
    "fax":               "heading",
    # menu line items  (parent: menu)
    "menu":              "list_item",
    "nm":                "list_item",
    "cnt":               "list_item",
    "price":             "list_item",
    "unitprice":         "list_item",
    "discountprice":     "list_item",
    "num":               "list_item",
    "itemsubtotal":      "list_item",
    "vatyn":             "list_item",
    "etc":               "list_item",
    # totals  (parent: sub_total)
    "sub_total":         "list_item",
    "subtotal_price":    "list_item",
    "tax_price":         "list_item",
    "discount_price":    "list_item",
    "service_price":     "list_item",
    "othersvc_price":    "list_item",
    "total":             "list_item",
    "total_price":       "list_item",
    "total_etc":         "list_item",
    "cashprice":         "list_item",
    "changeprice":       "list_item",
    "creditcardprice":   "list_item",
    "emoneyprice":       "list_item",
    # payment / meta
    "payment_info":      "other",
    "date":              "other",
    "time":              "other",
    "cashier":           "other",
    "void_menu":         "other",
    # table
    "table":             "table",
}


def cord_layout_label(path_tokens: Iterable[str]) -> str:
    """Return the layout label for a gt_parse path.

    Uses the deepest (last) token that appears in _CORD_KEY_LABEL so that
    leaf keys (e.g. 'store_name') override their parent ('store_info').
    """
    tokens = [str(t).lower() for t in path_tokens if t]
    # Walk from deepest to shallowest for most-specific match
    for token in reversed(tokens):
        if token in _CORD_KEY_LABEL:
            return _CORD_KEY_LABEL[token]
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

