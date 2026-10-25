"""
DocLayNet category-id mapping helpers.
"""
from __future__ import annotations

from typing import Any


# Canonical DocLayNet class order used by category_id (COCO-style, 1-based).
DOCLAYNET_ID_TO_NAME = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}


def doclaynet_category_name(raw_category_id: Any) -> str:
    if isinstance(raw_category_id, int):
        return DOCLAYNET_ID_TO_NAME.get(raw_category_id, "Text")
    return "Text"

