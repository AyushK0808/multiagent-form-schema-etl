"""
DocLayNet category-id mapping helpers.
"""
from __future__ import annotations

from typing import Any


# Canonical DocLayNet class order used by category_id.
DOCLAYNET_ID_TO_NAME = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}


def doclaynet_category_name(raw_category_id: Any) -> str:
    if isinstance(raw_category_id, int):
        return DOCLAYNET_ID_TO_NAME.get(raw_category_id, "Text")
    return "Text"

