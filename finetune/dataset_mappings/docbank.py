"""
DocBank-specific label mapping helpers.
"""
from __future__ import annotations

from typing import Any, Dict


# DocBank uses string token-level labels; map to the 6-class production space.
DOCBANK_LABEL_MAP: Dict[str, str] = {
    "title":      "heading",
    "section":    "heading",
    "paragraph":  "paragraph",
    "list":       "list_item",
    "table":      "table",
    "figure":     "caption",
    "caption":    "caption",
    "abstract":   "paragraph",
    "equation":   "other",
    "footer":     "other",
    "reference":  "paragraph",
    "date":       "other",
    "author":     "other",
}


def docbank_token_label(raw_label: Any) -> str:
    return DOCBANK_LABEL_MAP.get(str(raw_label).strip().lower(), "other")
