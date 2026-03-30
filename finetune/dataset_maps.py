"""
finetune/dataset_maps.py
========================
Per-dataset raw label → production LABEL2ID key mappings.
Each dict translates a dataset's native vocabulary into the 6-class
label space defined in config.py.
"""
from __future__ import annotations

from typing import Dict

# PubLayNet category_id (1-indexed integers from COCO format)
PUBLAYNET_MAP: Dict[int, str] = {
    1: "paragraph",   # Text
    2: "heading",     # Title
    3: "list_item",   # List
    4: "table",       # Table
    5: "caption",     # Figure
}

# DocLayNet class names
DOCLAYNET_MAP: Dict[str, str] = {
    "Caption":        "caption",
    "Footnote":       "other",
    "Formula":        "other",
    "List-item":      "list_item",
    "Page-footer":    "other",
    "Page-header":    "heading",
    "Picture":        "caption",
    "Section-header": "heading",
    "Table":          "table",
    "Text":           "paragraph",
    "Title":          "heading",
}

# FUNSD entity types
FUNSD_MAP: Dict[str, str] = {
    "question": "heading",
    "answer":   "paragraph",
    "header":   "heading",
    "other":    "other",
}

# DocBank token-level layout labels
DOCBANK_MAP: Dict[str, str] = {
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