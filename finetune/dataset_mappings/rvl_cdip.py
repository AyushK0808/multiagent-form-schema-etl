"""
RVL-CDIP class mapping helpers.
"""
from __future__ import annotations

from typing import Any


RVL_CDIP_CLASSES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific_report",
    "scientific_publication",
    "specification",
    "file_folder",
    "news_article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]


def rvl_cdip_label_name(raw_label: Any) -> str:
    if isinstance(raw_label, int):
        if 0 <= raw_label < len(RVL_CDIP_CLASSES):
            return RVL_CDIP_CLASSES[raw_label]
        return "rvl_cdip_unknown"

    if isinstance(raw_label, str):
        text = raw_label.strip().lower().replace(" ", "_")
        return text or "rvl_cdip_unknown"

    return "rvl_cdip_unknown"

