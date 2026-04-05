"""
SROIE-specific label mapping helpers.
"""
from __future__ import annotations

from typing import Any, Dict, List


SROIE_KEY_FIELDS = {"company", "date", "address", "total"}

# SROIE entities keys → layout label
# Matches the actual HF dataset schema: entities = {company, date, address, total}
SROIE_ENTITY_LABEL: Dict[str, str] = {
    "company": "heading",
    "date":    "other",
    "address": "paragraph",
    "total":   "list_item",
}

SROIE_TEXT_TO_LABEL: Dict[str, str] = {
    "invoice":  "heading",
    "receipt":  "heading",
    "tax":      "heading",
    "total":    "list_item",
    "amount":   "list_item",
    "subtotal": "list_item",
    "gst":      "list_item",
    "date":     "other",
    "time":     "other",
    "address":  "paragraph",
    "phone":    "paragraph",
    "company":  "heading",
    "store":    "heading",
}


def sroie_text_label(text: str) -> str:
    lower = text.strip().lower()
    for keyword, label in SROIE_TEXT_TO_LABEL.items():
        if keyword in lower:
            return label
    return "paragraph"


def sroie_entity_label(entity_key: str) -> str:
    """Map an entities-dict key (company/date/address/total) to a layout label."""
    return SROIE_ENTITY_LABEL.get(str(entity_key).strip().lower(), "paragraph")


def sroie_word_label(word: str, entities: Dict[str, str]) -> str:
    """Derive a layout label for a word by matching against entity values."""
    text = word.strip().lower()
    company = str(entities.get("company", "")).lower()
    total   = str(entities.get("total", "")).lower()
    date    = str(entities.get("date", "")).lower()
    address = str(entities.get("address", "")).lower()
    if company and company in text:
        return "heading"
    if total and total in text:
        return "list_item"
    if date and date in text:
        return "other"
    if address and any(part in text for part in address.split() if len(part) > 3):
        return "paragraph"
    return sroie_text_label(text)


def sroie_bbox_from_points(points: Any) -> List[float] | None:
    """Convert SROIE bbox [[x0,y0],[x1,y1],...] or flat list to [x0,y0,x1,y1]."""
    if not points:
        return None
    if isinstance(points, (list, tuple)):
        flat = []
        for p in points:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                flat.extend([float(p[0]), float(p[1])])
            elif isinstance(p, (int, float)):
                flat.append(float(p))
        if len(flat) >= 4:
            xs = flat[0::2]
            ys = flat[1::2]
            return [min(xs), min(ys), max(xs), max(ys)]
    return None
