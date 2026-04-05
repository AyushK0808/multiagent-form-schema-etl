"""
Kleister-NDA specific parsing and label mapping helpers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# NDA clause types → layout label
KLEISTER_NDA_ENTITY_MAP: Dict[str, str] = {
    "party":              "heading",
    "effective_date":     "other",
    "jurisdiction":       "paragraph",
    "term":               "paragraph",
    "governing_law":      "paragraph",
    "confidentiality":    "paragraph",
    "termination":        "paragraph",
    "amendment":          "paragraph",
    "arbitration":        "paragraph",
    "indemnification":    "paragraph",
    "liability":          "paragraph",
    "non_compete":        "paragraph",
    "non_solicitation":   "paragraph",
    "ip_ownership":       "paragraph",
    "signature":          "other",
}

# Fields that are spatially anchored (prefer LayoutLM over Donut at fusion)
SPATIAL_FIELDS = {"effective_date", "signature", "party"}


def kleister_nda_entity_label(entity_type: str) -> str:
    return KLEISTER_NDA_ENTITY_MAP.get(str(entity_type).strip().lower(), "paragraph")


def kleister_nda_is_spatial(entity_type: str) -> bool:
    return str(entity_type).strip().lower() in SPATIAL_FIELDS


def kleister_nda_bbox(span: Any, width: int, height: int) -> Optional[List[float]]:
    if isinstance(span, dict):
        bbox = span.get("bbox") or span.get("box")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    return None
