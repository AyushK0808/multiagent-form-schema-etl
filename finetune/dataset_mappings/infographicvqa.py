"""
InfographicVQA-specific parsing helpers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def infographicvqa_question_type(question: str) -> str:
    """Heuristically classify an InfographicVQA question for segment labelling."""
    q = question.strip().lower()
    if any(w in q for w in ("table", "row", "column", "cell", "grid")):
        return "table"
    if any(w in q for w in ("title", "heading", "header", "label of")):
        return "heading"
    if any(w in q for w in ("list", "items", "bullet", "legend")):
        return "list_item"
    if any(w in q for w in ("caption", "figure", "chart", "graph", "image")):
        return "caption"
    return "paragraph"


def infographicvqa_ocr_token_label(token: Dict[str, Any]) -> str:
    raw = str(token.get("label") or token.get("type") or "").strip().lower()
    if raw in ("title", "heading", "header"):
        return "heading"
    if raw in ("table", "cell", "grid"):
        return "table"
    if raw in ("list", "list_item", "bullet", "legend"):
        return "list_item"
    if raw in ("caption", "figure", "chart"):
        return "caption"
    return "paragraph"


def infographicvqa_bbox(token: Dict[str, Any], width: int, height: int) -> Optional[List[float]]:
    bbox = token.get("bbox") or token.get("box")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    return None
