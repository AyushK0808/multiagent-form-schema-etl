"""
finetune/metrics.py
===================
Evaluation helpers used by both trainers.

  - _levenshtein      : edit distance (no external deps)
  - compute_cer       : character error rate over a batch
  - assign_labels_by_containment : maps OCR word bboxes → segment labels
  - norm_bbox         : pixel → 0-1000 scale
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from config import LABEL2ID


# ---------------------------------------------------------------------------
# CER (character error rate)
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """Character error rate: total edit distance / total reference characters."""
    total_chars = total_edits = 0
    for pred, ref in zip(predictions, references):
        total_edits += _levenshtein(pred, ref)
        total_chars += len(ref)
    return total_edits / max(total_chars, 1)


# ---------------------------------------------------------------------------
# Spatial label assignment
# ---------------------------------------------------------------------------

def norm_bbox(bbox: List[float], img_w: int, img_h: int) -> List[int]:
    """Scale a pixel bounding box to the 0-1000 range LayoutLMv3 expects."""
    x0, y0, x1, y1 = bbox
    return [
        int(x0 * 1000 / max(img_w, 1)),
        int(y0 * 1000 / max(img_h, 1)),
        int(x1 * 1000 / max(img_w, 1)),
        int(y1 * 1000 / max(img_h, 1)),
    ]


def assign_labels_by_containment(
    word_bboxes_norm: List[Tuple[int, int, int, int]],   # 0-1000 scale
    segments_norm: List[Dict],                            # [{bbox: 0-1000, label: str}]
) -> List[int]:
    """
    For each OCR word (given by its normalised bbox), find the first annotated
    segment whose bbox contains the word's centre point.

    Falls back to LABEL2ID['other'] for any word not covered by a segment.
    This is the unified label-assignment algorithm for both word-level datasets
    (FUNSD, DocBank) and region-level datasets (PubLayNet, DocLayNet).
    """
    result = []
    for wb in word_bboxes_norm:
        cx = (wb[0] + wb[2]) / 2.0
        cy = (wb[1] + wb[3]) / 2.0
        label_id = LABEL2ID["other"]
        for seg in segments_norm:
            sx0, sy0, sx1, sy1 = seg["bbox"]
            if sx0 <= cx <= sx1 and sy0 <= cy <= sy1:
                label_id = LABEL2ID.get(seg["label"], LABEL2ID["other"])
                break
        result.append(label_id)
    return result