"""
Layout-aware structural modeling using LayoutLMv3.

Strategy
--------
1. If a fine-tuned checkpoint exists at the path given in config
   (model.layout_model), load it and use it for token classification.
2. Otherwise fall back to the fast regex/heuristic analyzer introduced
   after the original layoutlmv3-base proved unusable (its classifier
   head is always MISSING when loaded from the base checkpoint, yielding
   random predictions).

The public API is identical in both cases:
    analyzer = LayoutAnalyzer()
    result   = analyzer.analyze(blocks, page_image)
    # result["clause_graph"]  → {section_key: text, ...}
    # result["predictions"]   → [(word, label_str), ...]
    # result["num_clauses"]   → int

Fine-tuned model training
-------------------------
See finetune/train.py and finetune/data/dataset_generator.py.

    # generate synthetic data + train in one step:
    python finetune/train.py --generate --n_samples 300 --epochs 5

    # then point config.model.layout_model at the output:
    # config.model.layout_model = "models/layoutlmv3-nda/checkpoint_best"
"""

import re
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from PIL import Image
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
LABEL2ID = {
    "paragraph": 0,
    "heading":   1,
    "list_item": 2,
    "table":     3,
    "caption":   4,
    "other":     5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Keep old LABEL_MAP for any downstream code that imported it
LABEL_MAP = ID2LABEL


# ---------------------------------------------------------------------------
# Heuristic heading patterns (used in both analyzers)
# ---------------------------------------------------------------------------
_HEADING_PATTERNS = [
    re.compile(r"^\d+(\.\d+)*\.?\s"),          # "1.", "1.1 ...", "2.3.1 ..."
    re.compile(r"^Article\s+\d+", re.I),
    re.compile(r"^Section\s+\d+", re.I),
    re.compile(r"^\(\d+\)\s"),                 # "(1) ..."
    re.compile(r"^[A-Z][A-Z\s]{3,}$"),         # ALL-CAPS e.g. "GOVERNING LAW"
    re.compile(r"^[A-Z]\.\s"),                  # "A. ..."
]


def _heuristic_label(text: str) -> str:
    """Return a string label from heuristic rules."""
    stripped = text.strip()
    if not stripped:
        return "other"
    for pat in _HEADING_PATTERNS:
        if pat.search(stripped):
            return "heading"
    if len(stripped) < 60 and stripped.endswith(":"):
        return "heading"
    if re.match(r"^[\-\*\•]\s", stripped) or re.match(r"^\([a-zA-Z]\)\s", stripped):
        return "list_item"
    return "paragraph"


def _section_key(heading_text: str) -> str:
    """Derive a stable dict key from a heading string."""
    text = heading_text.strip().rstrip(":")
    key  = re.sub(r"\s+", "_", text)
    key  = re.sub(r"[^\w]", "_", key).strip("_")
    return key[:64] if key else "section"


def _normalize_bbox(bbox: Tuple, width: int, height: int) -> List[int]:
    """Normalize a PDF bbox to LayoutLMv3's 0-1000 scale."""
    x0, y0, x1, y1 = bbox
    return [
        int(1000 * x0 / max(width, 1)),
        int(1000 * y0 / max(height, 1)),
        int(1000 * x1 / max(width, 1)),
        int(1000 * y1 / max(height, 1)),
    ]


def _build_clause_graph_from_labeled_words(
    words: List[str], labels: List[str]
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Shared post-processing: build a clause graph from (word, label) pairs.
    Returns (clause_graph, predictions_list).
    """
    clause_graph: Dict[str, str] = {}
    predictions:  List[Tuple[str, str]] = []

    current_key:    Optional[str] = None
    current_tokens: List[str]     = []

    def _flush():
        nonlocal current_tokens
        if current_key and current_tokens:
            existing = clause_graph.get(current_key, "")
            clause_graph[current_key] = (existing + " " + " ".join(current_tokens)).strip()
        current_tokens.clear()

    for word, label in zip(words, labels):
        predictions.append((word, label))
        if label == "heading":
            _flush()
            current_key = _section_key(word)
        else:
            if current_key is None:
                current_key = "preamble"
            current_tokens.append(word)

    _flush()
    return clause_graph, predictions


# ---------------------------------------------------------------------------
# Heuristic analyzer (no model required)
# ---------------------------------------------------------------------------

class HeuristicLayoutAnalyzer:
    """
    Fast block-level classifier using regex rules.
    Zero dependencies beyond the standard library + Pillow.
    """

    def analyze(self, blocks: List[Dict], page_image: Image.Image) -> Dict:
        words_all:  List[str] = []
        labels_all: List[str] = []

        for block in blocks:
            text = block.get("text", "").strip()
            if not text:
                continue
            label = _heuristic_label(text)
            if label == "heading":
                words_all.append(text)
                labels_all.append("heading")
            else:
                for w in text.split():
                    words_all.append(w)
                    labels_all.append(label)

        clause_graph, predictions = _build_clause_graph_from_labeled_words(
            words_all, labels_all
        )
        logger.info(f"[Heuristic] Found {len(clause_graph)} clauses")
        return {
            "clause_graph": clause_graph,
            "predictions":  predictions,
            "num_clauses":  len(clause_graph),
        }


# ---------------------------------------------------------------------------
# Fine-tuned LayoutLMv3 analyzer
# ---------------------------------------------------------------------------

class FineTunedLayoutAnalyzer:
    """
    Token classifier using a fine-tuned LayoutLMv3ForTokenClassification model.

    The model must have been trained with the label set in LABEL2ID above
    (see finetune/train.py).  Pass ignore_mismatched_sizes=False here because
    the checkpoint *should* contain the classifier head.
    """

    def __init__(self, model_path: str):
        import torch
        from transformers import (
            LayoutLMv3Processor,
            LayoutLMv3ForTokenClassification,
        )

        logger.info(f"[LayoutLMv3] Loading fine-tuned model from: {model_path}")
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_path, apply_ocr=False
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=False,  # fine-tuned checkpoint has classifier head
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"[LayoutLMv3] Model ready on {self.device}")

    def analyze(self, blocks: List[Dict], page_image: Image.Image) -> Dict:
        import torch

        width, height = page_image.size
        words: List[str]       = []
        boxes: List[List[int]] = []
        word_block_map: List[int] = []    # word idx → block idx

        for b_idx, block in enumerate(blocks):
            text = block.get("text", "").strip()
            if not text:
                continue
            bbox = block.get("bbox")
            if bbox:
                norm = _normalize_bbox(bbox, width, height)
            else:
                norm = [0, 0, 1000, 1000]

            for w in text.split():
                words.append(w)
                boxes.append(norm)
                word_block_map.append(b_idx)

        if not words:
            logger.warning("[LayoutLMv3] No words to process")
            return {"clause_graph": {}, "predictions": [], "num_clauses": 0}

        # Encode — LayoutLMv3 accepts up to 512 tokens; chunk if needed
        MAX_WORDS = 500   # conservative: 1 word ≈ 1.1 tokens on average
        chunks = [words[i:i+MAX_WORDS] for i in range(0, len(words), MAX_WORDS)]
        boxes_chunks = [boxes[i:i+MAX_WORDS] for i in range(0, len(boxes), MAX_WORDS)]

        predicted_labels: List[str] = []

        for chunk_words, chunk_boxes in zip(chunks, boxes_chunks):
            encoding = self.processor(
                page_image,
                chunk_words,
                boxes=chunk_boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            word_ids = encoding.word_ids(batch_index=0)
            encoding.pop("offset_mapping", None)

            enc_device = {k: v.to(self.device) for k, v in encoding.items()}
            with torch.no_grad():
                logits = self.model(**enc_device).logits   # (1, seq_len, num_labels)

            token_preds = logits.argmax(-1).squeeze(0).cpu().tolist()

            # Map back: take first sub-word prediction for each word
            prev_wid = None
            word_pred: Dict[int, int] = {}
            for token_idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid != prev_wid:          # first sub-word
                    word_pred[wid] = token_preds[token_idx]
                prev_wid = wid

            for i in range(len(chunk_words)):
                label_id = word_pred.get(i, LABEL2ID["other"])
                predicted_labels.append(ID2LABEL.get(label_id, "other"))

        # Post-process: group by block, take majority vote for block label
        block_label_votes: Dict[int, List[str]] = {}
        for word_idx, b_idx in enumerate(word_block_map):
            block_label_votes.setdefault(b_idx, []).append(
                predicted_labels[word_idx] if word_idx < len(predicted_labels) else "other"
            )

        # Rebuild word list with block-majority labels
        final_words:  List[str] = []
        final_labels: List[str] = []

        for b_idx, block in enumerate(blocks):
            text = block.get("text", "").strip()
            if not text:
                continue
            votes = block_label_votes.get(b_idx, ["paragraph"])
            majority_label = max(set(votes), key=votes.count)

            if majority_label == "heading":
                final_words.append(text)
                final_labels.append("heading")
            else:
                for w in text.split():
                    final_words.append(w)
                    final_labels.append(majority_label)

        clause_graph, predictions = _build_clause_graph_from_labeled_words(
            final_words, final_labels
        )
        logger.info(f"[LayoutLMv3] Found {len(clause_graph)} clauses")
        return {
            "clause_graph": clause_graph,
            "predictions":  predictions,
            "num_clauses":  len(clause_graph),
        }


# ---------------------------------------------------------------------------
# Factory — picks the right analyzer automatically
# ---------------------------------------------------------------------------

def _is_finetuned_checkpoint(path: str) -> bool:
    """
    Return True only if `path` is a directory containing a fine-tuned
    LayoutLMv3 classifier (i.e. it has a config.json AND pytorch_model.bin
    or model.safetensors, and the config mentions num_labels > 1).
    """
    p = Path(path)
    if not p.is_dir():
        return False
    has_config  = (p / "config.json").exists()
    has_weights = (p / "pytorch_model.bin").exists() or any(p.glob("*.safetensors"))
    if not (has_config and has_weights):
        return False
    # Check that the config actually has num_labels set to our label count
    try:
        import json
        cfg = json.loads((p / "config.json").read_text())
        return cfg.get("num_labels", 0) == len(LABEL2ID)
    except Exception:
        return False


class LayoutAnalyzer:
    """
    Public entry point.  Transparently uses the fine-tuned model when available,
    otherwise falls back to the heuristic analyzer.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            try:
                from config.config import get_config
                model_path = get_config().model.layout_model
            except Exception:
                model_path = "microsoft/layoutlmv3-base"

        if _is_finetuned_checkpoint(model_path):
            logger.info(f"Fine-tuned checkpoint detected at '{model_path}' — using LayoutLMv3")
            self._backend = FineTunedLayoutAnalyzer(model_path)
        else:
            logger.info(
                f"No fine-tuned checkpoint at '{model_path}' "
                "— using heuristic analyzer (run finetune/train.py to train)"
            )
            self._backend = HeuristicLayoutAnalyzer()

    def analyze(self, blocks: List[Dict], page_image: Image.Image) -> Dict:
        return self._backend.analyze(blocks, page_image)


# ---------------------------------------------------------------------------
# Convenience wrapper (backwards compat)
# ---------------------------------------------------------------------------

def layout_and_structure(blocks: List[Dict], page_image: Image.Image) -> Dict:
    return LayoutAnalyzer().analyze(blocks, page_image)["clause_graph"]