"""
layout_analysis/layout_structure.py
====================================
Layout-aware structural modeling.

Backend selection priority
--------------------------
1. AdapterLayoutAnalyzer  — fine-tuned LoRA adapter found at adapter_root/
2. FineTunedLayoutAnalyzer — legacy monolithic checkpoint (backward compat)
3. HeuristicLayoutAnalyzer — zero-dependency regex fallback

The public API is identical across all three:
    analyzer = LayoutAnalyzer()
    result   = analyzer.analyze(blocks, page_image)
    # result["clause_graph"]  → {section_key: text, ...}
    # result["predictions"]   → [(word, label_str), ...]
    # result["num_clauses"]   → int

LoRA adapter training
---------------------
    python finetune/train_lora.py
    # Saves to models/adapters/group_{1,2,3}/layoutlmv3/

Legacy full fine-tune (still supported)
-----------------------------------------
    python finetune/train.py --model layoutlmv3
    # config.model.layout_model = "models/layoutlmv3-nda/checkpoint_best"
"""
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label constants  (must match finetune/config.py exactly)
# ---------------------------------------------------------------------------

LABEL2ID = {
    "paragraph": 0,
    "heading":   1,
    "list_item": 2,
    "table":     3,
    "caption":   4,
    "other":     5,
}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
LABEL_MAP  = ID2LABEL   # backward compat alias


# ---------------------------------------------------------------------------
# Shared heuristics
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    re.compile(r"^\d+(\.\d+)*\.?\s"),
    re.compile(r"^Article\s+\d+", re.I),
    re.compile(r"^Section\s+\d+", re.I),
    re.compile(r"^\(\d+\)\s"),
    re.compile(r"^[A-Z][A-Z\s]{3,}$"),
    re.compile(r"^[A-Z]\.\s"),
]


def _heuristic_label(text: str) -> str:
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
    text = heading_text.strip().rstrip(":")
    key  = re.sub(r"\s+", "_", text)
    key  = re.sub(r"[^\w]", "_", key).strip("_")
    return key[:64] if key else "section"


def _normalize_bbox(bbox: Tuple, width: int, height: int) -> List[int]:
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
# Backend 3 — Heuristic (always available)
# ---------------------------------------------------------------------------

class HeuristicLayoutAnalyzer:
    def analyze(self, blocks: List[Dict], page_image: Image.Image) -> Dict:
        words_all, labels_all = [], []
        for block in blocks:
            text  = block.get("text", "").strip()
            label = _heuristic_label(text) if text else "other"
            if label == "heading":
                words_all.append(text); labels_all.append("heading")
            else:
                for w in text.split():
                    words_all.append(w); labels_all.append(label)
        clause_graph, predictions = _build_clause_graph_from_labeled_words(words_all, labels_all)
        logger.info("[Heuristic] %d clauses", len(clause_graph))
        return {"clause_graph": clause_graph, "predictions": predictions,
                "num_clauses": len(clause_graph)}


# ---------------------------------------------------------------------------
# Backend 2 — Legacy monolithic fine-tuned checkpoint
# ---------------------------------------------------------------------------

class FineTunedLayoutAnalyzer:
    def __init__(self, model_path: str):
        import torch
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

        logger.info("[LayoutLMv3] Loading fine-tuned model from: %s", model_path)
        self.processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=False,
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info("[LayoutLMv3] Ready on %s", self.device)

    def analyze(self, blocks: List[Dict], page_image: Image.Image) -> Dict:
        import torch
        width, height = page_image.size
        words, boxes, word_block_map = [], [], []

        for b_idx, block in enumerate(blocks):
            text = block.get("text", "").strip()
            if not text:
                continue
            bbox = block.get("bbox")
            norm = _normalize_bbox(bbox, width, height) if bbox else [0, 0, 1000, 1000]
            for w in text.split():
                words.append(w); boxes.append(norm); word_block_map.append(b_idx)

        if not words:
            return {"clause_graph": {}, "predictions": [], "num_clauses": 0}

        predicted_labels = self._run_inference(words, boxes, page_image)
        return self._post_process(blocks, word_block_map, predicted_labels)

    def _run_inference(self, words, boxes, page_image) -> List[str]:
        import torch
        predicted_labels = []
        MAX_WORDS = 500
        for chunk_w, chunk_b in zip(
            [words[i:i+MAX_WORDS]  for i in range(0, len(words), MAX_WORDS)],
            [boxes[i:i+MAX_WORDS]  for i in range(0, len(boxes), MAX_WORDS)],
        ):
            enc = self.processor(
                page_image, chunk_w, boxes=chunk_b,
                truncation=True, padding="max_length", max_length=512,
                return_tensors="pt", return_offsets_mapping=True,
            )
            word_ids = enc.word_ids(batch_index=0)
            enc.pop("offset_mapping", None)
            enc_dev = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc_dev).logits
            token_preds = logits.argmax(-1).squeeze(0).cpu().tolist()
            prev_wid, word_pred = None, {}
            for tok_idx, wid in enumerate(word_ids):
                if wid is not None and wid != prev_wid:
                    word_pred[wid] = token_preds[tok_idx]
                prev_wid = wid
            for i in range(len(chunk_w)):
                predicted_labels.append(ID2LABEL.get(word_pred.get(i, LABEL2ID["other"]), "other"))
        return predicted_labels

    def _post_process(self, blocks, word_block_map, predicted_labels):
        block_votes: Dict[int, List[str]] = {}
        for wi, bi in enumerate(word_block_map):
            block_votes.setdefault(bi, []).append(
                predicted_labels[wi] if wi < len(predicted_labels) else "other"
            )
        final_words, final_labels = [], []
        for b_idx, block in enumerate(blocks):
            text = block.get("text", "").strip()
            if not text:
                continue
            votes  = block_votes.get(b_idx, ["paragraph"])
            majority = max(set(votes), key=votes.count)
            if majority == "heading":
                final_words.append(text); final_labels.append("heading")
            else:
                for w in text.split():
                    final_words.append(w); final_labels.append(majority)
        clause_graph, predictions = _build_clause_graph_from_labeled_words(final_words, final_labels)
        logger.info("[LayoutLMv3] %d clauses", len(clause_graph))
        return {"clause_graph": clause_graph, "predictions": predictions,
                "num_clauses": len(clause_graph)}


# ---------------------------------------------------------------------------
# Backend 1 — LoRA adapter-aware analyzer  (preferred)
# ---------------------------------------------------------------------------

class AdapterLayoutAnalyzer:
    """
    Uses the AdapterRouter to hot-swap the correct LoRA adapter for the
    document's schema group before running inference.

    Parameters
    ----------
    group_name  : one of "group_1", "group_2", "group_3".
                  If None, defaults to "group_2" (structural classification),
                  which is the most general-purpose group.
    adapter_root: override for the adapter root directory.
    """

    def __init__(
        self,
        group_name: Optional[str] = None,
        adapter_root: Optional[Path] = None,
    ):
        self.group_name   = group_name or "group_2"
        self.adapter_root = adapter_root
        self._fallback    = HeuristicLayoutAnalyzer()

    def analyze(self, blocks: List[Dict], page_image: Image.Image) -> Dict:
        import torch
        from layout_analysis.adapter_router import get_adapter_router

        router = get_adapter_router(self.adapter_root)
        model, processor = router.get_model_for_group(self.group_name)

        if model is None or processor is None:
            logger.warning(
                "[AdapterAnalyzer] No adapter for '%s' — using heuristic fallback",
                self.group_name,
            )
            return self._fallback.analyze(blocks, page_image)

        # Reuse the same inference + post-process logic from FineTunedLayoutAnalyzer
        _tmp = FineTunedLayoutAnalyzer.__new__(FineTunedLayoutAnalyzer)
        _tmp.processor = processor
        _tmp.model     = model
        _tmp.device    = next(model.parameters()).device

        return _tmp.analyze(blocks, page_image)


# ---------------------------------------------------------------------------
# Helpers for backend selection
# ---------------------------------------------------------------------------

def _adapter_root_has_any_group(adapter_root: Path) -> bool:
    for grp in ("group_1", "group_2", "group_3"):
        if (adapter_root / grp / "layoutlmv3").exists():
            return True
    return False


def _is_finetuned_checkpoint(path: str) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    has_config  = (p / "config.json").exists()
    has_weights = (p / "pytorch_model.bin").exists() or any(p.glob("*.safetensors"))
    if not (has_config and has_weights):
        return False
    try:
        cfg = json.loads((p / "config.json").read_text())
        if "num_labels" in cfg:
            return cfg.get("num_labels", 0) == len(LABEL2ID)
        elif "id2label" in cfg:
            return set(cfg["id2label"].values()) == set(LABEL2ID.keys())
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

class LayoutAnalyzer:
    """
    Public entry point.

    Backend selection order
    -----------------------
    1. LoRA adapter    — models/adapters/group_*/layoutlmv3/ exists
    2. Legacy checkpoint — config.model.layout_model is a valid checkpoint dir
    3. Heuristic        — always available fallback

    The group used for the LoRA adapter is determined at call time from the
    `group_name` kwarg passed to analyze(), allowing the orchestrator to
    route different document types to different adapter groups without
    creating a new LayoutAnalyzer instance.
    """

    def __init__(self, model_path: Optional[str] = None):
        try:
            from config.config import get_config
            cfg = get_config()
            self._adapter_root = cfg.paths.project_root / "models" / "adapters"
            if model_path is None:
                model_path = cfg.model.layout_model
        except Exception:
            self._adapter_root = Path("models") / "adapters"
            model_path = model_path or "microsoft/layoutlmv3-base"

        self._legacy_path    = model_path
        self._use_adapters   = _adapter_root_has_any_group(self._adapter_root)
        self._use_legacy     = (not self._use_adapters) and _is_finetuned_checkpoint(model_path)
        self._legacy_backend: Optional[FineTunedLayoutAnalyzer] = None
        self._heuristic      = HeuristicLayoutAnalyzer()

        if self._use_adapters:
            logger.info(
                "[LayoutAnalyzer] LoRA adapters detected at '%s' — using AdapterLayoutAnalyzer",
                self._adapter_root,
            )
        elif self._use_legacy:
            logger.info(
                "[LayoutAnalyzer] Legacy checkpoint at '%s' — using FineTunedLayoutAnalyzer",
                model_path,
            )
            self._legacy_backend = FineTunedLayoutAnalyzer(model_path)
        else:
            logger.info("[LayoutAnalyzer] No checkpoints found — using heuristic analyzer")

    def analyze(
        self,
        blocks: List[Dict],
        page_image: Image.Image,
        group_name: Optional[str] = None,
    ) -> Dict:
        """
        Parameters
        ----------
        blocks     : ingested DocumentBlock dicts
        page_image : PIL Image of the document page
        group_name : LoRA adapter group to use ("group_1"|"group_2"|"group_3").
                     If None and adapters are available, defaults to "group_2".
                     Ignored when using the legacy or heuristic backend.
        """
        if self._use_adapters:
            return AdapterLayoutAnalyzer(
                group_name=group_name or "group_2",
                adapter_root=self._adapter_root,
            ).analyze(blocks, page_image)

        if self._use_legacy and self._legacy_backend:
            return self._legacy_backend.analyze(blocks, page_image)

        return self._heuristic.analyze(blocks, page_image)


# ---------------------------------------------------------------------------
# Convenience wrapper (backward compat)
# ---------------------------------------------------------------------------

def layout_and_structure(
    blocks: List[Dict],
    page_image: Image.Image,
    group_name: Optional[str] = None,
) -> Dict:
    return LayoutAnalyzer().analyze(blocks, page_image, group_name=group_name)["clause_graph"]