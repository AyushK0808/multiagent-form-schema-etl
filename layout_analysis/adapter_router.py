"""
layout_analysis/adapter_router.py
===================================
AdapterRouter: loads LoRA adapters on demand and routes incoming documents
to the correct adapter group at inference time.

Design
------
- One shared base model (frozen) is loaded once into memory.
- Adapters are hot-swapped using PEFT's set_adapter / load_adapter APIs.
- An LRU cache (max 3 entries = all groups) prevents reloading from disk
  on repeated calls to the same group.
- The router is a singleton accessed via get_adapter_router().

Integration points
------------------
- layout_analysis/layout_structure.py  → AdapterLayoutAnalyzer uses this
- extraction/parallel_extractor.py     → _run_layoutlm passes group_name
- orchestration/orchestrator.py        → _layout_node looks up group from
  state["schema_recognition"]["form_name"] via group_for_schema()

Fallback
--------
If the requested adapter directory does not exist (adapters not yet trained),
AdapterRouter transparently falls back to the HeuristicLayoutAnalyzer so the
pipeline never hard-fails.
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

_BASE_MODEL_ID = "microsoft/layoutlmv3-base"
_MAX_CACHED    = 3   # all three groups can live in memory simultaneously on GPU


class AdapterRouter:
    """
    Manages the shared LayoutLMv3 base model and swaps LoRA adapters
    per adapter group.

    Parameters
    ----------
    adapter_root : Path
        Root directory produced by train_lora.py, laid out as:
            <adapter_root>/
                group_1/layoutlmv3/  ← adapter_config.json + adapter weights
                group_2/layoutlmv3/
                group_3/layoutlmv3/
    """

    def __init__(self, adapter_root: Optional[Path] = None):
        if adapter_root is None:
            try:
                from config.config import get_config
                adapter_root = get_config().paths.project_root / "models" / "adapters"
            except Exception:
                adapter_root = Path("models") / "adapters"

        self.adapter_root = Path(adapter_root)
        self._base_model  = None
        self._processor   = None
        self._device      = None
        self._loaded_adapters: OrderedDict[str, bool] = OrderedDict()  # group_name → True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_model_for_group(self, group_name: str):
        """
        Return the PEFT model with the correct adapter active.
        Loads base + adapter lazily on first call.

        Returns (model, processor) or (None, None) if unavailable.
        """
        adapter_dir = self.adapter_root / group_name / "layoutlmv3"
        if not adapter_dir.exists():
            logger.warning(
                "[AdapterRouter] Adapter for '%s' not found at %s — "
                "falling back to heuristic analyzer",
                group_name, adapter_dir,
            )
            return None, None

        self._ensure_base_loaded(adapter_dir)
        self._ensure_adapter_loaded(group_name, adapter_dir)

        # Activate this group's adapter
        try:
            self._base_model.set_adapter(group_name)
        except Exception as exc:
            logger.error("[AdapterRouter] set_adapter failed: %s", exc)
            return None, None

        return self._base_model, self._processor

    def available_groups(self) -> List[str]:
        """Return group names for which an adapter checkpoint exists on disk."""
        groups = []
        for grp in ("group_1", "group_2", "group_3"):
            if (self.adapter_root / grp / "layoutlmv3").exists():
                groups.append(grp)
        return groups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_base_loaded(self, any_adapter_dir: Path) -> None:
        if self._base_model is not None:
            return

        import torch
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

        # Read label map from adapter dir (written by trainer)
        label_map_path = any_adapter_dir / "label_map.json"
        if label_map_path.exists():
            id2label = {int(k): v for k, v in json.loads(label_map_path.read_text()).items()}
            label2id = {v: k for k, v in id2label.items()}
            num_labels = len(id2label)
        else:
            from finetune.config import ID2LABEL, LABEL2ID, NUM_LABELS
            id2label, label2id, num_labels = ID2LABEL, LABEL2ID, NUM_LABELS

        logger.info("[AdapterRouter] Loading base model %s …", _BASE_MODEL_ID)
        self._processor = LayoutLMv3Processor.from_pretrained(
            _BASE_MODEL_ID, apply_ocr=True
        )
        base = LayoutLMv3ForTokenClassification.from_pretrained(
            _BASE_MODEL_ID,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base.to(self._device).eval()
        self._base_model = base
        logger.info("[AdapterRouter] Base model ready on %s", self._device)

    def _ensure_adapter_loaded(self, group_name: str, adapter_dir: Path) -> None:
        if group_name in self._loaded_adapters:
            return

        try:
            from peft import PeftModel
        except ImportError:
            raise RuntimeError("peft not installed. Run: pip install peft")

        logger.info("[AdapterRouter] Loading adapter '%s' from %s …", group_name, adapter_dir)

        # First adapter: wrap base with PeftModel
        if not self._loaded_adapters:
            self._base_model = PeftModel.from_pretrained(
                self._base_model,
                str(adapter_dir),
                adapter_name=group_name,
            )
        else:
            # Subsequent adapters: load into the existing PeftModel
            self._base_model.load_adapter(str(adapter_dir), adapter_name=group_name)

        # LRU eviction if we exceed cache size
        if len(self._loaded_adapters) >= _MAX_CACHED:
            oldest = next(iter(self._loaded_adapters))
            logger.debug("[AdapterRouter] Evicting adapter '%s' from cache", oldest)
            try:
                self._base_model.delete_adapter(oldest)
            except Exception:
                pass
            del self._loaded_adapters[oldest]

        self._loaded_adapters[group_name] = True
        logger.info("[AdapterRouter] Adapter '%s' loaded and cached", group_name)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_router: Optional[AdapterRouter] = None


def get_adapter_router(adapter_root: Optional[Path] = None) -> AdapterRouter:
    global _router
    if _router is None:
        _router = AdapterRouter(adapter_root=adapter_root)
    return _router