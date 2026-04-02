"""
extraction/parallel_extractor.py
==================================
Parallel extraction fabric.

Runs Donut (OCR-free) and the LayoutLM/FormFiller pipeline concurrently
via a ThreadPoolExecutor, then packages both result sets into an
ExtractionBundle for the downstream ReflexivePolicyLayer.

LoRA adapter routing
--------------------
Both _run_layoutlm and ParallelExtractor accept an `adapter_group` parameter
("group_1" | "group_2" | "group_3").  The group is resolved upstream by the
orchestrator and passed down here so the correct LoRA adapter is active
when FormFiller calls the LayoutLMv3 token classifier.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers  (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class ExtractorResult:
    source:        str
    fields:        Dict[str, Any]
    confidences:   Dict[str, float]
    spatial_meta:  Dict[str, Any]
    elapsed_s:     float = 0.0
    error:         Optional[str] = None

    @property
    def field_coverage(self) -> float:
        if not self.fields:
            return 0.0
        return sum(1 for v in self.fields.values() if v is not None) / len(self.fields)

    @property
    def mean_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences.values()) / len(self.confidences)


@dataclass
class ExtractionBundle:
    donut:    Optional[ExtractorResult] = None
    layoutlm: Optional[ExtractorResult] = None

    @property
    def all_field_names(self):
        names = set()
        if self.donut:    names |= set(self.donut.fields)
        if self.layoutlm: names |= set(self.layoutlm.fields)
        return names

    def summary(self) -> str:
        parts = []
        for res in (self.donut, self.layoutlm):
            if res:
                parts.append(
                    f"{res.source}: coverage={res.field_coverage:.2f} "
                    f"conf={res.mean_confidence:.2f} t={res.elapsed_s:.1f}s"
                    + (f" ERROR={res.error}" if res.error else "")
                )
        return " | ".join(parts) if parts else "empty bundle"


# ---------------------------------------------------------------------------
# Donut runner  (unchanged — Donut is OCR-free, no adapter routing needed)
# ---------------------------------------------------------------------------

def _run_donut(
    page_image: Image.Image,
    schema: Dict,
    model_id: Optional[str] = None,
) -> ExtractorResult:
    t0 = time.perf_counter()
    try:
        from extraction.donut_extractor import DonutExtractor
        kwargs = {"model_id": model_id} if model_id else {}
        extractor = DonutExtractor(**kwargs)
        fields, confidences = extractor.extract(page_image, schema)
        return ExtractorResult(
            source="donut",
            fields=fields,
            confidences=confidences,
            spatial_meta={k: {"bbox": None, "page": 0} for k in fields},
            elapsed_s=time.perf_counter() - t0,
        )
    except Exception as exc:
        logger.error("[Parallel] Donut runner failed: %s", exc)
        field_names = list(schema.get("fields", {}).keys())
        return ExtractorResult(
            source="donut",
            fields={k: None for k in field_names},
            confidences={k: 0.0 for k in field_names},
            spatial_meta={},
            elapsed_s=time.perf_counter() - t0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# LayoutLM / FormFiller runner — LoRA adapter group aware
# ---------------------------------------------------------------------------

def _run_layoutlm(
    blocks: list,
    page_image: Image.Image,
    schema: Dict,
    clause_graph: Dict,
    full_text: str,
    adapter_group: str = "group_2",
) -> ExtractorResult:
    """
    Run FormFiller with the LoRA adapter group that matches this document type.

    The `adapter_group` is forwarded to LayoutAnalyzer inside FormFiller via
    the `group_name` kwarg so the correct LoRA adapter is active during any
    LayoutLM inference calls within the populate() pipeline.
    """
    t0 = time.perf_counter()
    try:
        from extraction.form_filler import FormFiller
        filler = FormFiller(adapter_group=adapter_group)
        form   = filler.populate(
            clause_graph,
            schema,
            full_text=full_text,
            page_image=page_image,
        )
        fields = form.fields

        conf_scores: Dict[str, float] = {}
        for fname in fields:
            raw_conf = form.metadata.get("confidence_scores", {}).get(fname, 0.6)
            conf_scores[fname] = 0.0 if fields[fname] == "NaN" else float(raw_conf)
            if fields[fname] == "NaN":
                fields[fname] = None

        spatial: Dict[str, Any] = {}
        for b in blocks:
            for fname in fields:
                txt = str(b.get("text", ""))
                val = str(fields.get(fname) or "")
                if val and val in txt and fname not in spatial:
                    spatial[fname] = {"bbox": b.get("bbox"), "page": b.get("page", 0)}

        return ExtractorResult(
            source="layoutlm",
            fields=fields,
            confidences=conf_scores,
            spatial_meta=spatial,
            elapsed_s=time.perf_counter() - t0,
        )
    except Exception as exc:
        logger.error("[Parallel] LayoutLM runner failed: %s", exc)
        field_names = list(schema.get("fields", {}).keys())
        return ExtractorResult(
            source="layoutlm",
            fields={k: None for k in field_names},
            confidences={k: 0.0 for k in field_names},
            spatial_meta={},
            elapsed_s=time.perf_counter() - t0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Parallel orchestrator
# ---------------------------------------------------------------------------

class ParallelExtractor:
    """
    Launches Donut and LayoutLM concurrently and collects an ExtractionBundle.

    Parameters
    ----------
    donut_model_id : override the default Donut checkpoint
    max_workers    : thread pool size (default 2)
    adapter_group  : LoRA adapter group to activate for LayoutLM
                     ("group_1" | "group_2" | "group_3")
    """

    def __init__(
        self,
        donut_model_id: Optional[str] = None,
        max_workers: int = 2,
        adapter_group: str = "group_2",
    ):
        self.donut_model_id = donut_model_id
        self.max_workers    = max_workers
        self.adapter_group  = adapter_group

    def extract(
        self,
        blocks: list,
        page_image: Image.Image,
        schema: Dict,
        clause_graph: Dict,
        full_text: str = "",
    ) -> ExtractionBundle:
        bundle = ExtractionBundle()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures: Dict[str, Future] = {
                "donut": pool.submit(
                    _run_donut, page_image, schema, self.donut_model_id
                ),
                "layoutlm": pool.submit(
                    _run_layoutlm,
                    blocks, page_image, schema, clause_graph, full_text,
                    self.adapter_group,      # ← adapter group forwarded
                ),
            }

            for name, fut in futures.items():
                try:
                    result = fut.result(timeout=300)
                    if name == "donut":
                        bundle.donut = result
                    else:
                        bundle.layoutlm = result
                except Exception as exc:
                    logger.error("[Parallel] %s future raised: %s", name, exc)

        logger.info("[Parallel] %s", bundle.summary())
        return bundle