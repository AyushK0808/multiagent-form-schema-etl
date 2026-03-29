"""
Parallel extraction fabric.

Runs Donut (OCR-free, semantic generalisation) and the text/vision LLM pipeline
(layout-precise) concurrently via a ThreadPoolExecutor, then packages both
result sets — with spatial metadata where available — into an ExtractionBundle
that the downstream ReflexivePolicyLayer can fuse.

Design
------
* Donut   → image-only, high semantic coverage, no bbox output
* LayoutLM → text + bbox, precise localisation, layout-aware

Each extractor returns
    (fields: Dict[str, Any], confidences: Dict[str, float], metadata: Dict)

The metadata dict contains at minimum:
    source      : "donut" | "layoutlm"
    field_coverage : 0.0–1.0  (fraction of non-None fields)
    spatial_meta   : {field_name: bbox | None}
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExtractorResult:
    """Output of a single extractor."""
    source:        str                     # "donut" | "layoutlm"
    fields:        Dict[str, Any]          # field_name → value | None
    confidences:   Dict[str, float]        # field_name → 0.0–1.0
    spatial_meta:  Dict[str, Any]          # field_name → {"bbox": ..., "page": ...}
    elapsed_s:     float = 0.0
    error:         Optional[str] = None

    @property
    def field_coverage(self) -> float:
        if not self.fields:
            return 0.0
        filled = sum(1 for v in self.fields.values() if v is not None)
        return filled / len(self.fields)

    @property
    def mean_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences.values()) / len(self.confidences)


@dataclass
class ExtractionBundle:
    """Holds both extractor results for downstream fusion."""
    donut:    Optional[ExtractorResult] = None
    layoutlm: Optional[ExtractorResult] = None

    @property
    def all_field_names(self):
        names = set()
        if self.donut:
            names |= set(self.donut.fields)
        if self.layoutlm:
            names |= set(self.layoutlm.fields)
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
# Donut runner (wraps extraction.donut_extractor)
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
        logger.error(f"[Parallel] Donut runner failed: {exc}")
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
# LayoutLM / FormFiller runner
# ---------------------------------------------------------------------------

def _run_layoutlm(
    blocks: list,
    page_image: Image.Image,
    schema: Dict,
    clause_graph: Dict,
    full_text: str,
) -> ExtractorResult:
    t0 = time.perf_counter()
    try:
        from extraction.form_filler import FormFiller
        filler = FormFiller()
        form   = filler.populate(
            clause_graph,
            schema,
            full_text=full_text,
            page_image=page_image,
        )
        fields = form.fields

        # Build per-field confidence from form metadata
        conf_scores: Dict[str, float] = {}
        for fname in fields:
            # FormFiller stores 1.0 for text-LLM fills; we keep that or default 0.6
            raw_conf = form.metadata.get("confidence_scores", {}).get(fname, 0.6)
            # Penalise NaN sentinel
            conf_scores[fname] = 0.0 if fields[fname] == "NaN" else float(raw_conf)
            # Normalise NaN sentinel back to None for downstream
            if fields[fname] == "NaN":
                fields[fname] = None

        # Spatial metadata from blocks
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
        logger.error(f"[Parallel] LayoutLM runner failed: {exc}")
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
    max_workers    : thread pool size (default 2, one per extractor)
    """

    def __init__(
        self,
        donut_model_id: Optional[str] = None,
        max_workers: int = 2,
    ):
        self.donut_model_id = donut_model_id
        self.max_workers    = max_workers

    def extract(
        self,
        blocks: list,
        page_image: Image.Image,
        schema: Dict,
        clause_graph: Dict,
        full_text: str = "",
    ) -> ExtractionBundle:
        """
        Run both extractors concurrently.

        Parameters
        ----------
        blocks       : ingested DocumentBlock dicts (from ingestion.ingest_pdf)
        page_image   : PIL Image of document page
        schema       : field schema dict
        clause_graph : layout-derived clause mapping
        full_text    : concatenated document text
        """
        bundle: ExtractionBundle = ExtractionBundle()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures: Dict[str, Future] = {
                "donut": pool.submit(
                    _run_donut, page_image, schema, self.donut_model_id
                ),
                "layoutlm": pool.submit(
                    _run_layoutlm, blocks, page_image, schema, clause_graph, full_text
                ),
            }

            for name, fut in futures.items():
                try:
                    result = fut.result(timeout=300)  # 5 min safety cap
                    if name == "donut":
                        bundle.donut = result
                    else:
                        bundle.layoutlm = result
                except Exception as exc:
                    logger.error(f"[Parallel] {name} future raised: {exc}")

        logger.info(f"[Parallel] {bundle.summary()}")
        return bundle