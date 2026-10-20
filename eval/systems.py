"""
eval/systems.py
===============
Computes each branch's raw output for a document once (disk-cached) and derives
every compared system from those outputs without further model calls.

Branches (cached per document)
------------------------------
donut      OCR-free generator, all pages
layout     layout-conditioned LLM extractor: text LLM, then vision LLM for nulls
text_pass  the layout branch's first (text-only) pass
vlm        vision LLM given the page images and the whole schema, no fusion

Systems (derived offline)
-------------------------
full               fusion policy + schema-agent repair
full_no_agent      fusion policy only
full_no_norm       fusion without value normalisation
full_no_spatial    fusion without the spatial-field preference rule
donut_only         Donut branch alone
layout_only        layout branch alone
text_only          text LLM first pass alone
vlm_only           vision LLM alone
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config import get_config
from extraction.parallel_extractor import (
    ExtractionBundle,
    ExtractorResult,
    _run_donut,
    _run_layoutlm,
)
from extraction.policy_layer import ReflexivePolicyLayer
from utils.llm_cache import usage_delta, usage_snapshot

logger = logging.getLogger(__name__)

BRANCH_CACHE = Path("data") / "cache" / "branches"
SYSTEMS = [
    "full", "full_no_agent", "full_no_norm", "full_no_spatial",
    "donut_only", "layout_only", "text_only", "vlm_only",
]


def _signature() -> str:
    cfg = get_config()
    parts = [cfg.model.donut_model, cfg.model.llm_model, cfg.groq.vision_model]
    return "__".join(p.replace("/", "-").replace(":", "-") for p in parts)


def _cache_file(dataset: str, doc_id: str) -> Path:
    return BRANCH_CACHE / _signature() / dataset / f"{doc_id}.json"


def vlm_extract(pages: list, schema: Dict) -> Dict[str, Any]:
    """Vision LLM over the pages with the whole schema; first non-null answer per field wins."""
    from extraction.llama_extractor import LlamaVisionExtractor

    extractor = LlamaVisionExtractor()
    result: Dict[str, Any] = {name: None for name in schema["fields"]}
    for page in pages:
        pending = [k for k, v in result.items() if v is None]
        if not pending:
            break
        sub = {**schema, "fields": {k: schema["fields"][k] for k in pending}}
        for name, value in extractor.extract(page, sub).items():
            if value is not None:
                result[name] = value
    return result


def compute_branches(doc, adapter_group: str = "group_2", refresh: bool = False) -> Dict[str, Any]:
    """Run every branch for one document, or load the cached outputs."""
    path = _cache_file(doc.dataset, doc.doc_id)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    from ingestion.ingestion import ingest_pdf
    from main import render_pages

    from extraction.llama_extractor import FAILURES

    cfg = get_config()
    schema = doc.schema
    failures_before = FAILURES["count"]
    out: Dict[str, Any] = {"doc_id": doc.doc_id, "dataset": doc.dataset, "timings": {}, "usage": {}}

    t0 = time.perf_counter()
    blocks, meta = ingest_pdf(str(doc.pdf_path))
    out["timings"]["ingest_s"] = time.perf_counter() - t0
    out["n_pages"] = meta["total_pages"]
    out["ocr_mean_confidence"] = meta.get("ocr_mean_confidence")
    pages = render_pages(doc.pdf_path, cfg.processing.max_pages)
    full_text = " ".join(str(b.get("text", "")) for b in blocks)

    before = usage_snapshot()
    donut = _run_donut(pages[0], schema, cfg.model.donut_model, pages)
    out["usage"]["donut"] = usage_delta(before)

    before = usage_snapshot()
    layout = _run_layoutlm(blocks, pages[0], schema, {}, full_text, adapter_group, pages)
    out["usage"]["layout"] = usage_delta(before)

    before = usage_snapshot()
    t0 = time.perf_counter()
    out["vlm"] = vlm_extract(pages, schema)
    out["timings"]["vlm_s"] = time.perf_counter() - t0
    out["usage"]["vlm"] = usage_delta(before)

    out["donut"] = asdict(donut)
    out["layout"] = asdict(layout)
    out["text_pass"] = (layout.extra or {}).get("text_pass") or {}

    out["vision_failures"] = FAILURES["count"] - failures_before
    out["branch_errors"] = [r.error for r in (donut, layout) if r.error]
    if out["vision_failures"] or out["branch_errors"]:
        # Do not cache a run that hit API or model errors; it would poison later results.
        logger.error("%s: %d vision failure(s), errors=%s -- not cached",
                     doc.doc_id, out["vision_failures"], out["branch_errors"])
        return out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def _result(d: Optional[Dict]) -> Optional[ExtractorResult]:
    return ExtractorResult(**d) if d else None


def _fuse(branches: Dict, schema: Dict, donut: bool, layout: bool, **policy_kwargs) -> Dict[str, Any]:
    bundle = ExtractionBundle(
        donut=_result(branches["donut"]) if donut else None,
        layoutlm=_result(branches["layout"]) if layout else None,
    )
    conf_floor = 1.0 - get_config().processing.confidence_threshold
    return ReflexivePolicyLayer(conf_floor=conf_floor, **policy_kwargs).fuse(bundle, schema).fields


def _repair(fields: Dict, schema: Dict) -> Dict:
    from schema.schema_agent import SchemaAgent

    cfg = get_config()
    agent = SchemaAgent(small_model=cfg.groq.small_model, synthesis_model=cfg.groq.synthesis_model)
    return agent.repair_fields(fields=fields, schema=schema, document_hint=schema.get("form_name", ""))


def apply_system(system: str, branches: Dict, schema: Dict) -> Dict[str, Any]:
    """Predicted field values for ``system`` from cached branch outputs."""
    if system == "vlm_only":
        return dict(branches["vlm"])
    if system == "text_only":
        return dict(branches["text_pass"])
    if system == "donut_only":
        return _fuse(branches, schema, donut=True, layout=False)
    if system == "layout_only":
        return _fuse(branches, schema, donut=False, layout=True)
    if system == "full_no_norm":
        return _fuse(branches, schema, True, True, normalise=False)
    if system == "full_no_spatial":
        return _fuse(branches, schema, True, True, spatial_preferred=set())
    fused = _fuse(branches, schema, True, True)
    if system == "full_no_agent":
        return fused
    if system == "full":
        return _repair(fused, schema)
    raise ValueError(f"Unknown system '{system}'. Choose from {SYSTEMS}")
