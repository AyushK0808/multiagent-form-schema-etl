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
full_no_grounding  fusion without dropping Donut values absent from the OCR text
donut_only         Donut branch alone, answers absent from the OCR text dropped
donut_only_ungrounded  Donut branch alone, no grounding (what Donut produces when OCR is sparse)
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
    _grounding_block,
    _run_donut,
    _run_layoutlm,
)
from extraction.policy_layer import ReflexivePolicyLayer
from utils.llm_cache import usage_delta, usage_snapshot

logger = logging.getLogger(__name__)

BRANCH_CACHE = Path("data") / "cache" / "branches"
SYSTEMS = [
    "full", "full_no_agent", "full_no_norm", "full_no_spatial", "full_no_grounding",
    "full_donut_fill_all", "full_no_donut_fill",
    "donut_only", "donut_only_ungrounded", "layout_only", "text_only", "vlm_only",
]


# Bump when Donut decoding changes; cached entries then recompute only the Donut branch.
DONUT_VERSION = 2


def _signature() -> str:
    cfg = get_config()
    vision = cfg.ollama_vision_model if cfg.vision_backend == "ollama" else cfg.groq.vision_model
    parts = [cfg.model.donut_model, cfg.model.llm_model, vision]
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


def _release_gpu_cache() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_branches(doc, adapter_group: str = "group_2", refresh: bool = False) -> Dict[str, Any]:
    """Run every branch for one document, or load the cached outputs."""
    path = _cache_file(doc.dataset, doc.doc_id)
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (cached.get("donut_version"), cached.get("donut_model")) != (DONUT_VERSION, get_config().model.donut_model):
            # Donut decoding changed; recompute only that branch and keep the cached layout and VLM outputs.
            from main import render_pages
            pages = render_pages(doc.pdf_path, get_config().processing.max_pages)
            before = usage_snapshot()
            cached["donut"] = asdict(_run_donut(pages[0], doc.schema, get_config().model.donut_model, pages))
            cached["usage"]["donut"] = usage_delta(before)
            cached["donut_version"] = DONUT_VERSION
            cached["donut_model"] = get_config().model.donut_model
            path.write_text(json.dumps(cached, ensure_ascii=False, default=str), encoding="utf-8")
        if "ocr_blocks" not in cached:
            # Backfill for entries cached before grounding existed; local OCR only, no LLM calls.
            from ingestion.ingestion import ingest_pdf
            cached["ocr_blocks"] = _slim_blocks(ingest_pdf(str(doc.pdf_path))[0])
            path.write_text(json.dumps(cached, ensure_ascii=False, default=str), encoding="utf-8")
        return cached

    from ingestion.ingestion import ingest_pdf
    from main import render_pages

    from extraction.llama_extractor import DEGENERATE, FAILURES

    cfg = get_config()
    schema = doc.schema
    failures_before = FAILURES["count"]
    degenerate_before = DEGENERATE["count"]
    out: Dict[str, Any] = {"doc_id": doc.doc_id, "dataset": doc.dataset, "timings": {}, "usage": {}}

    t0 = time.perf_counter()
    blocks, meta = ingest_pdf(str(doc.pdf_path))
    out["timings"]["ingest_s"] = time.perf_counter() - t0
    out["ocr_blocks"] = _slim_blocks(blocks)
    out["n_pages"] = meta["total_pages"]
    out["ocr_mean_confidence"] = meta.get("ocr_mean_confidence")
    pages = render_pages(doc.pdf_path, cfg.processing.max_pages)
    full_text = " ".join(str(b.get("text", "")) for b in blocks)

    before = usage_snapshot()
    donut = _run_donut(pages[0], schema, cfg.model.donut_model, pages)
    out["usage"]["donut"] = usage_delta(before)

    _release_gpu_cache()
    before = usage_snapshot()
    layout = _run_layoutlm(blocks, pages[0], schema, {}, full_text, adapter_group, pages)
    out["usage"]["layout"] = usage_delta(before)

    _release_gpu_cache()   # a local VLM server shares this GPU; give it back PyTorch's cached memory
    before = usage_snapshot()
    t0 = time.perf_counter()
    out["vlm"] = vlm_extract(pages, schema)
    out["timings"]["vlm_s"] = time.perf_counter() - t0
    out["usage"]["vlm"] = usage_delta(before)

    out["donut"] = asdict(donut)
    out["donut_version"] = DONUT_VERSION
    out["donut_model"] = cfg.model.donut_model
    out["layout"] = asdict(layout)
    out["text_pass"] = (layout.extra or {}).get("text_pass") or {}

    out["vision_failures"] = FAILURES["count"] - failures_before
    out["vision_degenerate"] = DEGENERATE["count"] - degenerate_before   # model loops: scored as empty
    out["branch_errors"] = [r.error for r in (donut, layout) if r.error]
    if out["vision_failures"] or out["branch_errors"] or out["vision_degenerate"]:
        # Do not cache a run that hit API or model errors; it would poison later results.
        logger.error("%s: %d vision failure(s), errors=%s -- not cached",
                     doc.doc_id, out["vision_failures"], out["branch_errors"])
        return out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def _slim_blocks(blocks: list) -> List[Dict[str, Any]]:
    return [{"text": str(b.get("text", "")), "confidence": float(b.get("confidence", 1.0))} for b in blocks]


def _result(d: Optional[Dict]) -> Optional[ExtractorResult]:
    return ExtractorResult(**d) if d else None


def _plausible(value: Any, field_type: str) -> bool:
    """Sanity check on a Donut answer: no tokenizer markers, not a text dump, parses as its type."""
    if not isinstance(value, str):
        return True
    from extraction.donut_extractor import _MAX_ANSWER_CHARS, _TOKENIZER_MARKER
    return len(value) <= _MAX_ANSWER_CHARS and not _TOKENIZER_MARKER.search(value)


def _ground_donut(donut: Optional[Dict], blocks: List[Dict], schema: Optional[Dict] = None) -> Optional[Dict]:
    """Donut's own confidences are uncalibrated (0.8-1.0 whether right or wrong). A value
    that appears in no OCR block is unsupported by the document text and is dropped; a
    supported one takes the confidence of the block it was found in."""
    if not donut:
        return donut
    fields, confs = dict(donut["fields"]), dict(donut["confidences"])
    for name, value in fields.items():
        if value is None:
            continue
        ftype = ((schema or {}).get("fields", {}).get(name) or {}).get("type", "string")
        if not _plausible(value, ftype):
            fields[name], confs[name] = None, 0.0
            continue
        block = _grounding_block(value, blocks)
        if block is None:
            fields[name], confs[name] = None, 0.0
        else:
            confs[name] = min(confs.get(name, 1.0), float(block.get("confidence", 1.0)))
    return {**donut, "fields": fields, "confidences": confs}


def _fuse(branches: Dict, schema: Dict, donut: bool, layout: bool, ground: bool = True,
          **policy_kwargs) -> Dict[str, Any]:
    donut_branch = branches["donut"]
    if ground:
        donut_branch = _ground_donut(donut_branch, branches.get("ocr_blocks", []), schema)
    bundle = ExtractionBundle(
        donut=_result(donut_branch) if donut else None,
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
    if system == "donut_only_ungrounded":
        return _fuse(branches, schema, donut=True, layout=False, ground=False)
    if system == "layout_only":
        return _fuse(branches, schema, donut=False, layout=True)
    if system == "full_no_norm":
        return _fuse(branches, schema, True, True, normalise=False)
    if system == "full_donut_fill_all":
        return _repair(_fuse(branches, schema, True, True, donut_fill="all"), schema)
    if system == "full_no_donut_fill":
        return _repair(_fuse(branches, schema, True, True, donut_fill="none"), schema)
    if system == "full_no_grounding":
        return _fuse(branches, schema, True, True, ground=False)
    if system == "full_no_spatial":
        return _fuse(branches, schema, True, True, spatial_preferred=set())
    fused = _fuse(branches, schema, True, True)
    if system == "full_no_agent":
        return fused
    if system == "full":
        return _repair(fused, schema)
    raise ValueError(f"Unknown system '{system}'. Choose from {SYSTEMS}")
