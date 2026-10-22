"""
eval/time_branches.py
=====================
Per-stage wall-clock latency of the pipeline, measured on real (uncached) model calls.

The benchmark's branch cache stores only OCR and VLM time, and replayed LLM answers
make those near zero, so latency is measured here instead. Run with the LLM cache
bypassed (LLM_CACHE=0) or the model stages will report cache-hit times.

Stages: ingest (OCR), donut, layout (LayoutLM + text LLM + vision fallback), vlm
(vision LLM alone), fuse (deterministic policy, no model call).
One warm-up document is run first and discarded so model loading is not counted.

Example
-------
LLM_CACHE=0 python -m eval.time_branches --per-dataset 5
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from config.config import get_config
from eval.data import load_benchmark
from eval.systems import _fuse, _release_gpu_cache, vlm_extract
from extraction.parallel_extractor import _run_donut, _run_layoutlm
from utils.llm_cache import cache_enabled

STAGES = ["ingest", "donut", "layout", "vlm", "fuse"]


def _time_doc(doc, adapter_group: str) -> dict:
    from ingestion.ingestion import ingest_pdf
    from main import render_pages

    cfg = get_config()
    t: dict = {}
    t0 = time.perf_counter()
    blocks, _ = ingest_pdf(str(doc.pdf_path))
    t["ingest"] = time.perf_counter() - t0
    pages = render_pages(doc.pdf_path, cfg.processing.max_pages)
    full_text = " ".join(str(b.get("text", "")) for b in blocks)

    t0 = time.perf_counter()
    donut = _run_donut(pages[0], doc.schema, cfg.model.donut_model, pages)
    t["donut"] = time.perf_counter() - t0

    _release_gpu_cache()
    t0 = time.perf_counter()
    layout = _run_layoutlm(blocks, pages[0], doc.schema, {}, full_text, adapter_group, pages)
    t["layout"] = time.perf_counter() - t0

    _release_gpu_cache()
    t0 = time.perf_counter()
    vlm_extract(pages, doc.schema)
    t["vlm"] = time.perf_counter() - t0

    from dataclasses import asdict
    branches = {"donut": asdict(donut), "layout": asdict(layout),
                "ocr_blocks": [{"text": str(b.get("text", "")), "confidence": float(b.get("confidence", 1.0))}
                               for b in blocks]}
    t0 = time.perf_counter()
    _fuse(branches, doc.schema, True, True)
    t["fuse"] = time.perf_counter() - t0

    t["pipeline"] = t["ingest"] + t["donut"] + t["layout"] + t["fuse"]   # full system without the VLM branch
    t["pipeline_plus_vlm"] = t["pipeline"] + t["vlm"]
    t["n_pages"] = len(pages)
    return t


def _p90(values: list) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["sroie", "cord", "synth"])
    ap.add_argument("--per-dataset", type=int, default=5)
    ap.add_argument("--subset-seed", type=int, default=1, help="different from the benchmark's seed 0")
    ap.add_argument("--adapter-group", default="group_2")
    ap.add_argument("--out", type=Path, default=Path("results_local") / "timing.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("extraction.policy_layer").setLevel(logging.ERROR)
    if cache_enabled():
        raise SystemExit("LLM cache is on; re-run with LLM_CACHE=0 or the model stages will be cache hits.")

    cfg = get_config()
    vision = cfg.ollama_vision_model if cfg.vision_backend == "ollama" else cfg.groq.vision_model
    print(f"vision backend: {cfg.vision_backend} ({vision}); text LLM: {cfg.model.llm_model}")

    docs = []
    for name in args.datasets:
        subset = load_benchmark(name, args.per_dataset, args.subset_seed, "test")[: args.per_dataset]
        docs.extend(subset)
    print(f"{len(docs)} documents ({', '.join(args.datasets)})")

    print(f"warm-up on {docs[0].doc_id} (discarded)")
    _time_doc(docs[0], args.adapter_group)

    rows = []
    for i, doc in enumerate(docs, 1):
        t = _time_doc(doc, args.adapter_group)
        rows.append({"doc_id": doc.doc_id, "dataset": doc.dataset, **t})
        print(f"[{i}/{len(docs)}] {doc.doc_id:<14} " + "  ".join(f"{s} {t[s]:6.2f}s" for s in STAGES)
              + f"  | pipeline {t['pipeline']:6.2f}s  vlm {t['vlm']:6.2f}s")

    cols = STAGES + ["pipeline", "pipeline_plus_vlm"]
    summary = {}
    print(f"\n{'group':<8} {'n':>3} " + " ".join(f"{c:>18}" for c in cols) + "   (mean / median / p90, seconds)")
    for group in [*args.datasets, "all"]:
        sel = [r for r in rows if group == "all" or r["dataset"] == group]
        if not sel:
            continue
        summary[group] = {c: {"mean": statistics.mean(r[c] for r in sel),
                              "median": statistics.median(r[c] for r in sel),
                              "p90": _p90([r[c] for r in sel])} for c in cols}
        cells = " ".join(
            f"{summary[group][c]['mean']:>5.1f}/{summary[group][c]['median']:>5.1f}/{summary[group][c]['p90']:>5.1f}"
            for c in cols)
        print(f"{group:<8} {len(sel):>3} {cells}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"vision_model": vision, "vision_backend": cfg.vision_backend,
                                    "rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
