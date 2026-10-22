"""
eval/run_benchmark.py
=====================
Evaluate systems on a benchmark and write one JSONL row per (document, system).

Branch outputs are cached on disk, so interrupted runs resume and extra
systems or ablations are derived without new model calls.

Examples
--------
python -m eval.run_benchmark --dataset cord --limit 20
python -m eval.run_benchmark --dataset sroie --systems full vlm_only donut_only
python -m eval.run_benchmark --dataset sroie --branches-only      # fill the cache only
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from eval.data import load_benchmark
from eval.systems import SYSTEMS, apply_system, compute_branches
from evaluation.evaluator import ExtractionMetrics
from utils.llm_cache import cache_stats

logger = logging.getLogger("eval")


def _progress(i: int, n: int, doc_id: str, start: float, t0: float, f1, system: str = "") -> str:
    """One progress line: position, last-doc time, elapsed, ETA from the mean time per document."""
    elapsed = time.time() - start
    eta = elapsed / i * (n - i)
    f1_txt = f"  running {system} F1={f1:.3f}" if f1 is not None else ""
    return (f"[{i}/{n} {100 * i / n:3.0f}%] {doc_id}  {time.time() - t0:5.1f}s  "
            f"elapsed {elapsed / 60:.1f}m  ETA {eta / 60:.1f}m{f1_txt}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["sroie", "cord", "synth"])
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--split", default="test", choices=["test", "validation"],
                    help="validation is for tuning the policy; report numbers on test only")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate a seeded random subset")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--adapter-group", default="group_2")
    ap.add_argument("--branches-only", action="store_true", help="Compute and cache branch outputs only")
    ap.add_argument("--refresh", action="store_true", help="Ignore cached branch outputs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)-7s %(message)s")

    # The policy logs every disagreement at WARNING, once per derived system; the per-field
    # audit trail is kept in the returned issues, so keep the progress output readable.
    logging.getLogger("extraction.policy_layer").setLevel(logging.ERROR)

    docs = load_benchmark(args.dataset, args.limit, args.subset_seed, args.split)
    print(f"{args.dataset}: {len(docs)} documents")

    out_dir = args.out / (args.dataset if args.split == "test" else f"{args.dataset}_val")
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {s: (out_dir / f"{s}.jsonl").open("w", encoding="utf-8") for s in args.systems}
    totals = {s: ExtractionMetrics() for s in args.systems}

    start = time.time()
    failed_docs = []   # documents whose vision or branch calls failed; their scores are not valid
    try:
        for i, doc in enumerate(docs, 1):
            t0 = time.time()
            branches = compute_branches(doc, args.adapter_group, refresh=args.refresh)
            if branches.get("vision_failures") or branches.get("branch_errors"):
                failed_docs.append(doc.doc_id)
                print(f"    WARNING {doc.doc_id}: {branches.get('vision_failures', 0)} vision failure(s), "
                      f"errors={branches.get('branch_errors')} -- scored as empty, not cached")
            if args.branches_only:
                print(_progress(i, len(docs), doc.doc_id, start, t0, None))
                continue
            for system in args.systems:
                pred = apply_system(system, branches, doc.schema)
                per_doc = ExtractionMetrics().compute_field_level(pred, doc.gold, doc.field_types)
                totals[system].update(pred, doc.gold, doc.field_types)
                handles[system].write(json.dumps({
                    "doc_id": doc.doc_id, "dataset": doc.dataset, "system": system,
                    "pred": pred, "gold": doc.gold, "field_types": doc.field_types,
                    "metrics": per_doc, "n_pages": branches.get("n_pages"),
                }, ensure_ascii=False, default=str) + "\n")
            lead = "full" if "full" in totals else args.systems[0]
            print(_progress(i, len(docs), doc.doc_id, start, t0, totals[lead].get_metrics()["f1"], lead))
            if i % 10 == 0 or i == len(docs):
                print(f"    cache={cache_stats()}")
    finally:
        for h in handles.values():
            h.close()

    if failed_docs:
        print(f"\n*** {len(failed_docs)}/{len(docs)} documents had failed model calls; the scores below "
              f"count them as empty and are NOT valid. Fix the cause and re-run (failed docs are not cached). ***")
    if args.branches_only:
        return
    print(f"\n{'system':<18} {'F1':>6} {'P':>6} {'R':>6} {'EM':>6} {'fill':>6} {'halluc':>7}")
    for system in args.systems:
        m = totals[system].get_metrics()
        print(f"{system:<18} {m['f1']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['exact_accuracy']:>6.3f} {m['fill_rate']:>6.3f} {m['hallucination_rate']:>7.3f}")


if __name__ == "__main__":
    main()
