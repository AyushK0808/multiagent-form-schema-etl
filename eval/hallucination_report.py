"""
eval/hallucination_report.py
============================
Break the hallucination rate down by field, from the JSONL files a benchmark run wrote.

A "hallucination" is a non-null prediction where the gold value is null. On CORD most of
them can be a field the receipt does not print (e.g. a subtotal line) that the model fills
with a value read from elsewhere, so this report also counts how often the hallucinated
value equals another gold field of the same document (a copied value, not an invented one).

Offline: reads files only, no model calls.

Example
-------
python -m eval.hallucination_report --dir results_local/cord --systems full vlm_only layout_only
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluation.evaluator import canonicalize


def _report(path: Path) -> None:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    n_pred, hallu = Counter(), Counter()
    n_gold, missed = Counter(), Counter()
    copied = Counter()
    for r in rows:
        types = r.get("field_types") or {}
        gold = {k: canonicalize(v, types.get(k, "string")) for k, v in r["gold"].items()}
        for field in set(r["pred"]) | set(gold):
            ftype = types.get(field, "string")
            pred = canonicalize(r["pred"].get(field), ftype)
            if pred is not None and not str(pred).strip():
                pred = None
            truth = gold.get(field)
            n_pred[field] += pred is not None
            n_gold[field] += truth is not None
            if pred is not None and truth is None:
                hallu[field] += 1
                others = {v for k, v in gold.items() if k != field and v is not None}
                copied[field] += pred in others
            elif pred is None and truth is not None:
                missed[field] += 1

    total_pred, total_h = sum(n_pred.values()), sum(hallu.values())
    print(f"\n{path.stem}: {total_h}/{total_pred} predictions hallucinated "
          f"({100 * total_h / max(total_pred, 1):.1f}%), {sum(copied.values())} of them copy another gold field")
    print(f"  {'field':<18} {'gold':>5} {'pred':>5} {'halluc':>7} {'copied':>7} {'missed':>7}")
    for field in sorted(set(n_pred) | set(n_gold), key=lambda f: -hallu[f]):
        print(f"  {field:<18} {n_gold[field]:>5} {n_pred[field]:>5} {hallu[field]:>7} {copied[field]:>7} {missed[field]:>7}")
    rest_pred = total_pred - n_pred.get("subtotal_price", 0)
    rest_h = total_h - hallu.get("subtotal_price", 0)
    if "subtotal_price" in n_pred and rest_pred:
        print(f"  excluding subtotal_price: {rest_h}/{rest_pred} ({100 * rest_h / rest_pred:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="results directory, e.g. results_local/cord")
    ap.add_argument("--systems", nargs="+", default=["full", "layout_only", "vlm_only"])
    args = ap.parse_args()
    for system in args.systems:
        _report(args.dir / f"{system}.jsonl")


if __name__ == "__main__":
    main()
