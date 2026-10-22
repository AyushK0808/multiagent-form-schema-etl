"""
finetune/aggregate_results.py
=============================
Aggregates per-seed run_config.json / metrics.json / test_metrics.json into
mean ± std (sample std, ddof=1) and emits JSON plus LaTeX table rows.

Runs are grouped by (train_mode, group); the seeds are the runs in a cell.

Usage: python aggregate_results.py [--models-root models] [--out paper/finetune_results]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _ms(values):
    a = np.asarray(values, dtype=float)
    return [float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0]


def collect(models_root: Path):
    cells = defaultdict(list)
    for cfg_path in sorted(models_root.glob("*/group_*/layoutlmv3/run_config.json")):
        run = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())
        val = json.loads((run / "metrics.json").read_text())
        test_path = run / "test_metrics.json"
        test = json.loads(test_path.read_text()) if test_path.exists() else None
        if test is not None:
            other = list(run.glob("test_metrics_on_*.json"))
            test["cross"] = json.loads(other[0].read_text()) if other else None
        cells[(cfg.get("train_mode", "lora"), cfg["group"])].append((cfg, val, test))
    return cells


def aggregate(cells):
    out = {}
    for (mode, group), runs in sorted(cells.items()):
        first = runs[0][0]
        cell = {
            "n_seeds": len(runs),
            "seeds": [c["seed"] for c, _, _ in runs],
            "trainable_params": first["trainable_params"],
            "total_params": first["total_params"],
            "trainable_pct": 100 * first["trainable_params"] / first["total_params"],
            "adapter_mb": float(np.mean([c["adapter_bytes"] for c, _, _ in runs])) / 2**20,
            "val": {k: _ms([v[f"eval_{k}"] for _, v, _ in runs]) for k in ("macro_f1", "accuracy")},
            "test": {},
        }
        tests = [t for _, _, t in runs if t]
        if tests:
            cell["test"] = {k: _ms([t[k] for t in tests]) for k in ("macro_f1", "accuracy")}
            classes = sorted({c for t in tests for c in t["per_class_f1"]})
            cell["test"]["per_class_f1"] = {
                c: _ms([t["per_class_f1"][c] for t in tests if c in t["per_class_f1"]])
                for c in classes
            }
            cross = [t["cross"] for t in tests if t.get("cross")]
            if cross:
                cell["test"]["cross_group"] = {
                    "test_group": cross[0]["test_group"],
                    "macro_f1": _ms([c["macro_f1"] for c in cross]),
                    "accuracy": _ms([c["accuracy"] for c in cross]),
                }
            cell["test"]["n_seeds"] = len(tests)
            cell["test"]["examples"] = tests[0]["test_examples"]
        out[f"{mode}/{group}"] = cell
    return out


def _fmt(ms, n=2):
    if n < 2:
        return f"{ms[0]:.3f}"
    return f"{ms[0]:.3f} $\\pm$ {ms[1]:.3f}"


def latex_rows(agg) -> str:
    rows = []
    for key, c in agg.items():
        mode, group = key.split("/")
        name = "LoRA" if mode == "lora" else "Frozen backbone"
        test = (_fmt(c["test"]["macro_f1"]), _fmt(c["test"]["accuracy"])) if c["test"] else ("--", "--")
        rows.append(
            f"{name} & {group.replace('_', ' ').title()} & {c['trainable_pct']:.2f}\\% & "
            f"{_fmt(c['val']['macro_f1'])} & {test[0]} & {test[1]} \\\\"
        )
    return "\n".join(rows)


def lora_table(agg) -> str:
    """Tabular for the LoRA vs frozen-backbone comparison (validation and test)."""
    lines = [
        r"\begin{tabular}{llrcccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Group} & \textbf{Trainable} & \textbf{Seeds} & "
        r"\textbf{Val F1} & \textbf{Test F1} & \textbf{Test acc.} \\",
        r"\midrule",
    ]
    for key in sorted(agg, key=lambda k: (k.split("/")[1], k.split("/")[0] != "lora")):
        mode, group = key.split("/")
        c, t = agg[key], agg[key]["test"]
        name = "LoRA" if mode == "lora" else "Frozen + head"
        params = f"{c['trainable_pct']:.2g}\\%"
        lines.append(
            f"{name} & {group.replace('_', ' ').title()} & {params} & {c['n_seeds']} & "
            f"{_fmt(c['val']['macro_f1'], c['n_seeds'])} & "
            f"{_fmt(t['macro_f1'], c['n_seeds']) if t else '--'} & {_fmt(t['accuracy'], c['n_seeds']) if t else '--'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def cross_table(agg) -> str:
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"\textbf{Adapter} & \textbf{Own-group test F1} & \textbf{Other-group test F1} \\",
             r"\midrule"]
    for group in ("group_1", "group_2"):
        c = agg.get(f"lora/{group}")
        if not c or not c["test"]:
            continue
        cg = c["test"].get("cross_group")
        lines.append(f"{group.replace('_', ' ').title()} & {_fmt(c['test']['macro_f1'])} & "
                     f"{_fmt(cg['macro_f1']) if cg else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, default=Path("paper") / "finetune_results")
    args = ap.parse_args()

    agg = aggregate(collect(args.models_root))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    args.out.with_suffix(".tex").write_text(latex_rows(agg) + "\n", encoding="utf-8")
    (args.out.parent / "tab_lora.tex").write_text(lora_table(agg), encoding="utf-8")
    (args.out.parent / "tab_crossgroup.tex").write_text(cross_table(agg), encoding="utf-8")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
