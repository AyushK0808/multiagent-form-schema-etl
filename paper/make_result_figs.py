"""Result figures drawn from the benchmark JSONL files, timing.json and finetune_results.json.

python paper/make_result_figs.py -> paper/fig_perfield.pdf, fig_ecdf.pdf, fig_ablation.pdf,
                                    fig_latency.pdf, fig_halluc.pdf, fig_perclass.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.evaluator import ExtractionMetrics, canonicalize, token_f1  # noqa: E402

OUT = ROOT / "paper"
RES = ROOT / "results_local"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
SYS = {"vlm_only": ("VLM only", "#d1495b"), "layout_only": ("OCR-LLM branch", "#1f6f8b"),
       "full": ("Full pipeline", "#2f2f2f")}
DATA = {"sroie": "SROIE (100)", "cord": "CORD (50)"}


def rows(ds, system):
    return [json.loads(l) for l in (RES / ds / f"{system}.jsonl").open(encoding="utf-8")]


def field_f1(r, field):
    ft = (r.get("field_types") or {}).get(field, "string")
    p, g = canonicalize(r["pred"].get(field), ft), canonicalize(r["gold"].get(field), ft)
    if p is not None and not str(p).strip():
        p = None
    if p is None and g is None:
        return None
    return token_f1(p, g) if p is not None and g is not None else 0.0


def short(f):
    return f.replace("_price", "").replace("price", "").replace("creditcard", "credit\ncard").replace("_", "\n")


def pooled(ds, system):
    """Corpus-level metrics, pooled over fields as in the paper's tables."""
    m = ExtractionMetrics()
    for r in rows(ds, system):
        m.update(r["pred"], r["gold"], r.get("field_types"))
    return m.get_metrics()


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def perfield():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [4, 6]})
    for ax, ds in zip(axes, DATA):
        fields = list(rows(ds, "full")[0]["gold"])
        w = 0.8 / len(SYS)
        for j, (s, (lab, col)) in enumerate(SYS.items()):
            rs = rows(ds, s)
            ys = [np.mean([v for r in rs if (v := field_f1(r, f)) is not None]) for f in fields]
            ax.bar(np.arange(len(fields)) + (j - 1) * w, ys, w, color=col, label=lab)
        ax.set_xticks(range(len(fields)))
        ax.set_xticklabels([short(f) for f in fields], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(DATA[ds], fontsize=8)
    axes[0].set_ylabel("Token-overlap F1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig_perfield.pdf")


def ecdf():
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0), sharey=True)
    for ax, ds in zip(axes, DATA):
        for s, (lab, col) in SYS.items():
            f = np.sort([r["metrics"]["f1"] for r in rows(ds, s)])
            ax.step(f, np.arange(1, len(f) + 1) / len(f), where="post", color=col, lw=1.3, label=lab)
        ax.set_title(DATA[ds].split()[0], fontsize=8)
        ax.set_xlabel("Per-document F1")
    axes[0].set_ylabel(r"Share of documents $\leq$ F1")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_ecdf.pdf")


ABL = [("full", "Full pipeline"), ("full_no_grounding", "no grounding"), ("full_no_agent", "no schema repair"),
       ("full_no_norm", "no normalization"), ("full_no_spatial", "no spatial rule"),
       ("full_donut_fill_all", "Donut fills all"), ("full_no_donut_fill", "Donut fills none")]


def ablation():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    base = {ds: pooled(ds, "full")["f1"] for ds in DATA}
    labels = [l for _, l in ABL[1:]]
    y = np.arange(len(labels))
    for j, (ds, col) in enumerate(zip(DATA, ["#1f6f8b", "#e3a008"])):
        d = [pooled(ds, s)["f1"] - base[ds] for s, _ in ABL[1:]]
        ax.barh(y + (j - 0.5) * 0.38, d, 0.38, color=col, label=DATA[ds].split()[0])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ F1 vs. full pipeline")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_ablation.pdf")


def latency():
    t = json.loads((RES / "timing.json").read_text())["summary"]
    stages = [("ingest", "Ingestion", "#9ca3af"), ("donut", "Donut", "#e3a008"),
              ("layout", "OCR-LLM branch", "#1f6f8b")]
    dss = [d for d in ("sroie", "cord", "synth") if d in t]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    y = np.arange(len(dss))
    left = np.zeros(len(dss))
    for k, lab, col in stages:
        v = np.array([t[d][k]["median"] for d in dss])
        ax.barh(y - 0.19, v, 0.36, left=left, color=col, label=lab)
        left += v
    # CORD standalone-VLM rows were answered from Ollama's prompt cache, so they are left out
    vlm = [np.nan if d == "cord" else t[d]["vlm"]["median"] for d in dss]
    ax.barh(y + 0.19, vlm, 0.36, color="#d1495b", label="Standalone VLM")
    ax.set_yticks(y)
    ax.set_yticklabels([{"sroie": "SROIE", "cord": "CORD", "synth": "Synthetic"}[d] for d in dss])
    ax.invert_yaxis()
    ax.set_xlabel("Median seconds per document")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2)
    fig.tight_layout()
    save(fig, "fig_latency.pdf")


def halluc():
    """CORD: predictions made where the gold value is empty, per field."""
    fields = list(rows("cord", "full")[0]["gold"])
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    w = 0.8 / len(SYS)
    for j, (s, (lab, col)) in enumerate(SYS.items()):
        cnt = dict.fromkeys(fields, 0)
        for r in rows("cord", s):
            ft = r.get("field_types") or {}
            for f in fields:
                p = canonicalize(r["pred"].get(f), ft.get(f, "string"))
                g = canonicalize(r["gold"].get(f), ft.get(f, "string"))
                cnt[f] += p is not None and bool(str(p).strip()) and g is None
        ax.bar(np.arange(len(fields)) + (j - 1) * w, [cnt[f] for f in fields], w, color=col, label=lab)
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels([short(f) for f in fields], fontsize=7)
    ax.set_ylabel("Hallucinated values (of 50)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_halluc.pdf")


def perclass():
    res = json.loads((OUT / "finetune_results.json").read_text())
    classes = ["paragraph", "heading", "table", "caption", "list_item", "other"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for j, (g, col) in enumerate([("group_1", "#1f6f8b"), ("group_2", "#d1495b")]):
        pc = res[f"lora/{g}"]["test"]["per_class_f1"]
        ax.bar(np.arange(len(classes)) + (j - 0.5) * 0.38, [pc.get(c, [0, 0])[0] for c in classes], 0.38,
               yerr=[pc.get(c, [0, 0])[1] for c in classes], capsize=1.5, color=col, label=g.replace("_", " ").title())
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=7)
    ax.set_ylabel(r"Test F1 (mean $\pm$ std)")
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    fig.tight_layout()
    save(fig, "fig_perclass.pdf")


if __name__ == "__main__":
    for f in (perfield, ecdf, ablation, latency, halluc, perclass):
        f()
    base = {ds: pooled(ds, "full")["f1"] for ds in DATA}
    print("full F1 (paper: sroie .801, cord .836):", {k: round(v, 3) for k, v in base.items()})
