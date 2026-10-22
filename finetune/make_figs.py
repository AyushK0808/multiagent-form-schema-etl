"""
finetune/make_figs.py
=====================
Vector figures for the LoRA section, drawn from the per-run training_log.csv files
and paper/finetune_results.json (see aggregate_results.py).

  paper/training_curves.pdf   validation macro-F1 per epoch, every seed, both groups
  paper/training_metrics.pdf  validation vs test macro-F1, LoRA vs frozen backbone

Usage: python finetune/make_figs.py [--models-root models] [--out-dir paper]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUP_COLOR = {"group_1": "#1f6f8b", "group_2": "#d1495b"}


def _epochs(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [int(r["epoch"]) for r in rows], [float(r["eval_macro_f1"]) for r in rows]


def training_curves(models_root: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for group, color in GROUP_COLOR.items():
        for i, log in enumerate(sorted(models_root.glob(f"adapters*/{group}/layoutlmv3/training_log.csv"))):
            ep, f1 = _epochs(log)
            ax.plot(ep, f1, color=color, alpha=0.75, lw=1.1,
                    label=group.replace("_", " ").title() if i == 0 else None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0.2, 0.9)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def metrics_bars(results: Path, out: Path) -> None:
    agg = json.loads(results.read_text())
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    cells = [("lora/group_1", "G1\nLoRA"), ("frozen/group_1", "G1\nfrozen"),
             ("lora/group_2", "G2\nLoRA"), ("frozen/group_2", "G2\nfrozen")]
    cells = [(k, label) for k, label in cells if k in agg]
    width = 0.38
    for j, (split, hatch) in enumerate((("val", ""), ("test", "//"))):
        xs, ys, es = [], [], []
        for i, (key, _) in enumerate(cells):
            data = agg[key][split]
            if not data:
                continue
            xs.append(i + (j - 0.5) * width)
            ys.append(data["macro_f1"][0])
            es.append(data["macro_f1"][1])
        ax.bar(xs, ys, width, yerr=es, capsize=2, hatch=hatch, color="#1f6f8b" if j == 0 else "#8fb8c9",
               edgecolor="white", label="validation" if j == 0 else "test")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([label for _, label in cells], fontsize=8)
    ax.set_ylabel("Macro-F1 (mean $\\pm$ std)")
    ax.set_ylim(0.5, 0.9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _seed_logs(models_root: Path, kind: str, group: str) -> list[list[dict]]:
    """Per-seed training logs (list of row dicts) for `kind` in {adapters, baseline_frozen}."""
    logs = []
    for path in sorted(models_root.glob(f"{kind}*/{group}/layoutlmv3/training_log.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            logs.append(list(csv.DictReader(f)))
    return logs


def _mean_by_epoch(logs: list[list[dict]], key: str):
    """Mean of `key` over seeds at each epoch, using the seeds that reached that epoch."""
    by_epoch: dict[int, list[float]] = {}
    for rows in logs:
        for r in rows:
            if r.get(key) not in (None, ""):
                by_epoch.setdefault(int(r["epoch"]), []).append(float(r[key]))
    ep = sorted(by_epoch)
    return ep, [sum(by_epoch[e]) / len(by_epoch[e]) for e in ep]


def loss_curves(models_root: Path, out: Path) -> None:
    """Train and validation loss per epoch, LoRA vs frozen head, seed means."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.2), sharey=True)
    for ax, group in zip(axes, GROUP_COLOR):
        for kind, color, name in (("adapters", GROUP_COLOR[group], "LoRA"), ("baseline_frozen", "#888888", "frozen")):
            logs = _seed_logs(models_root, kind, group)
            for key, ls in (("train_loss", "-"), ("eval_loss", "--")):
                ep, v = _mean_by_epoch(logs, key)
                if ep:
                    ax.plot(ep, v, color=color, ls=ls, lw=1.2, label=f"{name} {key.split('_')[0]}")
        ax.set_title(group.replace("_", " ").title(), fontsize=8)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Loss (seed mean)")
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def class_curves(models_root: Path, out: Path) -> None:
    """Validation F1 of each layout class per epoch for the LoRA adapters, seed means."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.2), sharey=True)
    for ax, group in zip(axes, GROUP_COLOR):
        logs = _seed_logs(models_root, "adapters", group)
        keys = sorted({k for rows in logs for k in rows[0] if k.startswith("eval_f1_")})
        for key in keys:
            ep, v = _mean_by_epoch(logs, key)
            ax.plot(ep, v, lw=1.1, label=key[len("eval_f1_"):].replace("_", " "))
        ax.set_title(group.replace("_", " ").title(), fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    axes[0].set_ylabel("Validation F1 per class")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def class_gain(results: Path, out: Path) -> None:
    """Test F1 gain of LoRA over the frozen head per class."""
    agg = json.loads(results.read_text())
    classes = ["paragraph", "heading", "table", "caption", "list_item", "other"]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for j, (group, color) in enumerate(GROUP_COLOR.items()):
        lora, frozen = agg[f"lora/{group}"]["test"]["per_class_f1"], agg[f"frozen/{group}"]["test"]["per_class_f1"]
        xs = [i + (j - 0.5) * 0.38 for i, c in enumerate(classes) if c in lora and c in frozen]
        ys = [lora[c][0] - frozen[c][0] for c in classes if c in lora and c in frozen]
        ax.bar(xs, ys, 0.38, color=color, label=group.replace("_", " ").title())
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(classes)))
    short = {"paragraph": "para-\ngraph", "list_item": "list\nitem"}
    ax.set_xticklabels([short.get(c, c) for c in classes], fontsize=7)
    ax.set_ylabel("Test F1 gain over frozen head")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def cross_group(results: Path, out: Path) -> None:
    """Macro-F1 matrix: adapter trained on a group (rows) evaluated on each group's test set (columns)."""
    agg = json.loads(results.read_text())
    m = [[0.0] * 2 for _ in range(2)]
    for i, g in enumerate(GROUP_COLOR):
        m[i][i] = agg[f"lora/{g}"]["test"]["macro_f1"][0]
        cg = agg[f"lora/{g}"]["test"]["cross_group"]
        m[i][1 - i] = cg["macro_f1"][0]
    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    im = ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i][j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if m[i][j] > 0.5 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Group 1\ntest", "Group 2\ntest"], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Group 1\nadapter", "Group 2\nadapter"], fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Test macro-F1")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def schema_pr(results: Path, out: Path) -> None:
    res = json.loads(results.read_text())
    sig = [r["sigma"] for r in res["curve"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(sig, [r["precision"] for r in res["curve"]], color="#1f6f8b", marker="o", ms=3, label="precision")
    ax.plot(sig, [r["recall"] for r in res["curve"]], color="#d1495b", marker="s", ms=3, label="recall")
    ax.plot(sig, [r["false_accept_rate"] for r in res["curve"]], color="#888888", ls="--", label="false accept")
    ax.axvline(res["operating_sigma"], color="black", lw=0.8, ls=":")
    ax.set_xlabel(r"Similarity threshold $\sigma$")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", type=Path, default=Path("models"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper"))
    args = ap.parse_args()
    training_curves(args.models_root, args.out_dir / "training_curves.pdf")
    metrics_bars(args.out_dir / "finetune_results.json", args.out_dir / "training_metrics.pdf")
    loss_curves(args.models_root, args.out_dir / "fig_ft_loss.pdf")
    class_curves(args.models_root, args.out_dir / "fig_ft_classcurves.pdf")
    class_gain(args.out_dir / "finetune_results.json", args.out_dir / "fig_ft_gain.pdf")
    cross_group(args.out_dir / "finetune_results.json", args.out_dir / "fig_ft_cross.pdf")
    schema_json = Path("results_final") / "schema_matching.json"
    if schema_json.exists():
        schema_pr(schema_json, args.out_dir / "schema_registry.pdf")


if __name__ == "__main__":
    main()
