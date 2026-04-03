"""
finetune/metrics_logger.py
===========================
Shared per-epoch CSV logging and matplotlib plot generation used by all
four trainers (layoutlmv3, donut, lora_layoutlmv3, lora_donut).

Outputs saved under <output_dir>/
  training_log.csv         — one row per epoch: epoch, train_loss, eval_loss, ...
  plots/
    loss_curve.png          — train_loss + eval_loss over epochs
    primary_metric.png      — macro_f1 (LayoutLMv3) or cer (Donut) over epochs
    learning_rate.png       — LR schedule over epochs
    overview.png            — 2×2 subplot combining all key signals

Usage in a trainer
------------------
    from metrics_logger import EpochCSVLogger, generate_training_plots

    logger_cb = EpochCSVLogger(output_dir / "training_log.csv")
    trainer = Trainer(..., callbacks=[..., logger_cb])
    trainer.train()
    generate_training_plots(output_dir, primary_metric="eval_macro_f1", higher_is_better=True)
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (dark theme matching the Streamlit UI)
# ---------------------------------------------------------------------------

_BG          = "#0d0d0d"
_PANEL       = "#181818"
_ORANGE      = "#ff7a00"
_ORANGE_LT   = "#ffaa55"
_CYAN        = "#4cc9f0"
_GREEN       = "#80ed99"
_RED         = "#ff6b6b"
_MUTED       = "#999999"
_GRID        = "#282828"
_TEXT        = "#eeeeee"
_SUBTEXT     = "#aaaaaa"

# Skip these eval metric suffixes — they're timing noise, not model quality
_SKIP_SUFFIXES = ("_runtime", "_samples_per_second", "_steps_per_second", "_mem_gpu")


# ---------------------------------------------------------------------------
# HuggingFace Trainer callback
# ---------------------------------------------------------------------------

class EpochCSVLogger:
    """
    HuggingFace TrainerCallback that writes one CSV row per evaluation epoch.

    Attaches to the Trainer via the `callbacks` argument.  Works for both
    the standard Trainer and Seq2SeqTrainer.

    The CSV is rewritten on every evaluation so partial runs produce a valid
    (if incomplete) file.

    Columns
    -------
    epoch           integer epoch number
    train_loss      mean training loss over the epoch (last logged step value)
    learning_rate   LR at the last training log in the epoch
    eval_loss       validation loss
    eval_<metric>   all numeric evaluation metrics (F1, CER, accuracy, …)
    """

    # Make it importable as a TrainerCallback without forcing the import at
    # module load time (transformers may not be installed in all environments).
    @staticmethod
    def _base():
        from transformers import TrainerCallback
        return TrainerCallback

    def __init_subclass__(cls, **kwargs):  # noqa
        super().__init_subclass__(**kwargs)

    # We build the actual class lazily so the module can be imported without
    # transformers installed (useful for the UI layer).
    pass


def _make_epoch_csv_logger(csv_path: Path):
    """Return an instantiated EpochCSVLogger TrainerCallback."""
    from transformers import TrainerCallback

    class _EpochCSVLogger(TrainerCallback):
        def __init__(self):
            self._rows: List[Dict[str, Any]] = []
            self._csv_path = Path(csv_path)
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            """
            Fires after each evaluation pass (once per epoch when
            evaluation_strategy='epoch').  Metrics are already computed here.
            """
            if not metrics:
                return

            epoch = round(metrics.get("epoch", state.epoch or 0))
            row: Dict[str, Any] = {"epoch": epoch}

            # Pull most recent train loss + LR from log history
            for entry in reversed(state.log_history):
                if "loss" in entry and "eval_loss" not in entry:
                    row["train_loss"] = round(float(entry["loss"]), 6)
                    if "learning_rate" in entry:
                        row["learning_rate"] = entry["learning_rate"]
                    break

            # Add all numeric eval metrics, skipping timing noise
            for k, v in metrics.items():
                if k == "epoch":
                    continue
                if any(k.endswith(s) for s in _SKIP_SUFFIXES):
                    continue
                if isinstance(v, (int, float)):
                    row[k] = round(float(v), 6)

            # Deduplicate by epoch (overwrite if same epoch appears twice)
            self._rows = [r for r in self._rows if r["epoch"] != epoch]
            self._rows.append(row)
            self._rows.sort(key=lambda r: r["epoch"])
            self._flush()
            logger.info(
                "[MetricsLogger] epoch=%s  %s",
                epoch,
                "  ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)),
            )

        def _flush(self):
            if not self._rows:
                return
            # Build a stable column order
            fieldnames: List[str] = ["epoch"]
            for row in self._rows:
                for k in row:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in self._rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

    return _EpochCSVLogger()


# Public alias used by trainers
def EpochCSVLogger(csv_path: Path):  # noqa: N802  (intentional factory function)
    """Factory: return a ready-to-use TrainerCallback that logs to *csv_path*."""
    return _make_epoch_csv_logger(Path(csv_path))


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_training_plots(
    output_dir: Path,
    primary_metric: str,
    higher_is_better: bool = True,
    model_label: str = "",
) -> None:
    """
    Read <output_dir>/training_log.csv and produce four PNG plots under
    <output_dir>/plots/.

    Parameters
    ----------
    output_dir      : directory that contains training_log.csv
    primary_metric  : column name to use as the headline metric, e.g.
                      "eval_macro_f1"  or  "eval_cer"
    higher_is_better: True for F1 / accuracy / exact_match; False for CER / loss
    model_label     : optional string appended to plot titles
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe in training scripts
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    csv_path = Path(output_dir) / "training_log.csv"
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        logger.warning("[MetricsLogger] training_log.csv not found at %s — skipping plots", csv_path)
        return

    # ── Read CSV ──────────────────────────────────────────────────────────
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if v != "" else None
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)

    if not rows:
        logger.warning("[MetricsLogger] training_log.csv is empty — skipping plots")
        return

    epochs = [r["epoch"] for r in rows]

    def _col(name: str) -> List[Optional[float]]:
        return [r.get(name) for r in rows]

    def _has(name: str) -> bool:
        return any(r.get(name) is not None for r in rows)

    # ── Shared style helper ───────────────────────────────────────────────
    def _style_ax(ax, title: str, xlabel: str = "Epoch", ylabel: str = ""):
        ax.set_facecolor(_PANEL)
        ax.set_title(title, color=_TEXT, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color=_SUBTEXT, fontsize=9)
        ax.set_ylabel(ylabel, color=_SUBTEXT, fontsize=9)
        ax.tick_params(colors=_SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.grid(True, color=_GRID, linestyle="--", linewidth=0.6, alpha=0.8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    def _legend(ax):
        leg = ax.legend(
            facecolor=_BG, edgecolor=_GRID,
            labelcolor=_TEXT, fontsize=8,
        )
        leg.get_frame().set_alpha(0.9)

    title_suffix = f" — {model_label}" if model_label else ""

    # ── 1. Loss curve ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG)
    _style_ax(ax, f"Loss Curve{title_suffix}", ylabel="Loss")
    if _has("train_loss"):
        ax.plot(epochs, _col("train_loss"), color=_ORANGE, linewidth=1.8,
                marker="o", markersize=4, label="Train loss")
    if _has("eval_loss"):
        ax.plot(epochs, _col("eval_loss"), color=_CYAN, linewidth=1.8,
                marker="s", markersize=4, linestyle="--", label="Val loss")
    _legend(ax)
    plt.tight_layout()
    _save(fig, plots_dir / "loss_curve.png")

    # ── 2. Primary metric ─────────────────────────────────────────────────
    if _has(primary_metric):
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG)
        direction = "↑ higher is better" if higher_is_better else "↓ lower is better"
        metric_label = primary_metric.replace("eval_", "").replace("_", " ").title()
        _style_ax(
            ax,
            f"{metric_label} over Epochs{title_suffix}",
            ylabel=f"{metric_label}  ({direction})",
        )
        vals = _col(primary_metric)
        ax.plot(epochs, vals, color=_GREEN, linewidth=2.0,
                marker="D", markersize=5, label=metric_label)
        # Mark best epoch
        non_none = [(e, v) for e, v in zip(epochs, vals) if v is not None]
        if non_none:
            best_e, best_v = (max if higher_is_better else min)(non_none, key=lambda x: x[1])
            ax.axvline(best_e, color=_ORANGE, linewidth=1, linestyle=":", alpha=0.7)
            ax.annotate(
                f"best: {best_v:.4f}",
                xy=(best_e, best_v),
                xytext=(6, -14 if higher_is_better else 6),
                textcoords="offset points",
                color=_ORANGE_LT, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=_ORANGE_LT, lw=0.8),
            )
        _legend(ax)
        plt.tight_layout()
        _save(fig, plots_dir / "primary_metric.png")

    # ── 3. Learning rate schedule ─────────────────────────────────────────
    if _has("learning_rate"):
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=_BG)
        _style_ax(ax, f"Learning Rate Schedule{title_suffix}", ylabel="Learning Rate")
        ax.plot(epochs, _col("learning_rate"), color=_ORANGE_LT,
                linewidth=1.8, marker="o", markersize=3, label="LR")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
        _legend(ax)
        plt.tight_layout()
        _save(fig, plots_dir / "learning_rate.png")

    # ── 4. Overview (2×2 grid) ────────────────────────────────────────────
    n_panels = sum([
        _has("train_loss") or _has("eval_loss"),
        _has(primary_metric),
        _has("learning_rate"),
        _has("eval_accuracy") or _has("eval_exact_match"),
    ])
    ncols = 2
    nrows = max(1, (n_panels + 1) // 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows), facecolor=_BG)
    axes_flat = axes.flatten() if n_panels > 1 else [axes]
    idx = 0

    # Panel A — losses
    if _has("train_loss") or _has("eval_loss"):
        ax = axes_flat[idx]; idx += 1
        _style_ax(ax, "Loss", ylabel="Loss")
        if _has("train_loss"):
            ax.plot(epochs, _col("train_loss"), color=_ORANGE, lw=1.8,
                    marker="o", ms=3, label="Train")
        if _has("eval_loss"):
            ax.plot(epochs, _col("eval_loss"), color=_CYAN, lw=1.8,
                    marker="s", ms=3, ls="--", label="Val")
        _legend(ax)

    # Panel B — primary metric
    if _has(primary_metric):
        ax = axes_flat[idx]; idx += 1
        metric_label = primary_metric.replace("eval_", "").replace("_", " ").title()
        _style_ax(ax, metric_label, ylabel=metric_label)
        ax.plot(epochs, _col(primary_metric), color=_GREEN, lw=1.8,
                marker="D", ms=4, label=metric_label)
        _legend(ax)

    # Panel C — learning rate
    if _has("learning_rate"):
        ax = axes_flat[idx]; idx += 1
        _style_ax(ax, "Learning Rate", ylabel="LR")
        ax.plot(epochs, _col("learning_rate"), color=_ORANGE_LT,
                lw=1.8, marker="o", ms=3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))

    # Panel D — secondary metric (accuracy / exact_match)
    for sec_col, sec_label in [
        ("eval_accuracy",    "Accuracy"),
        ("eval_exact_match", "Exact Match"),
    ]:
        if _has(sec_col):
            ax = axes_flat[idx]; idx += 1
            _style_ax(ax, sec_label, ylabel=sec_label)
            ax.plot(epochs, _col(sec_col), color=_RED, lw=1.8,
                    marker="^", ms=4, label=sec_label)
            _legend(ax)
            break

    # Hide unused panels
    for ax in axes_flat[idx:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Training Overview{title_suffix}",
        color=_TEXT, fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, plots_dir / "overview.png")

    logger.info("[MetricsLogger] Plots saved to %s", plots_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(str(path), dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("[MetricsLogger] Saved %s", path.name)