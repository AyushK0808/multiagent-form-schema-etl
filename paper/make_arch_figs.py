"""Draw the architecture and parallel-extraction figures (vector PDF, large labels).

python paper/make_arch_figs.py   ->  paper/architecture.pdf, paper/parallel_ex.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent
INK, FILL, ACCENT, GREY = "#1f2937", "#eef2f7", "#dbeafe", "#9ca3af"


def box(ax, cx, cy, w, h, text, fill=FILL, dashed=False, size=9):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=GREY if dashed else INK, lw=1.0, ls="--" if dashed else "-"))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.15)


def arrow(ax, p, q, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.0,
                                 color=GREY if dashed else INK, ls="--" if dashed else "-"))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def architecture() -> None:
    """One-column, top-to-bottom version of the eight-stage pipeline."""
    fig, ax = canvas(3.75, 7.6)
    cx, w, s = 1.45, 2.4, 7.5
    side = 3.25
    box(ax, cx, 7.35, 1.4, 0.35, "PDF document", size=s)
    box(ax, cx, 6.75, w, 0.5, "Ingestion\nnative text or Tesseract OCR,\nword confidence", size=s)
    box(ax, cx, 6.1, w, 0.45, "1  Adapter routing\nschema name → adapter group", fill=ACCENT, size=s)
    box(ax, side, 6.1, 0.75, 0.45, "LoRA layout\nadapters", dashed=True, size=s - 1)
    arrow(ax, (cx, 7.175), (cx, 7.0))
    arrow(ax, (cx, 6.5), (cx, 6.325))
    arrow(ax, (side - 0.375, 6.1), (cx + w / 2, 6.1), dashed=True)
    # parallel panel
    ax.add_patch(FancyBboxPatch((0.1, 3.9), 2.7, 1.8, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="white", ec=INK, lw=1.0, ls="-"))
    ax.text(cx, 5.53, "2  Parallel extraction", ha="center", va="center", fontsize=s + 0.5, color=INK, weight="bold")
    box(ax, 0.78, 5.0, 1.25, 0.55, "Donut\nOCR-free, one\nquestion per field", size=s - 0.5)
    box(ax, 2.12, 5.0, 1.25, 0.55, "OCR-LLM\ntext LLM over\nOCR text", size=s - 0.5)
    box(ax, 0.78, 4.35, 1.25, 0.45, "Ground answers\nin OCR text", fill=ACCENT, size=s - 0.5)
    box(ax, 2.12, 4.35, 1.25, 0.45, "VLM fallback\nfor empty fields", fill=ACCENT, size=s - 0.5)
    arrow(ax, (0.78, 4.725), (0.78, 4.575))
    arrow(ax, (2.12, 4.725), (2.12, 4.575))
    ax.text(cx, 4.0, "branches run in parallel", ha="center", va="center", fontsize=6.5, color=GREY)
    arrow(ax, (cx, 5.875), (cx, 5.72))
    # sequential stages
    names = ["3  Policy fusion: evidence rules", "4  Schema repair: LLM agent", "5  Schema resolve: registry match",
             "6  Validation: type and range checks", "7  Persist: SQLite", "8  Finalize: fields, scores, issue log"]
    ys = [3.5 - 0.55 * i for i in range(6)]
    arrow(ax, (cx, 3.88), (cx, ys[0] + 0.18))
    for i, (y, t) in enumerate(zip(ys, names)):
        box(ax, cx, y, w, 0.36, t, fill=ACCENT if i == 0 else FILL, size=s - 0.5)
        if i:
            arrow(ax, (cx, ys[i - 1] - 0.18), (cx, y + 0.18))
    box(ax, side, ys[2], 0.75, 0.36, "Schema\nregistry", size=s - 1)
    arrow(ax, (side - 0.375, ys[2]), (cx + w / 2, ys[2]))
    ax.text(cx, 0.3, "Every stage records its errors in the shared state;\nlater stages continue on the partial state.",
            fontsize=6.5, color=GREY, ha="center", va="center")
    fig.savefig(OUT / "architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def line(ax, pts, dashed=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=GREY if dashed else INK, lw=1.0, ls="--" if dashed else "-", solid_capstyle="butt")


def architecture_wide() -> None:
    """Two-column figure: ingestion, the eight orchestrator stages, and the parallel branches in detail."""
    fig, ax = canvas(7.1, 4.6)
    s = 7.5
    # top row
    box(ax, 0.6, 3.3, 0.9, 0.5, "PDF\ndocument", size=s)
    box(ax, 1.85, 3.3, 1.4, 0.8, "Ingestion\nnative text or\nTesseract OCR,\nword confidence", size=s)
    box(ax, 3.4, 3.3, 1.3, 0.8, "1  Adapter routing\nschema name ->\nadapter group", fill=ACCENT, size=s)
    box(ax, 3.4, 4.25, 1.3, 0.45, "LoRA layout\nadapters", dashed=True, size=s)
    arrow(ax, (1.05, 3.3), (1.15, 3.3))
    arrow(ax, (2.55, 3.3), (2.75, 3.3))
    arrow(ax, (3.4, 4.02), (3.4, 3.7), dashed=True)
    # parallel panel
    ax.add_patch(FancyBboxPatch((4.4, 2.1), 2.65, 2.1, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="white", ec=INK, lw=1.0, ls="-"))
    ax.text(5.725, 4.05, "2  Parallel extraction", ha="center", va="center", fontsize=s + 0.5, color=INK, weight="bold")
    box(ax, 5.05, 3.4, 1.15, 0.55, "Donut\nOCR-free, one\nquestion per field", size=s - 0.5)
    box(ax, 6.4, 3.4, 1.15, 0.55, "OCR-LLM\ntext LLM over\nOCR text", size=s - 0.5)
    box(ax, 5.05, 2.65, 1.15, 0.45, "Ground answers\nin OCR text", fill=ACCENT, size=s - 0.5)
    box(ax, 6.4, 2.65, 1.15, 0.45, "VLM fallback\nfor empty fields", fill=ACCENT, size=s - 0.5)
    arrow(ax, (5.05, 3.12), (5.05, 2.88))
    arrow(ax, (6.4, 3.12), (6.4, 2.88))
    ax.text(5.725, 2.25, "branches run in parallel", ha="center", fontsize=6.5, color=GREY)
    arrow(ax, (4.05, 3.3), (4.4, 3.3))
    # bottom row
    names = ["3  Policy\nfusion\nevidence rules", "4  Schema\nrepair\nLLM agent", "5  Schema\nresolve\nregistry match",
             "6  Validation\ntype and range\nchecks", "7  Persist\nSQLite", "8  Finalize\nfields, scores,\nissue log"]
    xs = [0.6 + 1.16 * i for i in range(6)]
    for i, (x, t) in enumerate(zip(xs, names)):
        box(ax, x, 1.0, 1.0, 0.8, t, fill=ACCENT if i == 0 else FILL, size=6.3)
        if i:
            arrow(ax, (xs[i - 1] + 0.5, 1.0), (x - 0.5, 1.0))
    line(ax, [(5.725, 2.1), (5.725, 1.8), (0.6, 1.8)])
    arrow(ax, (0.6, 1.8), (0.6, 1.4))
    box(ax, 2.92, 0.2, 1.3, 0.32, "Schema registry", size=s - 0.5)
    arrow(ax, (2.92, 0.36), (2.92, 0.6))
    ax.text(7.0, 0.2, "Every stage records its errors in the shared state;\nlater stages continue on the partial state.",
            fontsize=6.5, color=GREY, ha="right", va="center")
    fig.savefig(OUT / "architecture_wide.pdf", bbox_inches="tight")
    plt.close(fig)


def parallel() -> None:
    fig, ax = canvas(3.6, 3.7)
    box(ax, 1.8, 3.4, 2.4, 0.4, "Page images + OCR blocks")
    box(ax, 0.9, 2.5, 1.5, 0.55, "Donut\nanswers + token\nconfidences", size=8.5)
    box(ax, 2.7, 2.5, 1.5, 0.55, "OCR-LLM\ntext pass, then\nVLM fallback", size=8.5)
    box(ax, 0.9, 1.7, 1.5, 0.5, "Drop answers not\nfound in OCR text", fill=ACCENT, size=8.5)
    box(ax, 1.8, 0.85, 3.2, 0.7,
        "Fusion per field: agree -> OCR-LLM value;\nconflict -> OCR-LLM unless its value is\nunusable and Donut's is usable", fill=ACCENT, size=8.5)
    box(ax, 1.8, 0.15, 2.4, 0.3, "Fused fields + issue log", size=8.5)
    for p, q in [((1.2, 3.2), (0.9, 2.78)), ((2.4, 3.2), (2.7, 2.78)), ((0.9, 2.22), (0.9, 1.95)),
                 ((0.9, 1.45), (1.3, 1.2)), ((2.7, 2.22), (2.3, 1.2)), ((1.8, 0.5), (1.8, 0.3))]:
        arrow(ax, p, q)
    ax.text(3.45, 3.1, "run in parallel", fontsize=7.5, color=GREY, ha="right")
    fig.savefig(OUT / "parallel_ex.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    architecture_wide()
    parallel()
    print("wrote architecture.pdf, architecture_wide.pdf, parallel_ex.pdf")
