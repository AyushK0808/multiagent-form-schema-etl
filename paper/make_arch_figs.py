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
    fig, ax = canvas(3.6, 5.0)
    cx = 1.7
    box(ax, 1.3, 4.65, 1.9, 0.42, "PDF document")
    box(ax, cx, 3.95, 2.2, 0.5, "Ingestion\nnative text / Tesseract OCR")
    box(ax, 0.8, 2.95, 1.45, 0.7, "Donut\n(OCR-free, one\nquestion per field)")
    box(ax, 2.6, 2.95, 1.45, 0.7, "OCR-LLM branch\ntext LLM, then VLM\nfor empty fields")
    box(ax, 0.8, 2.15, 1.45, 0.42, "Ground in\nOCR text", fill=ACCENT)
    box(ax, cx, 1.45, 2.4, 0.5, "Evidence-based fusion\n(per-field source and reason)", fill=ACCENT)
    box(ax, cx, 0.7, 2.4, 0.5, "Schema repair, validation\nSQLite")
    box(ax, 3.0, 4.65, 1.1, 0.5, "Schema\nregistry", size=8)
    for p, q in [((1.3, 4.44), (1.3, 4.2)), ((1.2, 3.7), (0.9, 3.3)), ((2.2, 3.7), (2.5, 3.3)),
                 ((0.8, 2.6), (0.8, 2.36)), ((0.9, 1.94), (1.3, 1.7)), ((2.5, 2.6), (2.1, 1.7)),
                 ((cx, 1.2), (cx, 0.95))]:
        arrow(ax, p, q)
    arrow(ax, (2.6, 4.4), (2.6, 4.2), dashed=True)
    box(ax, 3.0, 2.15, 0.95, 0.5, "LoRA layout\nadapters*", dashed=True, size=7)
    ax.text(1.8, 0.2, "*evaluated on its own; not used by the extraction branches", fontsize=7, color=GREY, ha="center")
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
