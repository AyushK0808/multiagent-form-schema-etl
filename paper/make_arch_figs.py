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
    parallel()
    print("wrote architecture.pdf, parallel_ex.pdf")
