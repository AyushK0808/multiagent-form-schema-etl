"""
finetune/config.py
==================
Shared constants: label space, dataset specifications, and task prompt.
Must stay in sync with layout_analysis/layout_structure.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Label space — must match layout_analysis/layout_structure.py exactly
# ---------------------------------------------------------------------------

LABEL2ID: Dict[str, int] = {
    "paragraph": 0,
    "heading":   1,
    "list_item": 2,
    "table":     3,
    "caption":   4,
    "other":     5,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

DONUT_TASK_PROMPT = "<s_schema_recognition>"


# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    name:             str
    repo_id:          str
    schema_name:      Optional[str]
    description:      str
    annotation_type:  str
    curriculum_order: int
    config_name:      Optional[str] = None   # ← add this
    trust_remote_code: bool = False           # ← add this


DATASET_SPECS: Dict[str, DatasetSpec] = {
    # ── Originally included ──────────────────────────────────────────────
    "FUNSD": DatasetSpec(
        name="FUNSD", repo_id="nielsr/funsd",
        schema_name="funsd_form", description="Forms",
        annotation_type="word", curriculum_order=3,
    ),
    "CORD": DatasetSpec(
        name="CORD", repo_id="naver-clova-ix/cord-v2",
        schema_name="cord_receipt", description="Receipts",
        annotation_type="segment", curriculum_order=1,
    ),
    "SROIE": DatasetSpec(
        name="SROIE", repo_id="rth/sroie-2019-v2",
        schema_name="sroie_invoice", description="Invoices",
        annotation_type="segment", curriculum_order=2,
    ),
    "DOCVQA": DatasetSpec(
    name="DOCVQA", repo_id="lmms-lab/DocVQA",
    schema_name="docvqa_reasoning", description="Document reasoning",
    annotation_type="qa", curriculum_order=4,
    config_name="DocVQA",          # ← fix: was missing config name
),
"RVL-CDIP": DatasetSpec(
    name="RVL-CDIP", repo_id="hf-tuner/rvl-cdip-document-classification",
    schema_name=None, description="Document diversity",
    annotation_type="doc", curriculum_order=3,
    trust_remote_code=True,        # ← fix: uses legacy dataset script
),
    # ── Critical adds ────────────────────────────────────────────────────
    "PUBLAYNET": DatasetSpec(
        name="PUBLAYNET", repo_id="jordanparker6/publaynet",
        schema_name="publaynet_layout", description="Academic paper layout",
        annotation_type="segment", curriculum_order=2,
    ),
    "DOCLAYNET": DatasetSpec(
    name="DOCLAYNET", repo_id="docling-project/DocLayNet-v1.2",
    schema_name="doclaynet_layout", description="Multi-domain document layout",
    annotation_type="segment", curriculum_order=2,
    trust_remote_code=True,        # ← fix: uses legacy dataset script
),
    "KLEISTER_NDA": DatasetSpec(
        name="KLEISTER_NDA", repo_id="lhoestq/kleister-nda",
        schema_name="kleister_nda", description="NDA clause extraction",
        annotation_type="word", curriculum_order=5,
    ),
    # ── Additional adds ──────────────────────────────────────────────────
    "DOCBANK": DatasetSpec(
        name="DOCBANK", repo_id="maveriq/DocBank",
        schema_name="docbank_layout", description="Academic paper token classification",
        annotation_type="word", curriculum_order=3,
    ),
    # "DEEPFORM": DatasetSpec(
    #     name="DEEPFORM", repo_id="jvamvas/deepform",
    #     schema_name="deepform_form", description="SEC filing form fields",
    #     annotation_type="word", curriculum_order=3,
    # ),
    # "XFUND": DatasetSpec(
    #     name="XFUND", repo_id="naver-clova-ix/xfund",
    #     schema_name="xfund_multilingual_form", description="Multilingual forms",
    #     annotation_type="word", curriculum_order=4,
    # ),
    "INFOGRAPHICVQA": DatasetSpec(
        name="INFOGRAPHICVQA", repo_id="Ryoo72/InfographicsVQA",
        schema_name="infographic_vqa", description="Infographic visual reasoning",
        annotation_type="qa", curriculum_order=5,
    ),
    "SYNTHDOG_EN": DatasetSpec(
        name="SYNTHDOG_EN", repo_id="naver-clova-ix/synthdog-en",
        schema_name="synthdog_ocr", description="Synthetic OCR-free reading",
        annotation_type="qa", curriculum_order=1,
    ),
}