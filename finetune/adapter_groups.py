"""
finetune/adapter_groups.py
==========================
Central definition of the three LoRA adapter groups.

Every other module that needs to know which datasets belong to which group,
or which adapter to load for a given schema/document type, imports from here.

Group layout
------------
GROUP_1  (curriculum orders 1-2)  layout-primitive / receipt-like
GROUP_2  (curriculum orders 3)    structural classification
GROUP_3  (curriculum orders 4-5)  reasoning-heavy / clause extraction

The SCHEMA_TO_GROUP mapping is used at inference time by AdapterRouter to
translate a recognised schema name into the correct adapter directory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Group definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdapterGroupSpec:
    name:             str          # e.g. "group_1"
    label:            str          # human-readable description
    datasets:         List[str]    # DATASET_SPECS keys
    curriculum_orders: List[int]   # orders covered

    # Relative sub-path under the adapter output root
    @property
    def subdir(self) -> str:
        return self.name


ADAPTER_GROUPS: List[AdapterGroupSpec] = [
    AdapterGroupSpec(
        name="group_1",
        label="Layout-primitive / receipt-like",
        datasets=["CORD", "SROIE", "SYNTHDOG_EN"],
        curriculum_orders=[1, 2],
    ),
    AdapterGroupSpec(
        name="group_2",
        label="Structural classification",
        datasets=["FUNSD", "RVL-CDIP", "DOCLAYNET", "DOCBANK"],
        curriculum_orders=[3],
    ),
    AdapterGroupSpec(
        name="group_3",
        label="Reasoning-heavy / clause extraction",
        datasets=["DOCVQA", "KLEISTER_NDA", "INFOGRAPHICVQA"],
        curriculum_orders=[4, 5],
    ),
]

# Quick lookup: dataset name → group
DATASET_TO_GROUP: Dict[str, AdapterGroupSpec] = {
    ds: grp
    for grp in ADAPTER_GROUPS
    for ds in grp.datasets
}

# ---------------------------------------------------------------------------
# Schema-name → group mapping (used by AdapterRouter at inference)
# These are the label_text values produced by the normalizers.
# ---------------------------------------------------------------------------

SCHEMA_TO_GROUP: Dict[str, str] = {
    # Group 1
    "cord_receipt":     "group_1",
    "sroie_invoice":    "group_1",
    "synthdog_ocr":     "group_1",

    # Group 2
    "funsd_form":           "group_2",
    "rvl_cdip":             "group_2",
    "doclaynet_layout":     "group_2",
    "docbank_layout":       "group_2",
    # RVL-CDIP exposes many sub-labels; catch them all
    "advertisement":        "group_2",
    "budget":               "group_2",
    "email":                "group_2",
    "file_folder":          "group_2",
    "form":                 "group_2",
    "handwritten":          "group_2",
    "invoice":              "group_2",
    "letter":               "group_2",
    "memo":                 "group_2",
    "news_article":         "group_2",
    "presentation":         "group_2",
    "questionnaire":        "group_2",
    "resume":               "group_2",
    "scientific_publication": "group_2",
    "scientific_report":    "group_2",
    "specification":        "group_2",

    # Group 3
    "docvqa_reasoning":     "group_3",
    "kleister_nda":         "group_3",
    "infographic_vqa":      "group_3",
}

# Fallback group when schema is unknown
DEFAULT_GROUP = "group_2"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def group_for_schema(schema_name: str) -> str:
    """Return the group name for a recognised schema label."""
    return SCHEMA_TO_GROUP.get(schema_name, DEFAULT_GROUP)


def group_spec(group_name: str) -> Optional[AdapterGroupSpec]:
    return next((g for g in ADAPTER_GROUPS if g.name == group_name), None)


def adapter_path(output_root: Path, group_name: str) -> Path:
    """Canonical path for a group's LoRA adapter checkpoint."""
    return output_root / group_name