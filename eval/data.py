"""
eval/data.py
============
Benchmark loaders. Every document is exposed as a PDF on disk (page images are
wrapped in a one-page PDF) so evaluation runs go through the same ingestion
path as production input, plus its gold field values and the schema to fill.

Datasets
--------
sroie   rth/sroie-2019-v2, test split: company, date, address, total
cord    naver-clova-ix/cord-v2, test split: subtotal, tax, total, cash, change
synth   regenerated stress set (see eval/make_synthetic.py)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz

PDF_CACHE = Path("data") / "cache" / "eval_pdfs"


@dataclass
class EvalDoc:
    doc_id: str
    dataset: str
    pdf_path: Path
    gold: Dict[str, Any]
    schema: Dict[str, Any]

    @property
    def field_types(self) -> Dict[str, str]:
        return {k: v.get("type", "string") for k, v in self.schema["fields"].items()}


def _field(ftype: str, description: str, required: bool = False) -> Dict[str, Any]:
    return {"type": ftype, "description": description, "required": required}


SROIE_SCHEMA = {
    "form_name": "sroie_receipt",
    "fields": {
        "company": _field("string", "Name of the company or store that issued the receipt", True),
        "date": _field("date", "Date printed on the receipt", True),
        "address": _field("string", "Street address of the company printed on the receipt", True),
        "total": _field("number", "Total amount paid, as printed on the receipt", True),
    },
}

CORD_SCHEMA = {
    "form_name": "cord_receipt",
    "fields": {
        "subtotal_price": _field("number", "Subtotal of the items before tax and service charges"),
        "tax_price": _field("number", "Tax amount charged"),
        "total_price": _field("number", "Total amount to pay", True),
        "cashprice": _field("number", "Cash amount tendered by the customer"),
        "changeprice": _field("number", "Change returned to the customer"),
        "creditcardprice": _field("number", "Amount paid by credit card"),
    },
}

_CORD_PATHS = {
    "subtotal_price": ("sub_total", "subtotal_price"),
    "tax_price": ("sub_total", "tax_price"),
    "total_price": ("total", "total_price"),
    "cashprice": ("total", "cashprice"),
    "changeprice": ("total", "changeprice"),
    "creditcardprice": ("total", "creditcardprice"),
}


def _image_to_pdf(image, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import io
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    w, h = image.size
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(str(out_path))
    doc.close()
    return out_path


def _scalar(value: Any) -> Optional[str]:
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def _select(indices: List[int], limit: Optional[int], seed: int) -> List[int]:
    if limit is None or limit >= len(indices):
        return indices
    return sorted(random.Random(seed).sample(indices, limit))


def _split_rows(repo: str, split: str):
    """(dataset, row indices). ``validation`` is the tuning split: CORD's own, and for
    SROIE the seeded 10% of train that fine-tuning also held out. Never score tuning on test."""
    from datasets import load_dataset

    if split == "test":
        ds = load_dataset(repo, split="test")
        return ds, list(range(len(ds)))
    if split != "validation":
        raise ValueError(f"split must be 'test' or 'validation', got {split!r}")
    if repo == "naver-clova-ix/cord-v2":
        ds = load_dataset(repo, split="validation")
        return ds, list(range(len(ds)))
    ds = load_dataset(repo, split="train")
    val = ds.train_test_split(test_size=0.1, seed=42)["test"]
    return val, list(range(len(val)))


def load_sroie(limit: Optional[int] = None, seed: int = 0, split: str = "test") -> List[EvalDoc]:
    ds, rows = _split_rows("rth/sroie-2019-v2", split)
    docs = []
    for i in _select(rows, limit, seed):
        ex = ds[i]
        ent = ex["objects"]["entities"]
        gold = {k: _scalar(ent.get(k)) for k in SROIE_SCHEMA["fields"]}
        tag = "" if split == "test" else "val-"
        pdf = _image_to_pdf(ex["image"], PDF_CACHE / "sroie" / f"{tag}{i:04d}.pdf")
        docs.append(EvalDoc(f"sroie-{tag}{i:04d}", "sroie", pdf, gold, SROIE_SCHEMA))
    return docs


def load_cord(limit: Optional[int] = None, seed: int = 0, split: str = "test") -> List[EvalDoc]:
    ds, rows = _split_rows("naver-clova-ix/cord-v2", split)
    docs = []
    for i in _select(rows, limit, seed):
        ex = ds[i]
        parse = json.loads(ex["ground_truth"])["gt_parse"]
        gold = {}
        for name, (group, key) in _CORD_PATHS.items():
            node = parse.get(group) or {}
            gold[name] = _scalar(node.get(key)) if isinstance(node, dict) else None
        tag = "" if split == "test" else "val-"
        pdf = _image_to_pdf(ex["image"], PDF_CACHE / "cord" / f"{tag}{i:04d}.pdf")
        docs.append(EvalDoc(f"cord-{tag}{i:04d}", "cord", pdf, gold, CORD_SCHEMA))
    return docs


def load_synth(limit: Optional[int] = None, seed: int = 0, split: str = "test", root: Path = Path("data") / "eval" / "synth") -> List[EvalDoc]:
    docs = []
    for gold_file in sorted(root.glob("*.json")):
        meta = json.loads(gold_file.read_text(encoding="utf-8"))
        pdf = root / meta["pdf"]
        docs.append(EvalDoc(gold_file.stem, "synth", pdf, meta["gold"], meta["schema"]))
    if limit:
        keep = set(_select(list(range(len(docs))), limit, seed))
        docs = [d for i, d in enumerate(docs) if i in keep]
    return docs


LOADERS = {"sroie": load_sroie, "cord": load_cord, "synth": load_synth}


def load_benchmark(name: str, limit: Optional[int] = None, seed: int = 0, split: str = "test") -> List[EvalDoc]:
    if name not in LOADERS:
        raise ValueError(f"Unknown benchmark '{name}'. Choose from {sorted(LOADERS)}")
    return LOADERS[name](limit=limit, seed=seed, split=split)
