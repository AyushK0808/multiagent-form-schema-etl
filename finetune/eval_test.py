"""
finetune/eval_test.py
=====================
Held-out test-split evaluation of trained LayoutLMv3 runs (LoRA adapters and
the frozen-backbone baseline). Writes <run_dir>/test_metrics.json per run.

Test splits are the official `test` splits (DocLayNet: first test shard only,
capped); nothing here was used for training or early stopping.

Usage
-----
python eval_test.py                               # every run found under models/
python eval_test.py --runs models/adapters/group_1/layoutlmv3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from adapter_groups import ADAPTER_GROUPS
from data_loader import load_test_dataset
from label_spaces import LAYOUT_LABELS
from lora_layoutlmv3_trainer import ENCODING_VERSION, _preprocess_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

LABEL2ID = {label: i for i, label in enumerate(LAYOUT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
BASE_MODEL = "microsoft/layoutlmv3-base"


def _encoded_test(group: str, max_test: int, max_length: int, processor):
    """Encode (and cache) the group's test split. Returns (encoded, dataset_names)."""
    from datasets import load_from_disk
    from normalizers import NORMALIZATION_VERSION

    names = next(g.datasets for g in ADAPTER_GROUPS if g.name == group)
    cache = Path("data") / "intermediate" / "finetune_encoded_test" / (
        f"{group}_L{max_length}_t{max_test}_norm{NORMALIZATION_VERSION}_enc{ENCODING_VERSION}"
    )
    if (cache / "encoded").exists() and (cache / "names.json").exists():
        enc = load_from_disk(str(cache / "encoded"))
        return enc, json.loads((cache / "names.json").read_text())

    raw = load_test_dataset(names, max_test_samples=max_test)
    ds_names = list(raw["dataset_name"])
    enc = _preprocess_dataset(raw, processor, max_length, LABEL2ID)
    cache.mkdir(parents=True, exist_ok=True)
    enc.save_to_disk(str(cache / "encoded"))
    (cache / "names.json").write_text(json.dumps(ds_names))
    return enc, ds_names


def _load_model(run_dir: Path, mode: str):
    from transformers import LayoutLMv3ForTokenClassification

    base = LayoutLMv3ForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    if mode == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(run_dir))
    else:
        base.classifier.load_state_dict(torch.load(run_dir / "classifier_head.pt"))
        model = base
    return model.eval().to("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _predict(model, encoded, batch_size: int):
    device = next(model.parameters()).device
    preds, golds = [], []
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start:start + batch_size]
        labels = batch.pop("labels")
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**inputs).logits
        preds.append(logits.argmax(-1).cpu())
        golds.append(labels)
    return torch.cat(preds).numpy(), torch.cat(golds).numpy()


def _scores(pred: np.ndarray, gold: np.ndarray) -> Dict:
    from sklearn.metrics import f1_score

    mask = gold != -100
    p, g = pred[mask], gold[mask]
    out = {
        "macro_f1": float(f1_score(g, p, average="macro", zero_division=0)),
        "accuracy": float((p == g).mean()),
        "tokens": int(g.size),
    }
    present = sorted(int(c) for c in np.unique(g))
    per_class = f1_score(g, p, labels=present, average=None, zero_division=0)
    out["per_class_f1"] = {ID2LABEL[c]: float(s) for c, s in zip(present, per_class)}
    out["per_class_support"] = {ID2LABEL[c]: int((g == c).sum()) for c in present}
    return out


def evaluate_run(run_dir: Path, max_test: int, max_length: int, batch_size: int,
                 test_group: str | None = None) -> Dict:
    from transformers import LayoutLMv3Processor

    cfg = json.loads((run_dir / "run_config.json").read_text())
    mode = cfg.get("train_mode", "lora")
    group = run_dir.parent.name
    test_group = test_group or group   # a different group gives the cross-group evaluation
    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL, apply_ocr=True, use_fast=True)
    encoded, ds_names = _encoded_test(test_group, max_test, max_length, processor)
    encoded.set_format("torch")

    model = _load_model(run_dir, mode)
    pred, gold = _predict(model, encoded, batch_size)
    result = {
        "run": str(run_dir), "group": group, "test_group": test_group, "mode": mode, "seed": cfg["seed"],
        "trainable_params": cfg["trainable_params"], "total_params": cfg["total_params"],
        "test_examples": len(encoded), **_scores(pred, gold),
        "per_dataset": {},
    }
    names = np.array(ds_names)
    for ds in sorted(set(ds_names)):
        idx = names == ds
        result["per_dataset"][ds] = {"examples": int(idx.sum()), **_scores(pred[idx], gold[idx])}
    (run_dir / ("test_metrics.json" if test_group == group else f"test_metrics_on_{test_group}.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("%s  test macro_f1=%.4f acc=%.4f", run_dir, result["macro_f1"], result["accuracy"])
    del model
    torch.cuda.empty_cache()
    return result


def discover_runs(models_root: Path) -> List[Path]:
    return sorted(p.parent for p in models_root.glob("*/group_*/layoutlmv3/run_config.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out test evaluation")
    ap.add_argument("--runs", nargs="+", type=Path, default=None)
    ap.add_argument("--models-root", type=Path, default=Path("models"))
    ap.add_argument("--cross", action="store_true",
                    help="evaluate each run on the other group's test set instead of its own")
    ap.add_argument("--max-test-per-dataset", type=int, default=1000)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    for run in args.runs or discover_runs(args.models_root):
        other = None
        if args.cross:
            other = next(g.name for g in ADAPTER_GROUPS if g.name != run.parent.name)
        evaluate_run(run, args.max_test_per_dataset, args.max_length, args.batch_size, other)


if __name__ == "__main__":
    main()
