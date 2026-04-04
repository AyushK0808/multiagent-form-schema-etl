from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

import fitz
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import get_config
from schema.schema import SchemaManager
from schema.schema_recognizer import SchemaRecognizer

logger = logging.getLogger(__name__)


UPLOAD_DIR = ROOT_DIR / "data" / "raw" / "ui_uploads"
FINETUNE_DIR = ROOT_DIR / "finetune"
SCHEMA_TRAINING_ROOT = FINETUNE_DIR / "models" / "schema_recognition"
LORA_TRAINING_ROOT = FINETUNE_DIR / "models" / "adapters"

# ---------------------------------------------------------------------------
# Dataset metadata  (descriptions + approximate sizes/sample counts)
# ---------------------------------------------------------------------------

DATASET_INFO: Dict[str, Dict] = {
    "CORD": {
        "description": "Consolidated Receipt Dataset — receipt key information extraction (vendor, total, tax, items).",
        "size": "~46 MB",
        "train_samples": "800",
        "val_samples": "100",
        "gated": False,
    },
    "SROIE": {
        "description": "Scanned Receipts OCR and Key Information Extraction from ICDAR 2019 challenge.",
        "size": "~300 MB",
        "train_samples": "626",
        "val_samples": "347",
        "gated": False,
    },
    "SYNTHDOG_EN": {
        "description": "Synthetic document dataset for OCR pretraining — diverse English document images.",
        "size": "~2.5 GB",
        "train_samples": "100 k",
        "val_samples": "5 k",
        "gated": False,
    },
    "FUNSD": {
        "description": "Form Understanding in Noisy Scanned Documents — semantic entity labeling for forms.",
        "size": "~10 MB",
        "train_samples": "149",
        "val_samples": "50",
        "gated": False,
    },
    "RVL-CDIP": {
        "description": "Document image classification across 16 categories: invoices, letters, memos, and more.",
        "size": "~38 GB",
        "train_samples": "320 k",
        "val_samples": "40 k",
        "gated": False,
    },
    "DOCLAYNET": {
        "description": "Multi-domain document layout annotation from IBM — financial, legal, scientific, government docs.",
        "size": "~30 GB",
        "train_samples": "69 k",
        "val_samples": "6.5 k",
        "gated": False,
    },
    "DOCBANK": {
        "description": "Academic paper token-level layout classification built from arXiv PDFs.",
        "size": "~12 GB",
        "train_samples": "400 k",
        "val_samples": "50 k",
        "gated": False,
    },
    "DOCVQA": {
        "description": "Document Visual Question Answering — reading comprehension on real document images. Requires HF token.",
        "size": "~20 GB",
        "train_samples": "39 k",
        "val_samples": "5 k",
        "gated": True,
    },
    "KLEISTER_NDA": {
        "description": "NDA clause extraction — key obligations, parties, and dates from non-disclosure agreements.",
        "size": "~50 MB",
        "train_samples": "254",
        "val_samples": "83",
        "gated": False,
    },
    "INFOGRAPHICVQA": {
        "description": "VQA on infographic images requiring chart, graph, and table understanding. Requires HF token.",
        "size": "~4 GB",
        "train_samples": "23 k",
        "val_samples": "2.8 k",
        "gated": True,
    },
}


def get_dataset_info(dataset_name: str) -> Dict:
    return DATASET_INFO.get(dataset_name, {
        "description": "No description available.",
        "size": "Unknown",
        "train_samples": "?",
        "val_samples": "?",
        "gated": False,
    })


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def ensure_ui_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file) -> Path:
    ensure_ui_dirs()
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    target = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    target.write_bytes(uploaded_file.getbuffer())
    logger.info("[UI] Saved upload '%s' to %s", uploaded_file.name, target)
    return target


def load_preview_image(file_path: Path) -> Image.Image:
    suffix = file_path.suffix.lower()
    logger.info("[UI] Loading preview image for %s", file_path)
    if suffix == ".pdf":
        with fitz.open(str(file_path)) as doc:
            page = doc.load_page(0)
            pix = page.get_pixmap()
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return Image.open(file_path).convert("RGB")


def run_document_pipeline(
    file_path: Path,
    form_name: Optional[str] = None,
    schema_id: Optional[str] = None,
) -> Dict[str, Any]:
    logger.info(
        "[UI] Running ETL pipeline for %s with form_name=%s schema_id=%s",
        file_path,
        form_name,
        schema_id,
    )
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Pipeline execution currently supports PDF inputs only.")

    from main import run_pipeline

    cfg = get_config()
    output_path = cfg.paths.output_dir / f"extracted_{file_path.stem}.json"
    output = run_pipeline(
        pdf_path=file_path,
        form_name=form_name,
        schema_id=schema_id,
        output_path=output_path,
    )
    logger.info("[UI] Pipeline completed for %s; output saved to %s", file_path, output_path)
    return {
        "output": output,
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def build_schema_template(form_name: str = "NewSchema") -> Dict[str, Any]:
    return {
        "form_name": form_name,
        "version": "1.0",
        "description": "",
        "fields": {
            "field_1": {
                "type": "string",
                "description": "Describe the field",
                "required": True,
                "examples": [],
            }
        },
    }


def recognize_schema_from_image(image: Image.Image) -> Dict[str, Any]:
    logger.info("[UI] Running schema recognition on preview image")
    cfg = get_config()
    recognizer = SchemaRecognizer(
        layout_model_path=cfg.model.schema_recognition_layout_model,
        donut_model_path=cfg.model.schema_recognition_donut_model,
        layout_fallback_model=cfg.model.schema_recognition_layout_fallback_model,
        donut_fallback_model=cfg.model.schema_recognition_donut_fallback_model,
    )
    prediction = recognizer.predict(image)
    logger.info(
        "[UI] Schema recognition predicted %s (confidence=%.3f source=%s)",
        prediction["schema_name"],
        prediction["confidence"],
        prediction["source"],
    )

    manager = SchemaManager()
    existing = manager.get_schema(form_name=prediction["schema_name"])
    schema_payload = existing or build_schema_template(prediction["schema_name"])
    # Ensure no None examples cause serialisation issues
    schema_payload = _sanitise_schema(schema_payload)
    return {
        "prediction": prediction,
        "schema_payload": schema_payload,
        "schema_found": existing is not None,
    }


def _sanitise_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values in examples lists and ensure valid structure."""
    schema = dict(schema)
    fields = schema.get("fields", {})
    sanitised_fields = {}
    for fname, fmeta in fields.items():
        if not isinstance(fmeta, dict):
            fmeta = {}
        examples = [e for e in fmeta.get("examples", []) if e is not None]
        sanitised_fields[fname] = {
            "type": fmeta.get("type", "string"),
            "description": fmeta.get("description", ""),
            "required": bool(fmeta.get("required", False)),
            "examples": examples,
        }
    schema["fields"] = sanitised_fields
    return schema


def list_schemas() -> List[Dict[str, Any]]:
    schemas = SchemaManager().list_schemas()
    logger.info("[UI] Loaded %d schema entries from store", len(schemas))
    return schemas


def load_schema_by_name(form_name: str) -> Optional[Dict[str, Any]]:
    schema = SchemaManager().get_schema(form_name=form_name)
    logger.info("[UI] Loading schema by form_name=%s found=%s", form_name, bool(schema))
    return _sanitise_schema(schema) if schema else None


def validate_schema_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Schema payload must be a JSON object.")
    if not payload.get("form_name"):
        raise ValueError("Schema payload must include 'form_name'.")
    fields = payload.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("Schema payload must include a non-empty 'fields' object.")
    for field_name, meta in fields.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Field '{field_name}' must be an object.")
        if not meta.get("type"):
            raise ValueError(f"Field '{field_name}' is missing 'type'.")


def save_schema_payload(payload: Dict[str, Any], schema_id: Optional[str] = None) -> str:
    validate_schema_payload(payload)
    resolved_schema_id = SchemaManager().add_schema(payload, schema_id=schema_id)
    logger.info(
        "[UI] Saved schema form_name=%s schema_id=%s",
        payload.get("form_name"),
        resolved_schema_id,
    )
    return resolved_schema_id


# ---------------------------------------------------------------------------
# Schema ↔ form-builder conversion helpers
# ---------------------------------------------------------------------------

FIELD_TYPES = ["string", "date", "number", "boolean", "currency", "email"]


def schema_to_rows(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a schema's fields dict to a list of row dicts for st.data_editor."""
    rows = []
    for fname, fmeta in schema.get("fields", {}).items():
        rows.append({
            "name": fname,
            "type": fmeta.get("type", "string"),
            "description": fmeta.get("description", ""),
            "required": bool(fmeta.get("required", False)),
            "examples": ", ".join(str(e) for e in fmeta.get("examples", []) if e is not None),
        })
    return rows


def rows_to_schema(
    rows: List[Dict[str, Any]],
    form_name: str,
    version: str,
    description: str,
) -> Dict[str, Any]:
    """Convert data-editor rows back into a schema dict."""
    fields: Dict[str, Any] = {}
    for row in rows:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        raw_examples = str(row.get("examples", "")).strip()
        examples = [e.strip() for e in raw_examples.split(",") if e.strip()]
        fields[name] = {
            "type": row.get("type", "string"),
            "description": str(row.get("description", "")).strip(),
            "required": bool(row.get("required", False)),
            "examples": examples,
        }
    return {
        "form_name": form_name.strip() or "NewSchema",
        "version": version.strip() or "1.0",
        "description": description.strip(),
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# Training command builder
# ---------------------------------------------------------------------------

def _append_flag(command: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def _append_values(command: List[str], flag: str, values: Iterable[str]) -> None:
    values = [v for v in values if v]
    if values:
        command.append(flag)
        command.extend(values)


def build_training_command(
    mode: str,
    model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    use_augmentation: bool,
    normal_datasets: Optional[Iterable[str]] = None,
    use_all_datasets: bool = False,
    curriculum: bool = False,
    lora_groups: Optional[Iterable[str]] = None,
    lora_datasets: Optional[Iterable[str]] = None,
) -> List[str]:
    command = [sys.executable]
    if mode == "full":
        command.extend([
            "train.py",
            "--model", model,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--max-length", str(max_length),
        ])
        _append_flag(command, "--all-datasets", use_all_datasets)
        if not use_all_datasets:
            _append_values(command, "--datasets", normal_datasets or [])
        _append_flag(command, "--curriculum", curriculum)
    else:
        command.extend([
            "train_lora.py",
            "--model", model,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--max-length", str(max_length),
        ])
        _append_values(command, "--groups", lora_groups or [])
        if lora_datasets:
            _append_values(command, "--datasets", lora_datasets)

    if not use_augmentation:
        command.append("--no-augment")
    return command


def run_training_command(command: List[str]) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    logger.info("[UI] Running training command: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=str(FINETUNE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": " ".join(command),
    }


def expected_training_targets(
    mode: str,
    model: str,
    lora_groups: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    if mode == "full":
        root = SCHEMA_TRAINING_ROOT
        if model in ("layoutlmv3", "both"):
            targets.append({
                "label": "LayoutLMv3",
                "path": root / "layoutlmv3",
                "primary_metric": "eval_macro_f1",
                "higher_is_better": True,
            })
        if model in ("donut", "both"):
            targets.append({
                "label": "Donut",
                "path": root / "donut",
                "primary_metric": "eval_cer",
                "higher_is_better": False,
            })
        return targets

    groups = list(lora_groups or [])
    for group in groups:
        if model in ("layoutlmv3", "both"):
            targets.append({
                "label": f"{group} / LayoutLMv3",
                "path": LORA_TRAINING_ROOT / group / "layoutlmv3",
                "primary_metric": "eval_macro_f1",
                "higher_is_better": True,
            })
        if model in ("donut", "both"):
            targets.append({
                "label": f"{group} / Donut",
                "path": LORA_TRAINING_ROOT / group / "donut",
                "primary_metric": "eval_cer",
                "higher_is_better": False,
            })
    return targets


class TrainingProcessMonitor:
    def __init__(self, command: List[str]) -> None:
        self.command = command
        self._process: subprocess.Popen[str] | None = None
        self._stdout_lines: Deque[str] = deque(maxlen=800)
        self._stderr_lines: Deque[str] = deque(maxlen=400)
        self._queue: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
        self._threads: List[threading.Thread] = []

    def start(self) -> "TrainingProcessMonitor":
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self._process = subprocess.Popen(
            self.command,
            cwd=str(FINETUNE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        if self._process.stdout is not None:
            self._threads.append(threading.Thread(
                target=self._reader,
                args=(self._process.stdout, "stdout"),
                daemon=True,
            ))
        if self._process.stderr is not None:
            self._threads.append(threading.Thread(
                target=self._reader,
                args=(self._process.stderr, "stderr"),
                daemon=True,
            ))
        for thread in self._threads:
            thread.start()
        return self

    def _reader(self, stream, stream_name: str) -> None:
        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip()
                self._queue.put((stream_name, line))
        finally:
            self._queue.put((stream_name, None))

    def poll(self) -> Optional[int]:
        if self._process is None:
            return None
        self._drain_queue()
        return self._process.poll()

    def wait(self) -> int:
        if self._process is None:
            raise RuntimeError("Training process was not started.")
        rc = self._process.wait()
        self._drain_queue(force=True)
        return rc

    def _drain_queue(self, force: bool = False) -> None:
        while True:
            try:
                stream_name, line = self._queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                continue
            if stream_name == "stdout":
                self._stdout_lines.append(line)
            else:
                self._stderr_lines.append(line)
        if force:
            for thread in self._threads:
                thread.join(timeout=0.2)

    def stdout_text(self) -> str:
        return "\n".join(self._stdout_lines)

    def stderr_text(self) -> str:
        return "\n".join(self._stderr_lines)

    def combined_tail(self, limit: int = 80) -> str:
        lines = list(self._stdout_lines) + list(self._stderr_lines)
        return "\n".join(lines[-limit:])


def read_training_csv(csv_path: Path) -> List[Dict[str, Any]]:
    import csv

    if not csv_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: Dict[str, Any] = {}
            for key, value in row.items():
                if value in ("", None):
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_current_stage(log_text: str) -> str:
    lines = [line for line in log_text.splitlines() if line.strip()]
    if not lines:
        return "Waiting for first log line"

    latest = lines[-1]
    patterns = [
        (r"Loading .* from", "Loading dataset"),
        (r"Normalize .* train|Normalize .* val", "Normalizing dataset"),
        (r"Augment .* train", "Augmenting training split"),
        (r"Preprocessing", "Encoding model inputs"),
        (r"Starting training", "Training in progress"),
        (r"epoch=", "Evaluating epoch metrics"),
        (r"Plots saved to|Saved overview\.png", "Writing training artifacts"),
        (r"Done", "Run finished"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, latest, re.IGNORECASE):
            return label
    return latest


def summarize_target_progress(target: Dict[str, Any], epochs: int) -> Dict[str, Any]:
    csv_rows = read_training_csv(Path(target["path"]) / "training_log.csv")
    metrics = read_json_if_exists(Path(target["path"]) / "metrics.json")
    current_epoch = int(max((row.get("epoch") or 0) for row in csv_rows)) if csv_rows else 0
    done = bool(metrics)
    progress = min(current_epoch / max(epochs, 1), 1.0)
    primary_metric = target["primary_metric"]
    latest_metric = None
    if csv_rows:
        latest_metric = csv_rows[-1].get(primary_metric)
    if latest_metric is None and metrics:
        latest_metric = metrics.get(primary_metric)
    return {
        **target,
        "csv_rows": csv_rows,
        "metrics": metrics,
        "current_epoch": current_epoch,
        "progress": 1.0 if done else progress,
        "done": done,
        "latest_metric": latest_metric,
        "plots_dir": Path(target["path"]) / "plots",
    }
