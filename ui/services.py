from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import fitz
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import get_config
from schema.schema import SchemaManager
from schema.schema_recognizer import SchemaRecognizer


UPLOAD_DIR = ROOT_DIR / "data" / "raw" / "ui_uploads"
FINETUNE_DIR = ROOT_DIR / "finetune"


def ensure_ui_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file) -> Path:
    ensure_ui_dirs()
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    target = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def load_preview_image(file_path: Path) -> Image.Image:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        with fitz.open(str(file_path)) as doc:
            page = doc.load_page(0)
            pix = page.get_pixmap()
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return Image.open(file_path).convert("RGB")


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
                "examples": [None],
            }
        },
    }


def recognize_schema_from_image(image: Image.Image) -> Dict[str, Any]:
    cfg = get_config()
    recognizer = SchemaRecognizer(
        layout_model_path=cfg.model.schema_recognition_layout_model,
        donut_model_path=cfg.model.schema_recognition_donut_model,
        layout_fallback_model=cfg.model.schema_recognition_layout_fallback_model,
        donut_fallback_model=cfg.model.schema_recognition_donut_fallback_model,
    )
    prediction = recognizer.predict(image)

    manager = SchemaManager()
    existing = manager.get_schema(form_name=prediction["schema_name"])
    schema_payload = existing or build_schema_template(prediction["schema_name"])
    return {
        "prediction": prediction,
        "schema_payload": schema_payload,
        "schema_found": existing is not None,
    }


def list_schemas() -> list[Dict[str, Any]]:
    return SchemaManager().list_schemas()


def load_schema_by_name(form_name: str) -> Optional[Dict[str, Any]]:
    return SchemaManager().get_schema(form_name=form_name)


def validate_schema_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Schema payload must be a JSON object")
    if not payload.get("form_name"):
        raise ValueError("Schema payload must include 'form_name'")
    fields = payload.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("Schema payload must include a non-empty 'fields' object")
    for field_name, meta in fields.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Field '{field_name}' must be an object")
        if not meta.get("type"):
            raise ValueError(f"Field '{field_name}' is missing 'type'")


def save_schema_payload(payload: Dict[str, Any], schema_id: Optional[str] = None) -> str:
    validate_schema_payload(payload)
    return SchemaManager().add_schema(payload, schema_id=schema_id)


def _append_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def _append_values(command: list[str], flag: str, values: Iterable[str]) -> None:
    values = [value for value in values if value]
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
) -> list[str]:
    command = [sys.executable]
    if mode == "full":
        command.extend(
            [
                "train.py",
                "--model",
                model,
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--learning-rate",
                str(learning_rate),
                "--max-length",
                str(max_length),
            ]
        )
        _append_flag(command, "--all-datasets", use_all_datasets)
        if not use_all_datasets:
            _append_values(command, "--datasets", normal_datasets or [])
        _append_flag(command, "--curriculum", curriculum)
    else:
        command.extend(
            [
                "train_lora.py",
                "--model",
                model,
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--learning-rate",
                str(learning_rate),
                "--max-length",
                str(max_length),
            ]
        )
        _append_values(command, "--groups", lora_groups or [])

    if not use_augmentation:
        command.append("--no-augment")
    return command


def run_training_command(command: list[str]) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
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
