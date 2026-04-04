"""
Schema recognition using fine-tuned LayoutLMv3 and Donut checkpoints.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from pytesseract import TesseractNotFoundError

logger = logging.getLogger(__name__)

_DONUT_TASK_PROMPT = "<s_schema_recognition>"


def _avg_max_prob(scores) -> float:
    if not scores:
        return 0.0
    probs = [torch.softmax(step, dim=-1).max(dim=-1).values.mean().item() for step in scores]
    return float(sum(probs) / len(probs))


def _is_local_checkpoint(model_ref: str) -> bool:
    return Path(model_ref).exists()


def _select_model_source(primary_model: str, fallback_model: Optional[str]) -> tuple[str, bool]:
    if _is_local_checkpoint(primary_model):
        return primary_model, False
    if fallback_model:
        logger.info(
            "[SchemaRecognizer] Local checkpoint '%s' not found, falling back to Hugging Face model '%s'",
            primary_model,
            fallback_model,
        )
        return fallback_model, True
    return primary_model, False


class LayoutLMv3SchemaRecognizer:
    def __init__(self, checkpoint_path: str, fallback_model: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.fallback_model = fallback_model
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.using_fallback = False

    def available(self) -> bool:
        return _is_local_checkpoint(self.checkpoint_path) or bool(self.fallback_model)

    def _load(self) -> None:
        if self.model is not None:
            return
        from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

        model_source, using_fallback = _select_model_source(self.checkpoint_path, self.fallback_model)
        self.using_fallback = using_fallback
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_source,
            apply_ocr=True,
            use_fast=False,
        )
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(model_source)
        self.model.to(self.device).eval()

    def predict(self, image) -> Optional[Dict]:
        if not self.available():
            return None
        self._load()
        try:
            encoded = self.processor(images=image, return_tensors="pt", truncation=True, padding="max_length")
        except TesseractNotFoundError:
            logger.warning(
                "[SchemaRecognizer] Skipping LayoutLMv3 schema recognition because Tesseract is not installed"
            )
            return None
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        label_id = int(torch.argmax(probs).item())
        label = self.model.config.id2label[str(label_id)] if str(label_id) in self.model.config.id2label else self.model.config.id2label[label_id]
        source = "layoutlmv3_hf_fallback" if self.using_fallback else "layoutlmv3"
        return {"schema_name": label, "confidence": float(probs[label_id].item()), "source": source}


class DonutSchemaRecognizer:
    def __init__(self, checkpoint_path: str, fallback_model: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.fallback_model = fallback_model
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.using_fallback = False

    def available(self) -> bool:
        return _is_local_checkpoint(self.checkpoint_path) or bool(self.fallback_model)

    def _load(self) -> None:
        if self.model is not None:
            return
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        model_source, using_fallback = _select_model_source(self.checkpoint_path, self.fallback_model)
        self.using_fallback = using_fallback
        self.processor = DonutProcessor.from_pretrained(model_source, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_source)
        self.model.to(self.device).eval()

    def predict(self, image) -> Optional[Dict]:
        if not self.available():
            return None
        self._load()
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        decoder_input_ids = self.processor.tokenizer(
            _DONUT_TASK_PROMPT,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=48,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        decoded = self.processor.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
        prediction = self._parse(decoded)
        if not prediction:
            return None
        prediction["confidence"] = _avg_max_prob(outputs.scores)
        prediction["source"] = "donut_hf_fallback" if self.using_fallback else "donut"
        return prediction

    @staticmethod
    def _parse(decoded: str) -> Optional[Dict]:
        stripped = decoded.replace(_DONUT_TASK_PROMPT, "").strip()
        try:
            obj = json.loads(stripped)
            schema_name = obj.get("schema")
            if schema_name:
                return {"schema_name": schema_name}
        except json.JSONDecodeError:
            pass
        return None


class SchemaRecognizer:
    def __init__(
        self,
        layout_model_path: str,
        donut_model_path: str,
        layout_fallback_model: Optional[str] = None,
        donut_fallback_model: Optional[str] = None,
    ):
        self.layout = LayoutLMv3SchemaRecognizer(layout_model_path, fallback_model=layout_fallback_model)
        self.donut = DonutSchemaRecognizer(donut_model_path, fallback_model=donut_fallback_model)

    def predict(self, image) -> Dict:
        candidates = [result for result in (self.layout.predict(image), self.donut.predict(image)) if result]
        if not candidates:
            raise RuntimeError(
                "Automatic schema recognition could not determine a schema. "
                "Provide --form/--schema-id, install Tesseract for LayoutLMv3 OCR, "
                "or fine-tune the schema recognition checkpoints."
            )

        by_name: Dict[str, Dict] = {}
        for candidate in candidates:
            existing = by_name.get(candidate["schema_name"])
            if existing is None or candidate["confidence"] > existing["confidence"]:
                by_name[candidate["schema_name"]] = candidate

        if len(by_name) == 1:
            return next(iter(by_name.values()))

        best = max(by_name.values(), key=lambda item: item["confidence"])
        logger.info("[SchemaRecognizer] Disagreement resolved in favor of %s (%.3f)", best["schema_name"], best["confidence"])
        return best
