"""
Schema recognition using fine-tuned LayoutLMv3 and Donut checkpoints.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

_DONUT_TASK_PROMPT = "<s_schema_recognition>"


def _avg_max_prob(scores) -> float:
    if not scores:
        return 0.0
    probs = [torch.softmax(step, dim=-1).max(dim=-1).values.mean().item() for step in scores]
    return float(sum(probs) / len(probs))


class LayoutLMv3SchemaRecognizer:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def available(self) -> bool:
        return self.checkpoint_path.exists()

    def _load(self) -> None:
        if self.model is not None:
            return
        from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

        self.processor = LayoutLMv3Processor.from_pretrained(str(self.checkpoint_path), apply_ocr=True)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(str(self.checkpoint_path))
        self.model.to(self.device).eval()

    def predict(self, image) -> Optional[Dict]:
        if not self.available():
            return None
        self._load()
        encoded = self.processor(images=image, return_tensors="pt", truncation=True, padding="max_length")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        label_id = int(torch.argmax(probs).item())
        label = self.model.config.id2label[str(label_id)] if str(label_id) in self.model.config.id2label else self.model.config.id2label[label_id]
        return {"schema_name": label, "confidence": float(probs[label_id].item()), "source": "layoutlmv3"}


class DonutSchemaRecognizer:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def available(self) -> bool:
        return self.checkpoint_path.exists()

    def _load(self) -> None:
        if self.model is not None:
            return
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        self.processor = DonutProcessor.from_pretrained(str(self.checkpoint_path))
        self.model = VisionEncoderDecoderModel.from_pretrained(str(self.checkpoint_path))
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
        prediction["source"] = "donut"
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
    def __init__(self, layout_model_path: str, donut_model_path: str):
        self.layout = LayoutLMv3SchemaRecognizer(layout_model_path)
        self.donut = DonutSchemaRecognizer(donut_model_path)

    def predict(self, image) -> Dict:
        candidates = [result for result in (self.layout.predict(image), self.donut.predict(image)) if result]
        if not candidates:
            raise FileNotFoundError("No schema recognition checkpoint found for LayoutLMv3 or Donut")

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
