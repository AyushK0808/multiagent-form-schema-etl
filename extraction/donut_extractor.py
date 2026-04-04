"""
OCR-free document extraction using Donut (Document Understanding Transformer).

Donut skips traditional OCR entirely: a Swin-Transformer vision encoder reads
the raw pixel image, and a BART-style decoder generates structured text.  We
run it in DocVQA mode — one question per schema field — so no fine-tuning is
required for new document types.

Model: naver-clova-ix/donut-base  (~400 MB)
Pull once with:  huggingface-cli download naver-clova-ix/donut-base
"""
from __future__ import annotations

import logging
import re
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

_DONUT_MODEL_ID = "naver-clova-ix/donut-base"
_TASK_PROMPT    = "<s_docvqa><s_question>{question}</s_question><s_answer>"


class DonutExtractor:
    """
    Field extractor that operates purely on pixel data — no OCR step.

    Each schema field is extracted via a VQA round-trip:
        question = "What is the <field description>?"
        answer   = Donut(image, question)

    Returns per-field confidence scores derived from the mean token probability
    of the generated answer sequence (higher = more certain).
    """

    def __init__(self, model_id: str = _DONUT_MODEL_ID):
        self.model_id   = model_id
        self._processor = None
        self._model     = None
        self._device    = "cpu"

    # ------------------------------------------------------------------
    # Lazy loading — only pulls weights when first called
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            logger.info(f"[Donut] Loading {self.model_id} …")
            self._processor = DonutProcessor.from_pretrained(self.model_id, use_fast=False)
            self._model     = VisionEncoderDecoderModel.from_pretrained(self.model_id)
            self._device    = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device).eval()
            logger.info(f"[Donut] Ready on {self._device}")
        except Exception as exc:
            logger.error(f"[Donut] Model load failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        page_image: Image.Image,
        schema: Dict,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract every schema field from the page image.

        Returns
        -------
        fields      : {field_name: value_or_None}
        confidences : {field_name: 0.0–1.0}
        """
        self._load()

        fields:      Dict[str, Any]   = {}
        confidences: Dict[str, float] = {}

        for field_name, meta in schema.get("fields", {}).items():
            question   = self._build_question(field_name, meta)
            value, conf = self._vqa(page_image, question)
            # Post-process answer to schema type
            value = self._coerce(value, meta.get("type", "string"))
            fields[field_name]      = value
            confidences[field_name] = conf
            logger.debug(f"[Donut] {field_name!r} → {value!r}  (conf={conf:.3f})")

        filled = sum(1 for v in fields.values() if v is not None)
        logger.info(f"[Donut] {filled}/{len(fields)} fields extracted")
        return fields, confidences

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_question(self, field_name: str, meta: Dict) -> str:
        description = meta.get("description", field_name.replace("_", " "))
        examples    = meta.get("examples", [])
        q = f"What is the {description}?"
        if examples:
            q += f" (e.g. {examples[0]})"
        return q

    def _vqa(
        self,
        image: Image.Image,
        question: str,
    ) -> Tuple[Optional[str], float]:
        """Single VQA inference pass. Returns (answer, confidence)."""
        try:
            prompt   = _TASK_PROMPT.format(question=question)
            enc_img  = self._processor(image, return_tensors="pt").pixel_values.to(self._device)
            dec_ids  = self._processor.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(self._device)

            with torch.no_grad():
                out = self._model.generate(
                    enc_img,
                    decoder_input_ids=dec_ids,
                    max_new_tokens=64,
                    pad_token_id=self._processor.tokenizer.pad_token_id,
                    eos_token_id=self._processor.tokenizer.convert_tokens_to_ids("</s_answer>"),
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Decode answer
            seq = self._processor.batch_decode(out.sequences)[0]
            answer = self._parse_answer(seq)

            # Confidence = mean max-probability across generated tokens
            confidence = self._score_confidence(out.scores)

            return (answer if answer else None), confidence

        except Exception as exc:
            logger.warning(f"[Donut] VQA failed for question={question!r}: {exc}")
            return None, 0.0

    @staticmethod
    def _parse_answer(sequence: str) -> str:
        """Pull the text between <s_answer> … </s_answer>."""
        m = re.search(r"<s_answer>(.*?)(</s_answer>|$)", sequence, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: strip all tags
        return re.sub(r"<[^>]+>", "", sequence).strip()

    @staticmethod
    def _score_confidence(scores) -> float:
        if not scores:
            return 0.5
        per_step = [
            F.softmax(s, dim=-1).max(dim=-1).values.mean().item()
            for s in scores
        ]
        return float(sum(per_step) / len(per_step))

    @staticmethod
    def _coerce(value: Optional[str], field_type: str) -> Any:
        if not value:
            return None
        value = value.strip()
        if not value or value.lower() in ("not mentioned", "n/a", "none", "unknown"):
            return None
        try:
            if field_type == "date":
                # Accept anything containing a 4-digit year
                return value if re.search(r"\d{4}", value) else None
            if field_type == "number":
                cleaned = re.sub(r"[^\d.]", "", value)
                return float(cleaned) if "." in cleaned else int(cleaned)
            if field_type == "boolean":
                return value.lower() in ("yes", "true", "1")
        except (ValueError, TypeError):
            pass
        return value