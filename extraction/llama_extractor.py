"""
Vision-based form extraction using a Groq-hosted multimodal model.

This module preserves the old import path (`LlamaVisionExtractor`) so the rest
of the pipeline does not need to change, but it now uses Groq instead of local
Ollama vision models.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from io import BytesIO
from typing import Any, Dict, Optional

from PIL import Image

from config.config import get_config

logger = logging.getLogger(__name__)


class GroqVisionExtractor:
    """Extracts schema fields from a page image using a Groq vision model."""

    def __init__(self, model: Optional[str] = None):
        cfg = get_config()
        self.model = model or cfg.groq.vision_model
        self.api_key = cfg.groq.api_key
        if not self.api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable not set")

        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("groq package not installed. Run: pip install groq") from exc

        self.client = Groq(api_key=self.api_key)
        self.max_page_size = cfg.processing.max_page_size

    def extract(self, page_image: Image.Image, schema: Dict) -> Dict[str, Any]:
        prompt = self._build_prompt(schema)
        image_url = self._image_to_data_url(page_image)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=512,
            )
            text = response.choices[0].message.content or ""
            return self._parse_response(text, schema)
        except Exception as exc:
            logger.error("[GroqVision] Extraction failed: %s", exc)
            return self._null_result(schema)

    def _build_prompt(self, schema: Dict) -> str:
        field_lines = []
        for field_name, meta in schema.get("fields", {}).items():
            field_type = meta.get("type", "string")
            description = meta.get("description", field_name)
            examples = meta.get("examples", [])
            required = meta.get("required", False)
            line = f'  "{field_name}": ({field_type}) {description}'
            if required:
                line += " [REQUIRED]"
            if examples:
                line += f" -- e.g. {', '.join(str(example) for example in examples)}"
            field_lines.append(line)

        fields_block = "\n".join(field_lines)
        skeleton = json.dumps({name: None for name in schema.get("fields", {})}, indent=2)
        return (
            "You are a document data extraction engine.\n"
            "Look at the document image and extract the requested fields.\n\n"
            "Rules:\n"
            "- Return ONLY a valid JSON object.\n"
            "- If a field is not visible, use null.\n"
            "- Do not invent values.\n"
            "- Dates should be ISO format YYYY-MM-DD when possible.\n"
            "- Numbers should be numeric values only.\n\n"
            f"Fields to extract:\n{fields_block}\n\n"
            f"Return exactly this JSON shape:\n{skeleton}"
        )

    def _image_to_data_url(self, image: Image.Image) -> str:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image = self._resize(image)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _resize(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_dim = max(width, height)
        if max_dim <= self.max_page_size:
            return image
        scale = self.max_page_size / max_dim
        return image.resize((int(width * scale), int(height * scale)))

    def _parse_response(self, text: str, schema: Dict) -> Dict[str, Any]:
        text = re.sub(r"```(?:json)?\s*|\s*```", "", text)
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1 or end <= start:
            logger.warning("[GroqVision] No JSON object in response")
            return self._null_result(schema)
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            logger.warning("[GroqVision] JSON parse error: %s", exc)
            return self._null_result(schema)

        return {
            name: self._coerce(data.get(name), meta.get("type", "string"))
            for name, meta in schema.get("fields", {}).items()
        }

    def _coerce(self, value: Any, expected_type: str) -> Any:
        if value is None or str(value).strip().lower() in ("null", "none", "n/a", ""):
            return None
        try:
            if expected_type == "date":
                text = str(value).strip()
                return text if re.search(r"\d{4}", text) else None
            if expected_type == "number":
                return float(value) if "." in str(value) else int(value)
            if expected_type == "boolean":
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ("true", "yes", "1")
            return str(value).strip() or None
        except (ValueError, TypeError):
            return None

    def _null_result(self, schema: Dict) -> Dict[str, Any]:
        return {name: None for name in schema.get("fields", {})}


LlamaVisionExtractor = GroqVisionExtractor
