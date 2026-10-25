"""
Vision-based form extraction using a Groq-hosted multimodal model.

This module preserves the old import path (`LlamaVisionExtractor`) so the rest
of the pipeline does not need to change, but it now uses Groq instead of local
Ollama vision models.
"""
from __future__ import annotations

import ast
import base64
import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict, Optional

from PIL import Image

from config.config import get_config
from utils.llm_cache import cached_call, make_key, record_usage

logger = logging.getLogger(__name__)

# Count of vision calls that failed (API error or unparseable reply). Evaluation
# runs read this to avoid caching results produced while the API was failing.
FAILURES = {"count": 0}
# Pages where the model kept looping even after a reload and a repetition penalty. This is the model
# failing to read the page (scored as an empty answer), not an infrastructure fault, so it is
# counted apart from FAILURES and reported in the results.
DEGENERATE = {"count": 0}


class DegenerateOutput(RuntimeError):
    """The local model produced a repetition loop on every attempt for this page."""


class VisionExtractor:
    """Extracts schema fields from a page image with a vision model (Groq-hosted or local Ollama)."""

    def __init__(self, model: Optional[str] = None):
        cfg = get_config()
        self.backend = cfg.vision_backend
        self.max_page_size = cfg.processing.max_page_size
        if self.backend == "ollama":
            self.model = model or cfg.ollama_vision_model
            self.client = None
            return
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
            def _request_ollama() -> str:
                import ollama
                messages = [{"role": "user", "content": prompt, "images": [image_url.split(",", 1)[1]]}]
                # Ollama's default 4096-token window is too small for a page image plus the prompt.
                options = {"temperature": 0.0, "num_predict": 512, "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192"))}
                try:
                    response = ollama.chat(model=self.model, messages=messages, options=options)
                except ollama.ResponseError as exc:
                    if "repeat" not in str(exc).lower():
                        raise
                    # Greedy decoding fell into a loop. A runner left in a bad state (e.g. after
                    # context-overflow errors) emits the same garbage for every request until the
                    # model is reloaded, so unload it and retry once with a repetition penalty.
                    ollama.generate(model=self.model, prompt="", keep_alive=0)
                    try:
                        response = ollama.chat(model=self.model, messages=messages,
                                               options={**options, "repeat_penalty": 1.15})
                    except ollama.ResponseError as exc2:
                        if "repeat" not in str(exc2).lower():
                            raise
                        raise DegenerateOutput(str(exc2)) from exc2
                record_usage("ollama", 1, response.get("prompt_eval_count", 0), response.get("eval_count", 0))
                return response["message"]["content"] or ""

            def _request() -> str:
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
                usage = getattr(response, "usage", None)
                record_usage("groq", 1, getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))
                return response.choices[0].message.content or ""

            if self.backend == "ollama":
                text = cached_call("ollama_vision", make_key(self.model, prompt, image_url), _request_ollama)
            else:
                text = cached_call("groq_vision", make_key(self.model, prompt, image_url), _request, groq=True)
            return self._parse_response(text or "", schema)
        except DegenerateOutput as exc:
            DEGENERATE["count"] += 1
            logger.warning("[Vision:%s] model looped on this page, scored as empty: %s", self.backend, exc)
            return self._null_result(schema)
        except Exception as exc:
            FAILURES["count"] += 1
            logger.error("[Vision:%s] Extraction failed: %s", self.backend, exc)
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
            FAILURES["count"] += 1
            logger.warning("[Vision:%s] No JSON object in response", self.backend)
            return self._null_result(schema)
        try:
            data = self._load_object(text[start:end])
        except ValueError as exc:
            FAILURES["count"] += 1
            logger.warning("[Vision:%s] JSON parse error: %s", self.backend, exc)
            return self._null_result(schema)

        return {
            name: self._coerce(data.get(name), meta.get("type", "string"))
            for name, meta in schema.get("fields", {}).items()
        }

    @staticmethod
    def _load_object(raw: str) -> Dict[str, Any]:
        """json.loads, then a tolerant retry for the near-JSON small models emit
        (single quotes, trailing commas, Python literals)."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError as first:
            repaired = re.sub(r",\s*([}\]])", r"\1", raw)
            repaired = re.sub(r"\b(null|true|false)\b", lambda m: m.group(1).capitalize() if m.group(1) != "null" else "None", repaired)
            try:
                data = ast.literal_eval(repaired)
            except (ValueError, SyntaxError):
                raise ValueError(str(first)) from first
            if not isinstance(data, dict):
                raise ValueError("response is not a JSON object") from first
            return data

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


GroqVisionExtractor = LlamaVisionExtractor = VisionExtractor   # old import names
