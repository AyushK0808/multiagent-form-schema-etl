"""
Vision-based form extraction using a local multimodal LLM via Ollama.

Used as a fallback when text-based extraction returns null fields.

Setup (pick whichever fits in your RAM):
    ollama pull moondream          # ~1.7 GB — lightest option
    ollama pull minicpm-v          # ~5 GB
    ollama pull llava:7b           # ~4.5 GB
    ollama pull llama3.2-vision    # ~10 GB VRAM / ~8 GB RAM with CPU offload
"""
import base64
import json
import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Ordered preference — first available model on the machine wins.
# Lightest models are listed first so low-RAM machines get a working default.
VISION_MODEL_PREFERENCE = [
    "moondream",
    "minicpm-v",
    "llava:7b",
    "llava",
    "llama3.2-vision",
]


def _normalise(name: str) -> str:
    """Strip ':latest' so bare names match tagged Ollama model names."""
    return name.removesuffix(":latest")


def _get_available_models_in_order(requested: Optional[str] = None) -> List[str]:
    """
    Return available vision models sorted by VISION_MODEL_PREFERENCE order.
    If requested provided, try it first.
    """
    try:
        import ollama

        available_raw: List[str] = [m.model for m in ollama.list().models]
        norm_to_raw = {_normalise(r): r for r in available_raw}

        ordered = []

        # if user explicitly requested model → try first
        if requested and _normalise(requested) in norm_to_raw:
            ordered.append(norm_to_raw[_normalise(requested)])

        # then go by preference
        for candidate in VISION_MODEL_PREFERENCE:
            norm = _normalise(candidate)
            if norm in norm_to_raw:
                model = norm_to_raw[norm]
                if model not in ordered:
                    ordered.append(model)

        if not ordered:
            logger.warning(
                f"[VisionLLM] No known vision models found. Available: {available_raw}"
            )

        logger.info(f"[VisionLLM] Model fallback order: {ordered}")
        return ordered

    except ImportError:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")
    except Exception as exc:
        logger.warning(f"[VisionLLM] Could not reach Ollama: {exc}")
        return []


class LlamaVisionExtractor:
    """
    Extracts form fields from a page image using a local Ollama vision model.

    - Model is resolved automatically from what is available on the machine.
    - On OOM the call is retried with CPU-only inference (num_gpu=0).
    """

    def __init__(self, model: Optional[str] = None):
        self.models = _get_available_models_in_order(model)

        if not self.models:
            raise RuntimeError(
            "[VisionLLM] No vision-capable model available in Ollama. "
            "Pull one first, e.g.: ollama pull moondream"
        )

        logger.info(f"[VisionLLM] Using fallback chain: {self.models}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, page_image: Image.Image, schema: Dict) -> Dict[str, Any]:
        """
        Try models in VISION_MODEL_PREFERENCE order until one succeeds.
        """

        img_b64 = self._image_to_b64(page_image)
        prompt = self._build_prompt(schema)

        for model in self.models:
            self.model = model
            logger.info(f"[VisionLLM] Trying model: {model}")

            for attempt, num_gpu in enumerate([-1, 0]):
                mode = "GPU" if num_gpu != 0 else "CPU-only"

                try:
                    logger.info(f"[VisionLLM] {model} Attempt {attempt + 1} ({mode})")

                    raw = self._call_ollama(prompt, img_b64, num_gpu=num_gpu)
                    result = self._parse_response(raw, schema)

                    filled = sum(1 for v in result.values() if v is not None)

                    if filled > 0:
                        logger.info(
                            f"[VisionLLM] SUCCESS with {model} "
                            f"({filled}/{len(result)} fields)"
                        )
                        return result

                    logger.warning(
                        f"[VisionLLM] {model} returned empty result — trying next model"
                    )

                except Exception as exc:
                    is_oom = any(
                        kw in str(exc).lower()
                        for kw in ("memory", "oom", "500")
                    )

                    if is_oom and num_gpu != 0:
                        logger.warning(
                            f"[VisionLLM] {model} OOM — retry CPU-only"
                        )
                        continue

                    logger.warning(
                        f"[VisionLLM] {model} failed: {exc} — trying next model"
                    )
                    break

        logger.error("[VisionLLM] All vision models failed")
        return self._null_result(schema)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str, img_b64: str,
                     num_gpu: int = -1) -> str:
        import ollama
        # Minimize hallucinations: low temp + constrained token budget
        options: Dict[str, Any] = {"temperature": 0.0, "num_predict": 256}
        if num_gpu == 0:
            options["num_gpu"] = 0      # force CPU-only inference
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            images=[img_b64],
            stream=False,
            format="json",
            options=options,
        )
        return response.get("response", "")

    def _build_prompt(self, schema: Dict) -> str:
        field_lines = []
        for field_name, meta in schema.get("fields", {}).items():
            ftype = meta.get("type", "string")
            desc = meta.get("description", field_name)
            examples = meta.get("examples", [])
            required = meta.get("required", False)
            line = f'  "{field_name}": ({ftype}) {desc}'
            if required:
                line += " [REQUIRED]"
            if examples:
                line += f" — e.g. {', '.join(str(e) for e in examples)}"
            field_lines.append(line)

        fields_block = "\n".join(field_lines)
        skeleton = json.dumps(
            {name: None for name in schema.get("fields", {})}, indent=2
        )
        return (
            "You are a document data-extraction engine.\n"
            "Look at the attached document image and extract the fields below.\n\n"
            "Rules:\n"
            "- Return ONLY a valid JSON object — no prose, no markdown fences.\n"
            "- If a field value is not visible in the document, use null.\n"
            "- For dates use ISO format YYYY-MM-DD when possible.\n"
            "- For numbers return only the numeric value.\n"
            "- Never invent values not present in the image.\n\n"
            f"Fields to extract:\n{fields_block}\n\n"
            f"Return a JSON object with exactly these keys:\n{skeleton}"
        )

    def _image_to_b64(self, image: Image.Image) -> str:
        buf = BytesIO()
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _parse_response(self, text: str, schema: Dict) -> Dict[str, Any]:
        text = re.sub(r"```(?:json)?\s*|\s*```", "", text)
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1 or end <= start:
            logger.warning("[VisionLLM] No JSON object in response")
            return self._null_result(schema)
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            logger.warning(f"[VisionLLM] JSON parse error: {exc}")
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
                s = str(value).strip()
                return s if re.search(r"\d{4}", s) else None
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