"""
Form population pipeline:

  1. Text LLM  — send full document text + schema, get JSON back.
  2. Vision LLM fallback — for any field still None after step 1,
     re-run extraction using a local vision model (Ollama) against
     the page image.
  3. Final sentinel — any field still None after both passes is set
     to the string "NaN" so callers always receive a complete record.
"""
import json
import logging
import re
from typing import Any, Dict, Optional

from utils.form import FormInstance
from config.config import get_config

logger = logging.getLogger(__name__)

# Sentinel used when both extraction passes fail for a field
_MISSING = "NaN"


class FormFiller:
    """
    Populates a form using:
      - Pass 1: text LLM (fast, works on native-text PDFs)
      - Pass 2: vision LLM via Ollama (fallback for fields that came back null)
    """

    def __init__(self, vision_model: str = "llama3.2-vision"):
        self.config = get_config()
        model_cfg = self.config.model

        self.model_name = model_cfg.llm_model
        self.temperature = model_cfg.llm_temperature
        self.max_tokens = model_cfg.llm_max_tokens
        self.vision_model = vision_model

        # Wire up the text LLM backend
        if self.model_name.startswith("ollama/"):
            self.ollama_model = self.model_name.replace("ollama/", "")
            self._text_call = self._call_ollama
        else:
            from transformers import pipeline as hf_pipeline
            self._hf = hf_pipeline(
                "text-generation",
                model=self.model_name,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                device_map=model_cfg.device,
            )
            self._text_call = self._call_hf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(self, clause_graph: Dict[str, str], schema: Dict,
                 full_text: str = "",
                 page_image=None) -> FormInstance:
        """
        Populate every form field with a three-pass strategy.

        Args:
            clause_graph: Hierarchical clause structure
            schema:       Field schema definition
            full_text:    Full plain-text of the document
            page_image:   PIL Image of the document page (used for vision fallback)

        Returns:
            Populated FormInstance — every field is either a real value or "NaN"
        """
        form = FormInstance(schema)
        logger.info(f"Populating form: {schema.get('form_name', 'Unknown')}")

        document_text = full_text or " ".join(str(v) for v in clause_graph.values())

        # ── Pass 1: text LLM ──────────────────────────────────────────
        extracted: Dict[str, Any] = {}
        if document_text.strip():
            logger.info("Pass 1: text LLM extraction")
            prompt = self._build_text_prompt(document_text, schema)
            raw = self._text_call(prompt)
            extracted = self._parse_response(raw, schema)
            _log_pass_result(extracted, pass_name="text LLM")
        else:
            logger.warning("Pass 1 skipped — no document text available")
            extracted = {name: None for name in schema["fields"]}

        # ── Pass 2: vision LLM for null fields ───────────────────────
        null_fields = [k for k, v in extracted.items() if v is None]
        if null_fields and page_image is not None:
            logger.info(
                f"Pass 2: vision LLM fallback for {len(null_fields)} null field(s): "
                f"{', '.join(null_fields)}"
            )
            vision_result = self._vision_extract(page_image, schema, null_fields)
            for field in null_fields:
                if vision_result.get(field) is not None:
                    extracted[field] = vision_result[field]
                    logger.info(f"  [vision] filled '{field}': {vision_result[field]!r}")
            _log_pass_result(extracted, pass_name="vision LLM")
        elif null_fields and page_image is None:
            logger.warning(
                "Pass 2 skipped — no page_image provided. "
                "Pass page_image= to FormFiller.populate() to enable vision fallback."
            )

        # ── Pass 3: sentinel for still-missing fields ─────────────────
        for field_name, value in extracted.items():
            form.fill(field_name, value if value is not None else _MISSING)

        null_after = [k for k, v in form.fields.items() if v == _MISSING]
        if null_after:
            logger.warning(f"Fields set to '{_MISSING}' (not found by any pass): {null_after}")

        return form

    # ------------------------------------------------------------------
    # Pass 1 — text LLM
    # ------------------------------------------------------------------

    def _build_text_prompt(self, document_text: str, schema: Dict) -> str:
        field_lines, skeleton = _schema_to_prompt_parts(schema)

        system = (
            "You are a precise data-extraction engine. "
            "Read the document and fill in every field listed below.\n"
            "Rules:\n"
            "- Output ONLY a single valid JSON object — no prose, no markdown fences.\n"
            "- If a field value is not present in the document, use null.\n"
            "- For dates use ISO format YYYY-MM-DD when possible.\n"
            "- For numbers return only the numeric value.\n"
            "- For booleans return true or false.\n"
            "- Never invent values that are not in the document."
        )
        user = (
            f"DOCUMENT:\n\"\"\"\n{document_text[:4000]}\n\"\"\"\n\n"
            f"FIELDS TO EXTRACT:\n{field_lines}\n\n"
            f"Return a JSON object with exactly these keys:\n{skeleton}"
        )

        if "llama" in self.model_name.lower() or self.model_name.startswith("ollama/"):
            return (
                f"<|start_header_id|>system<|end_header_id|>\n\n{system}"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        return f"<|system|>\n{system}\n</s>\n<|user|>\n{user}\n</s>\n<|assistant|>\n"

    def _call_ollama(self, prompt: str) -> str:
        import ollama
        response = ollama.generate(
            model=self.ollama_model,
            prompt=prompt,
            stream=False,
            options={
                "temperature": self.temperature,
                "num_predict": max(self.max_tokens, 512),
            },
        )
        return response["response"]

    def _call_hf(self, prompt: str) -> str:
        outputs = self._hf(prompt)
        return outputs[0]["generated_text"]

    # ------------------------------------------------------------------
    # Pass 2 — vision LLM
    # ------------------------------------------------------------------

    def _vision_extract(self, page_image, schema: Dict,
                        target_fields: list) -> Dict[str, Any]:
        """
        Run the vision LLM on *target_fields* only.
        Returns a dict; fields not in target_fields are absent from the result.
        """
        try:
            from extraction.llama_extractor import LlamaVisionExtractor
            extractor = LlamaVisionExtractor(model=self.vision_model)

            # Build a sub-schema containing only the null fields so the
            # vision model focuses its attention and returns a smaller JSON.
            sub_schema = {
                **schema,
                "fields": {k: v for k, v in schema["fields"].items()
                           if k in target_fields},
            }
            return extractor.extract(page_image, sub_schema)

        except Exception as exc:
            logger.error(f"[VisionLLM] Pass 2 failed: {exc}")
            return {}

    # ------------------------------------------------------------------
    # Response parsing (shared by both passes)
    # ------------------------------------------------------------------

    def _parse_response(self, response_text: str, schema: Dict) -> Dict[str, Any]:
        json_data = _extract_json(response_text)
        return {
            field_name: _coerce(
                json_data.get(field_name) if json_data else None,
                meta.get("type", "string"),
            )
            for field_name, meta in schema["fields"].items()
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _schema_to_prompt_parts(schema: Dict):
    """Return (field_lines_str, skeleton_str) for prompt construction."""
    lines = []
    for name, meta in schema["fields"].items():
        ftype = meta.get("type", "string")
        desc = meta.get("description", name)
        examples = meta.get("examples", [])
        required = meta.get("required", False)
        line = f'  "{name}": ({ftype}) {desc}'
        if required:
            line += " [REQUIRED]"
        if examples:
            line += f" — e.g. {', '.join(str(e) for e in examples)}"
        lines.append(line)
    skeleton = json.dumps({name: None for name in schema["fields"]}, indent=2)
    return "\n".join(lines), skeleton


def _extract_json(text: str) -> Optional[Dict]:
    """Pull the first JSON object from an LLM response string."""
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text)
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end <= start:
        logger.warning("No JSON object found in LLM response")
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse error: {exc}")
        return None


def _coerce(value: Any, expected_type: str) -> Any:
    """Coerce a raw LLM value to the declared schema type. Returns None on failure."""
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
    except (ValueError, TypeError) as exc:
        logger.debug(f"Coercion failed type={expected_type} value={value!r}: {exc}")
        return None


def _log_pass_result(extracted: Dict[str, Any], pass_name: str):
    filled = sum(1 for v in extracted.values() if v is not None)
    total = len(extracted)
    logger.info(f"  → {pass_name}: {filled}/{total} fields filled")


# ---------------------------------------------------------------------------
# Convenience wrapper (keeps orchestrator call-site working)
# ---------------------------------------------------------------------------

def populate_form(clause_graph: Dict[str, str], schema: Dict,
                  page_image=None) -> FormInstance:
    """Thin wrapper kept for backwards compatibility."""
    return FormFiller().populate(clause_graph, schema, page_image=page_image)