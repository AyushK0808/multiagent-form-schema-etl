"""
extraction/form_filler.py
==========================
Form population pipeline:

  Pass 1: Text LLM  — full document text + schema → JSON
  Pass 2: Vision LLM fallback  — Groq vision model for null fields
  Pass 3: Sentinel — remaining null fields set to "NaN"

LoRA adapter routing
--------------------
`adapter_group` is passed in from ParallelExtractor (which receives it from
the orchestrator).  It is stored on the instance and forwarded to any
LayoutAnalyzer calls made inside this class so the correct adapter is active.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from utils.form import FormInstance
from config.config import get_config

logger = logging.getLogger(__name__)

_MISSING = "NaN"


class FormFiller:

    def __init__(
        self,
        vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        adapter_group: str = "group_2",
    ):
        self.config       = get_config()
        model_cfg         = self.config.model
        self.model_name   = model_cfg.llm_model
        self.temperature  = model_cfg.llm_temperature
        self.max_tokens   = model_cfg.llm_max_tokens
        self.vision_model = vision_model
        self.adapter_group = adapter_group   # forwarded to LayoutAnalyzer

        if self.model_name.startswith("ollama/"):
            self.ollama_model = self.model_name.replace("ollama/", "")
            self._text_call   = self._call_ollama
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

    def populate(
        self,
        clause_graph: Dict[str, str],
        schema: Dict,
        full_text: str = "",
        page_image=None,
    ) -> FormInstance:
        form = FormInstance(schema)
        logger.info("Populating form: %s  (adapter_group=%s)",
                    schema.get("form_name", "Unknown"), self.adapter_group)

        document_text = full_text or " ".join(str(v) for v in clause_graph.values())

        # Pass 1 — text LLM
        extracted: Dict[str, Any] = {}
        if document_text.strip():
            prompt    = self._build_text_prompt(document_text, schema)
            raw       = self._text_call(prompt)
            extracted = self._parse_response(raw, schema)
            _log_pass_result(extracted, "text LLM")
        else:
            logger.warning("Pass 1 skipped — no document text")
            extracted = {name: None for name in schema["fields"]}

        # Pass 2 — vision LLM for null fields
        null_fields = [k for k, v in extracted.items() if v is None]
        if null_fields and page_image is not None:
            logger.info("Pass 2: vision fallback for %d field(s)", len(null_fields))
            vision_result = self._vision_extract(page_image, schema, null_fields)
            for f in null_fields:
                if vision_result.get(f) is not None:
                    extracted[f] = vision_result[f]
            _log_pass_result(extracted, "vision LLM")
        elif null_fields and page_image is None:
            logger.warning("Pass 2 skipped — no page_image provided")

        # Pass 3 — sentinel
        for field_name, value in extracted.items():
            form.fill(field_name, value if value is not None else _MISSING)

        null_after = [k for k, v in form.fields.items() if v == _MISSING]
        if null_after:
            logger.warning("Fields set to '%s': %s", _MISSING, null_after)

        return form

    # ------------------------------------------------------------------
    # Pass 1 helpers
    # ------------------------------------------------------------------

    def _build_text_prompt(self, document_text: str, schema: Dict) -> str:
        field_lines, skeleton = _schema_to_prompt_parts(schema)
        system = (
            "You are a precise data-extraction engine. "
            "Read the document and fill every field listed below.\n"
            "Rules:\n"
            "- Output ONLY a single valid JSON object — no prose, no markdown.\n"
            "- Missing values → null. Dates → ISO YYYY-MM-DD. Numbers → numeric only.\n"
            "- Never invent values."
        )
        user = (
            f"DOCUMENT:\n\"\"\"\n{document_text[:4000]}\n\"\"\"\n\n"
            f"FIELDS:\n{field_lines}\n\n"
            f"Return JSON with exactly these keys:\n{skeleton}"
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
            model=self.ollama_model, prompt=prompt, stream=False,
            options={"temperature": self.temperature, "num_predict": max(self.max_tokens, 512)},
        )
        return response["response"]

    def _call_hf(self, prompt: str) -> str:
        return self._hf(prompt)[0]["generated_text"]

    # ------------------------------------------------------------------
    # Pass 2 helpers
    # ------------------------------------------------------------------

    def _vision_extract(self, page_image, schema: Dict, target_fields: list) -> Dict[str, Any]:
        try:
            from extraction.llama_extractor import LlamaVisionExtractor
            extractor  = LlamaVisionExtractor(model=self.vision_model)
            sub_schema = {
                **schema,
                "fields": {k: v for k, v in schema["fields"].items() if k in target_fields},
            }
            return extractor.extract(page_image, sub_schema)
        except Exception as exc:
            logger.error("[VisionLLM] Pass 2 failed: %s", exc)
            return {}

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
# Module-level helpers  (unchanged)
# ---------------------------------------------------------------------------

def _schema_to_prompt_parts(schema: Dict):
    lines = []
    for name, meta in schema["fields"].items():
        line = f'  "{name}": ({meta.get("type","string")}) {meta.get("description", name)}'
        if meta.get("required"):
            line += " [REQUIRED]"
        if meta.get("examples"):
            line += f" -- e.g. {', '.join(str(e) for e in meta['examples'])}"
        lines.append(line)
    return "\n".join(lines), json.dumps({n: None for n in schema["fields"]}, indent=2)


def _extract_json(text: str) -> Optional[Dict]:
    text  = re.sub(r"```(?:json)?\s*|\s*```", "", text)
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _coerce(value: Any, expected_type: str) -> Any:
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


def _log_pass_result(extracted: Dict[str, Any], pass_name: str):
    filled = sum(1 for v in extracted.values() if v is not None)
    logger.info("  [%s] %d/%d fields filled", pass_name, filled, len(extracted))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def populate_form(
    clause_graph: Dict[str, str],
    schema: Dict,
    page_image=None,
    adapter_group: str = "group_2",
) -> FormInstance:
    return FormFiller(adapter_group=adapter_group).populate(
        clause_graph, schema, page_image=page_image
    )