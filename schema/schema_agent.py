"""
Schema Resolution Agent
=======================
Uses a Groq-hosted LLM (llama-3.3-70b-versatile by default — fast, cheap,
strong at structured output) to perform three policy-governed tasks:

  1. NORMALISE  — semantic field normalisation: remove jargon, standardise
                  field name semantics across document types.
  2. MAP        — map normalised fields onto the fields of a candidate schema
                  retrieved from the SchemaRegistry.
  3. SYNTHESISE — when no compatible schema is found, synthesise a new schema
                  definition from the observed fields and sample values.

All prompts request JSON-only output.  The agent validates the response
structure before returning — malformed JSON triggers one retry with an
explicit repair prompt.

Environment
-----------
  GROQ_API_KEY  (required)  — set in .env or environment
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_SMALL_MODEL = "llama-3.1-8b-instant"
_DEFAULT_SYNTHESIS_MODEL = "llama-3.3-70b-versatile"
_MAX_RETRIES        = 2


# ---------------------------------------------------------------------------
# Groq client helper
# ---------------------------------------------------------------------------

def _groq_client():
    """Return an initialised Groq client (raises if key is missing)."""
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable not set")
        return Groq(api_key=api_key)
    except ImportError:
        raise RuntimeError("groq package not installed. Run: pip install groq")


def _call_groq(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Single Groq chat completion.  Returns the assistant message text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _extract_json(text: str) -> Optional[Dict]:
    """Pull the first JSON object from an LLM response string."""
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text)
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SchemaAgent:
    """
    Groq-powered schema-resolution agent.

    Parameters
    ----------
    small_model : Groq model ID for normalization and mapping
    synthesis_model : Groq model ID for schema synthesis
    """

    def __init__(
        self,
        small_model: str = _DEFAULT_SMALL_MODEL,
        synthesis_model: str = _DEFAULT_SYNTHESIS_MODEL,
    ):
        self.small_model = small_model
        self.synthesis_model = synthesis_model
        self.client = _groq_client()

    # ------------------------------------------------------------------
    # 1. Field normalisation
    # ------------------------------------------------------------------

    def normalise_fields(
        self,
        raw_fields: Dict[str, Any],
        document_hint: str = "",
    ) -> Dict[str, Any]:
        """
        Normalise extracted field values:
          - Standardise date formats to ISO 8601.
          - Remove legalese / boilerplate from string values.
          - Correct obvious OCR artefacts (l→1, O→0, etc.).
          - Return the same keys with cleaned values.

        Parameters
        ----------
        raw_fields    : {field_name: raw_value}
        document_hint : a short description of the document type (optional)
        """
        system = (
            "You are a data-normalisation engine. "
            "Clean the field values in the JSON you receive. "
            "Rules:\n"
            "- Output ONLY a valid JSON object with the same keys.\n"
            "- Dates → ISO 8601 (YYYY-MM-DD) when possible.\n"
            "- Remove legal boilerplate like 'hereinafter referred to as', "
            "'as defined herein', parenthetical aliases like (the \"Company\").\n"
            "- Fix common OCR errors (capital O/I used as 0/1).\n"
            "- Keep null values as null. Do not invent information.\n"
            "- Do not rename keys."
        )
        user = (
            f"Document type: {document_hint or 'unknown'}\n\n"
            f"Fields to normalise:\n{json.dumps(raw_fields, indent=2)}"
        )

        result = self._call_with_retry(system, user, "normalise_fields", model=self.small_model)
        if result is None:
            logger.warning("[SchemaAgent] Normalisation failed — returning raw fields")
            return raw_fields

        # Merge: keep original for any key the LLM dropped
        merged = dict(raw_fields)
        merged.update(result)
        return merged

    # ------------------------------------------------------------------
    # 2. Schema mapping
    # ------------------------------------------------------------------

    def map_fields(
        self,
        source_fields: Dict[str, Any],
        target_schema: Dict,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Map source_fields onto the target_schema field names.

        Returns
        -------
        mapped_values : {target_field_name: value}
        field_mapping : {source_field: target_field}  (for audit)
        """
        target_field_info = {
            name: meta.get("description", name)
            for name, meta in target_schema.get("fields", {}).items()
        }

        system = (
            "You are a schema-mapping engine. "
            "Map the SOURCE fields to TARGET schema fields by semantic similarity.\n"
            "Output ONLY valid JSON with two keys:\n"
            "  \"mapping\": {source_field: target_field | null}\n"
            "  \"values\":  {target_field: mapped_value | null}\n"
            "Rules:\n"
            "- If a source field has no reasonable target match, map it to null.\n"
            "- Do not invent values. Use only the values provided in SOURCE.\n"
            "- For unmatched TARGET fields, include them in 'values' with null."
        )
        user = (
            f"SOURCE fields (name → value):\n{json.dumps(source_fields, indent=2)}\n\n"
            f"TARGET schema fields (name → description):\n"
            f"{json.dumps(target_field_info, indent=2)}"
        )

        result = self._call_with_retry(system, user, "map_fields", model=self.small_model)
        if result is None:
            logger.warning("[SchemaAgent] Mapping failed — returning empty mapping")
            empty_vals = {k: None for k in target_schema.get("fields", {})}
            return empty_vals, {}

        mapping       = result.get("mapping", {})
        mapped_values = result.get("values", {})

        # Ensure every target field is present
        for tname in target_schema.get("fields", {}):
            mapped_values.setdefault(tname, None)

        return mapped_values, mapping

    # ------------------------------------------------------------------
    # 3. Schema synthesis
    # ------------------------------------------------------------------

    def synthesise_schema(
        self,
        observed_fields: Dict[str, Any],
        document_hint: str = "",
    ) -> Dict:
        """
        Synthesise a new schema definition from the observed fields and
        their sample values.  The resulting schema can be registered in
        SchemaRegistry for future documents of the same type.

        Returns a schema dict compatible with the existing schema format.
        """
        sample_info = {
            k: str(v)[:80] if v is not None else None
            for k, v in observed_fields.items()
        }

        system = (
            "You are a schema-design assistant. "
            "Given a set of extracted field names and sample values from a document, "
            "produce a schema definition in the following JSON format:\n"
            "{\n"
            '  "form_name": "<CamelCase name>",\n'
            '  "version": "1.0",\n'
            '  "description": "<one-sentence description>",\n'
            '  "fields": {\n'
            '    "<field_name>": {\n'
            '      "type": "<date|string|number|boolean|currency|email>",\n'
            '      "description": "<clear description>",\n'
            '      "required": <true|false>,\n'
            '      "examples": ["<value>", ...]\n'
            "    },\n"
            "    ...\n"
            "  }\n"
            "}\n"
            "Rules:\n"
            "- Infer the correct type from the sample value.\n"
            "- Mark a field required only if it is clearly essential.\n"
            "- Output ONLY the JSON object — no prose."
        )
        user = (
            f"Document type hint: {document_hint or 'unknown'}\n\n"
            f"Observed fields and sample values:\n"
            f"{json.dumps(sample_info, indent=2)}"
        )

        result = self._call_with_retry(
            system,
            user,
            "synthesise_schema",
            model=self.synthesis_model,
            max_tokens=1500,
        )
        if result is None or "fields" not in result:
            logger.warning("[SchemaAgent] Synthesis failed — building fallback schema")
            return self._fallback_schema(observed_fields, document_hint)

        logger.info(f"[SchemaAgent] Synthesised schema: {result.get('form_name', '?')}")
        return result

    def repair_fields(
        self,
        fields: Dict[str, Any],
        schema: Dict,
        document_hint: str = "",
    ) -> Dict[str, Any]:
        """
        Repair low-quality or partially-invalid fields after policy fusion.
        Keeps the same keys and must not invent unsupported values.
        """
        field_info = {
            name: {
                "type": meta.get("type", "string"),
                "description": meta.get("description", name),
                "required": meta.get("required", False),
            }
            for name, meta in schema.get("fields", {}).items()
        }
        system = (
            "You are a document-extraction repair engine. "
            "Given candidate field values and the target schema, clean and repair values. "
            "Rules:\n"
            "- Output ONLY a valid JSON object with the same keys.\n"
            "- Do not invent values not supported by the candidate fields.\n"
            "- Keep unknown values as null.\n"
            "- Dates -> ISO 8601 when possible.\n"
            "- Numbers -> numeric values only.\n"
            "- Preserve semantically correct extracted values."
        )
        user = (
            f"Document type: {document_hint or schema.get('form_name', 'unknown')}\n\n"
            f"Schema:\n{json.dumps(field_info, indent=2)}\n\n"
            f"Candidate fields:\n{json.dumps(fields, indent=2)}"
        )
        result = self._call_with_retry(system, user, "repair_fields", model=self.small_model)
        if result is None:
            logger.warning("[SchemaAgent] Repair failed — returning original fields")
            return fields

        merged = dict(fields)
        merged.update(result)
        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        system: str,
        user: str,
        op_name: str,
        model: str,
        max_tokens: int = 1024,
    ) -> Optional[Dict]:
        for attempt in range(_MAX_RETRIES):
            try:
                raw = _call_groq(
                    self.client, model, system, user, max_tokens=max_tokens
                )
                parsed = _extract_json(raw)
                if parsed is not None:
                    return parsed
                # JSON parse failure — ask for repair
                logger.warning(f"[SchemaAgent] {op_name} attempt {attempt+1}: JSON parse failed, retrying")
                user = (
                    "Your previous response was not valid JSON. "
                    "Please output ONLY a valid JSON object — no explanation, no markdown.\n\n"
                    f"Original request:\n{user}"
                )
            except Exception as exc:
                logger.error(f"[SchemaAgent] {op_name} attempt {attempt+1} error: {exc}")
        return None

    @staticmethod
    def _fallback_schema(fields: Dict[str, Any], hint: str) -> Dict:
        """Minimal schema built without LLM when synthesis fails."""
        return {
            "form_name":   re.sub(r"\W+", "_", hint or "synthesised_schema").strip("_") or "Document",
            "version":     "1.0",
            "description": f"Auto-synthesised schema for {hint or 'unknown document type'}",
            "fields": {
                name: {
                    "type":        _infer_type(val),
                    "description": name.replace("_", " "),
                    "required":    False,
                }
                for name, val in fields.items()
            },
        }


def _infer_type(value: Any) -> str:
    """Guess a field type from its value."""
    if value is None:
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    s = str(value)
    if re.search(r"\d{4}-\d{2}-\d{2}", s):
        return "date"
    if re.match(r"[A-Z]{3}\s+[\d,]+", s):
        return "currency"
    if re.match(r"[\w._%+-]+@[\w.-]+\.\w{2,}", s):
        return "email"
    return "string"
