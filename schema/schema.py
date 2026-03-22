"""
Schema management for contract forms.

Schemas describe WHAT to extract (field name, type, description, examples).
HOW to extract it is entirely the LLM's job — no patterns or keywords here.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from config.config import get_config

logger = logging.getLogger(__name__)


class SchemaManager:
    """Manages form schemas for different document types."""

    def __init__(self, schema_dir: Optional[Path] = None):
        config = get_config()
        self.schema_dir = schema_dir or config.paths.schema_dir
        self._schemas: Dict[str, Dict] = {}
        self._load_schemas()

    def _load_schemas(self):
        if not self.schema_dir.exists():
            logger.warning(f"Schema directory not found: {self.schema_dir}")
            return
        for schema_file in self.schema_dir.glob("*.json"):
            try:
                with open(schema_file) as f:
                    schema = json.load(f)
                name = schema.get("form_name", schema_file.stem)
                self._schemas[name] = schema
                logger.info(f"Loaded schema: {name}")
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")

    def get_schema(self, form_name: str) -> Optional[Dict]:
        return self._schemas.get(form_name)

    def list_schemas(self) -> List[str]:
        return list(self._schemas.keys())

    def add_schema(self, schema: Dict) -> bool:
        form_name = schema.get("form_name")
        if not form_name:
            logger.error("Schema missing 'form_name' field")
            return False
        self._schemas[form_name] = schema
        schema_file = self.schema_dir / f"{form_name}.json"
        try:
            with open(schema_file, "w") as f:
                json.dump(schema, f, indent=2)
            logger.info(f"Saved schema: {form_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save schema {form_name}: {e}")
            return False


# ---------------------------------------------------------------------------
# Default schemas — lean, description-focused, no regex / keyword hints
# ---------------------------------------------------------------------------

DEFAULT_NDA_SCHEMA = {
    "form_name": "NDA_Form",
    "version": "1.0",
    "description": "Non-Disclosure Agreement extraction schema",
    "fields": {
        "effective_date": {
            "type": "date",
            "description": "The date on which the agreement becomes effective",
            "required": True,
            "examples": ["2024-01-15", "2023-12-01"],
        },
        "termination_notice": {
            "type": "string",
            "description": "The required notice period to terminate the agreement",
            "required": False,
            "examples": ["30 days", "60 days written notice"],
        },
        "governing_law": {
            "type": "string",
            "description": (
                "The jurisdiction or state whose laws govern the agreement "
                "(the actual jurisdiction name, not the heading 'Governing Law')"
            ),
            "required": True,
            "examples": ["State of California", "New York", "England and Wales"],
        },
        "disclosing_party": {
            "type": "string",
            "description": "Full legal name of the party disclosing confidential information",
            "required": True,
            "examples": ["Acme Corporation", "John Smith"],
        },
        "receiving_party": {
            "type": "string",
            "description": "Full legal name of the party receiving confidential information",
            "required": True,
            "examples": ["Beta Inc.", "Jane Doe"],
        },
        "confidentiality_period": {
            "type": "string",
            "description": "How long the confidentiality obligation lasts",
            "required": False,
            "examples": ["5 years", "indefinite"],
        },
    },
}

DEFAULT_EMPLOYMENT_SCHEMA = {
    "form_name": "Employment_Agreement",
    "version": "1.0",
    "description": "Employment agreement extraction schema",
    "fields": {
        "employee_name": {
            "type": "string",
            "description": "Full legal name of the employee",
            "required": True,
        },
        "employer_name": {
            "type": "string",
            "description": "Full legal name of the employer or company",
            "required": True,
        },
        "start_date": {
            "type": "date",
            "description": "The date employment begins",
            "required": True,
        },
        "position": {
            "type": "string",
            "description": "Job title or role of the employee",
            "required": True,
        },
        "salary": {
            "type": "currency",
            "description": "Annual salary with currency code",
            "required": True,
            "examples": ["USD 75000", "EUR 60000"],
        },
        "vacation_days": {
            "type": "number",
            "description": "Number of paid vacation days per year",
            "required": False,
            "constraints": {"min": 0, "max": 365},
        },
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_schema(form_name: str = "NDA_Form") -> Dict:
    """Load schema by name, falling back to built-in defaults."""
    manager = SchemaManager()
    schema = manager.get_schema(form_name)
    if schema:
        return schema
    defaults = {
        "NDA_Form": DEFAULT_NDA_SCHEMA,
        "Employment_Agreement": DEFAULT_EMPLOYMENT_SCHEMA,
    }
    return defaults.get(form_name, DEFAULT_NDA_SCHEMA)


def create_schema_file(form_name: str, output_dir: Optional[Path] = None) -> bool:
    """Write a built-in schema template to disk."""
    config = get_config()
    output_dir = output_dir or config.paths.schema_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    defaults = {
        "NDA_Form": DEFAULT_NDA_SCHEMA,
        "Employment_Agreement": DEFAULT_EMPLOYMENT_SCHEMA,
    }
    schema = defaults.get(form_name)
    if not schema:
        logger.error(f"No default schema for: {form_name}")
        return False
    output_path = output_dir / f"{form_name}.json"
    try:
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)
        logger.info(f"Created schema file: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create schema file: {e}")
        return False