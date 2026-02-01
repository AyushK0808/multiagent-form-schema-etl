"""
Schema management for contract forms.
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
        self._schemas = {}
        self._load_schemas()
    
    def _load_schemas(self):
        """Load all schemas from schema directory."""
        if not self.schema_dir.exists():
            logger.warning(f"Schema directory not found: {self.schema_dir}")
            return
        
        for schema_file in self.schema_dir.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                    schema_name = schema.get("form_name", schema_file.stem)
                    self._schemas[schema_name] = schema
                    logger.info(f"Loaded schema: {schema_name}")
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")
    
    def get_schema(self, form_name: str) -> Optional[Dict]:
        """Get schema by form name."""
        return self._schemas.get(form_name)
    
    def list_schemas(self) -> List[str]:
        """List available schema names."""
        return list(self._schemas.keys())
    
    def add_schema(self, schema: Dict) -> bool:
        """Add or update a schema."""
        form_name = schema.get("form_name")
        if not form_name:
            logger.error("Schema missing 'form_name' field")
            return False
        
        self._schemas[form_name] = schema
        
        # Optionally save to disk
        schema_file = self.schema_dir / f"{form_name}.json"
        try:
            with open(schema_file, 'w') as f:
                json.dump(schema, f, indent=2)
            logger.info(f"Saved schema: {form_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save schema {form_name}: {e}")
            return False


# Default schemas
DEFAULT_NDA_SCHEMA = {
    "form_name": "NDA_Form",
    "version": "1.0",
    "description": "Non-Disclosure Agreement extraction schema",
    "fields": {
        "effective_date": {
            "type": "date",
            "description": "Date when the agreement becomes effective",
            "section": "Effective",
            "required": True,
            "examples": ["2024-01-15", "2023-12-01"]
        },
        "termination_notice": {
            "type": "string",
            "description": "Notice period for termination",
            "section": "Termination",
            "required": False,
            "examples": ["30 days", "60 days written notice"]
        },
        "governing_law": {
            "type": "string",
            "description": "Jurisdiction whose laws govern the agreement",
            "section": "Governing",
            "required": True,
            "examples": ["State of California", "New York"]
        },
        "disclosing_party": {
            "type": "string",
            "description": "Party disclosing confidential information",
            "section": "Parties",
            "required": True
        },
        "receiving_party": {
            "type": "string",
            "description": "Party receiving confidential information",
            "section": "Parties",
            "required": True
        },
        "confidentiality_period": {
            "type": "string",
            "description": "Duration of confidentiality obligation",
            "section": "Confidentiality",
            "required": False,
            "examples": ["5 years", "indefinite"]
        }
    }
}

DEFAULT_EMPLOYMENT_SCHEMA = {
    "form_name": "Employment_Agreement",
    "version": "1.0",
    "description": "Employment agreement extraction schema",
    "fields": {
        "employee_name": {
            "type": "string",
            "section": "Parties",
            "required": True
        },
        "employer_name": {
            "type": "string",
            "section": "Parties",
            "required": True
        },
        "start_date": {
            "type": "date",
            "section": "Employment Period",
            "required": True
        },
        "position": {
            "type": "string",
            "section": "Position",
            "required": True
        },
        "salary": {
            "type": "currency",
            "section": "Compensation",
            "required": True,
            "examples": ["USD 75000", "EUR 60000"]
        },
        "vacation_days": {
            "type": "number",
            "section": "Benefits",
            "required": False,
            "constraints": {"min": 0, "max": 365}
        }
    }
}


def load_schema(form_name: str = "NDA_Form") -> Dict:
    """
    Load schema by name or return default.
    
    Args:
        form_name: Name of the form schema
        
    Returns:
        Schema dictionary
    """
    manager = SchemaManager()
    schema = manager.get_schema(form_name)
    
    if schema:
        return schema
    
    # Return default schemas
    defaults = {
        "NDA_Form": DEFAULT_NDA_SCHEMA,
        "Employment_Agreement": DEFAULT_EMPLOYMENT_SCHEMA
    }
    
    return defaults.get(form_name, DEFAULT_NDA_SCHEMA)


def create_schema_file(form_name: str, output_dir: Optional[Path] = None):
    """Create a schema file from default templates."""
    config = get_config()
    output_dir = output_dir or config.paths.schema_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    defaults = {
        "NDA_Form": DEFAULT_NDA_SCHEMA,
        "Employment_Agreement": DEFAULT_EMPLOYMENT_SCHEMA
    }
    
    schema = defaults.get(form_name)
    if not schema:
        logger.error(f"No default schema for: {form_name}")
        return False
    
    output_path = output_dir / f"{form_name}.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)
        logger.info(f"Created schema file: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create schema file: {e}")
        return False