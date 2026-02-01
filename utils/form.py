"""
Form data structure with validation support.
"""
from typing import Dict, Any, Optional, List
import json
import logging

logger = logging.getLogger(__name__)

class FormInstance:
    """Represents a structured form with field values and metadata."""
    
    def __init__(self, schema: Dict):
        """
        Initialize form from schema.
        
        Args:
            schema: Schema dictionary with form_name and fields
        """
        self.schema = schema
        self.form_name = schema.get("form_name", "Unknown")
        self.version = schema.get("version", "1.0")
        self.fields = {k: None for k in schema.get("fields", {})}
        self.metadata = {
            "extraction_errors": [],
            "validation_errors": [],
            "confidence_scores": {}
        }
    
    def fill(self, field: str, value: Any, confidence: float = 1.0):
        """
        Fill a field with a value.
        
        Args:
            field: Field name
            value: Field value
            confidence: Confidence score (0-1)
        """
        if field in self.fields:
            self.fields[field] = value
            self.metadata["confidence_scores"][field] = confidence
        else:
            logger.warning(f"Field '{field}' not in schema")
    
    def get(self, field: str, default: Any = None) -> Any:
        """Get field value with optional default."""
        return self.fields.get(field, default)
    
    def is_complete(self) -> bool:
        """Check if all required fields are filled."""
        schema_fields = self.schema.get("fields", {})
        
        for field_name, field_meta in schema_fields.items():
            if field_meta.get("required", False):
                if self.fields.get(field_name) is None:
                    return False
        
        return True
    
    def get_missing_fields(self) -> List[str]:
        """Get list of required fields that are missing."""
        missing = []
        schema_fields = self.schema.get("fields", {})
        
        for field_name, field_meta in schema_fields.items():
            if field_meta.get("required", False):
                if self.fields.get(field_name) is None:
                    missing.append(field_name)
        
        return missing
    
    def add_error(self, error_type: str, message: str):
        """Add an error message to metadata."""
        key = f"{error_type}_errors"
        if key not in self.metadata:
            self.metadata[key] = []
        self.metadata[key].append(message)
    
    def get_confidence(self, field: str) -> float:
        """Get confidence score for a field."""
        return self.metadata["confidence_scores"].get(field, 0.0)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "form": self.form_name,
            "version": self.version,
            "fields": self.fields,
            "metadata": self.metadata,
            "is_complete": self.is_complete()
        }
    
    def to_json(self, pretty: bool = False) -> str:
        """Convert to JSON string."""
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str):
        """Save form to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved form to {filepath}")
    
    @classmethod
    def from_dict(cls, data: Dict, schema: Dict) -> 'FormInstance':
        """Create FormInstance from dictionary."""
        instance = cls(schema)
        instance.fields = data.get("fields", {})
        instance.metadata = data.get("metadata", instance.metadata)
        return instance
    
    def __repr__(self):
        complete = "Complete" if self.is_complete() else "Incomplete"
        return f"FormInstance(name={self.form_name}, status={complete}, fields={len(self.fields)})"