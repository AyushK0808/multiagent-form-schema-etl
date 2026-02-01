"""
Form population with schema-guided extraction and validation.
"""
from typing import Dict, Any, List, Optional
import logging
from utils.form import FormInstance
from extraction.extraction import LLMExtractor
from config.config import get_config

logger = logging.getLogger(__name__)

class FormFiller:
    """Handles form population from clause graph using schema."""
    
    def __init__(self):
        self.extractor = LLMExtractor()
        self.config = get_config()
    
    def populate(self, clause_graph: Dict[str, str], schema: Dict) -> FormInstance:
        """
        Populate form fields from clause graph.
        
        Args:
            clause_graph: Hierarchical clause structure
            schema: Field schema definition
            
        Returns:
            Populated FormInstance
        """
        form = FormInstance(schema)
        
        logger.info(f"Populating form: {schema.get('form_name', 'Unknown')}")
        
        for field_name, field_meta in schema["fields"].items():
            value = self._extract_field(field_name, field_meta, clause_graph)
            
            # Apply validation if enabled
            if self.config.enable_validation:
                value = self._validate_field(field_name, value, field_meta)
            
            form.fill(field_name, value)
            
            logger.debug(f"Field '{field_name}': {value}")
        
        return form
    
    def _extract_field(self, field_name: str, field_meta: Dict, 
                      clause_graph: Dict[str, str]) -> Any:
        """Extract a single field value."""
        
        # Get target section(s)
        section_key = field_meta.get("section", "")
        field_type = field_meta.get("type", "string")
        examples = field_meta.get("examples", None)
        
        # Find matching sections
        context_text = self._find_context(section_key, clause_graph)
        
        if not context_text:
            logger.warning(f"No context found for field '{field_name}' in section '{section_key}'")
            return None
        
        # Extract using LLM
        return self.extractor.extract_field(
            field_name=field_name,
            field_type=field_type,
            context_text=context_text,
            examples=examples
        )
    
    def _find_context(self, section_key: str, clause_graph: Dict[str, str]) -> str:
        """Find relevant context text from clause graph."""
        
        # Direct match
        if section_key in clause_graph:
            return clause_graph[section_key]
        
        # Fuzzy match - find sections containing the key
        matching_sections = []
        for key, text in clause_graph.items():
            if section_key.lower() in key.lower() or section_key.lower() in text.lower():
                matching_sections.append(text)
        
        # Combine matching sections
        return " ".join(matching_sections) if matching_sections else ""
    
    def _validate_field(self, field_name: str, value: Any, 
                       field_meta: Dict) -> Any:
        """Validate extracted field value."""
        
        # Check required fields
        if field_meta.get("required", False) and value is None:
            logger.warning(f"Required field '{field_name}' is missing")
        
        # Check value constraints
        constraints = field_meta.get("constraints", {})
        
        if value is not None and constraints:
            # Min/max for numbers
            if "min" in constraints and isinstance(value, (int, float)):
                if value < constraints["min"]:
                    logger.warning(f"Field '{field_name}' below minimum: {value} < {constraints['min']}")
            
            if "max" in constraints and isinstance(value, (int, float)):
                if value > constraints["max"]:
                    logger.warning(f"Field '{field_name}' above maximum: {value} > {constraints['max']}")
            
            # Regex pattern for strings
            if "pattern" in constraints and isinstance(value, str):
                import re
                if not re.match(constraints["pattern"], value):
                    logger.warning(f"Field '{field_name}' doesn't match pattern: {constraints['pattern']}")
            
            # Enum values
            if "enum" in constraints:
                if value not in constraints["enum"]:
                    logger.warning(f"Field '{field_name}' not in allowed values: {constraints['enum']}")
        
        return value


def populate_form(clause_graph: Dict[str, str], schema: Dict) -> FormInstance:
    """
    Convenience function for form population.
    
    Args:
        clause_graph: Hierarchical clause structure
        schema: Field schema definition
        
    Returns:
        Populated FormInstance
    """
    filler = FormFiller()
    return filler.populate(clause_graph, schema)