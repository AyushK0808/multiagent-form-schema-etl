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
    
    def populate(self, clause_graph: Dict[str, str], schema: Dict, full_text: str = "") -> FormInstance:
        """
        Populate form fields from clause graph.
        
        Args:
            clause_graph: Hierarchical clause structure
            schema: Field schema definition
            full_text: Optional full text of document for regex extraction
            
        Returns:
            Populated FormInstance
        """
        form = FormInstance(schema)
        
        logger.info(f"Populating form: {schema.get('form_name', 'Unknown')}")
        
        # If full_text not provided, build from clause_graph
        if not full_text and clause_graph:
            # Join all clause text
            full_text = " ".join([str(v) for v in clause_graph.values()])
        
        for field_name, field_meta in schema["fields"].items():
            value = self._extract_field(field_name, field_meta, clause_graph, full_text)
            
            # Apply validation if enabled
            if self.config.enable_validation:
                value = self._validate_field(field_name, value, field_meta)
            
            form.fill(field_name, value)
            
            logger.debug(f"Field '{field_name}': {value}")
        
        return form
    
    def _extract_field(self, field_name: str, field_meta: Dict, 
                      clause_graph: Dict[str, str], full_text: str = "") -> Any:
        """Extract a single field value."""
        
        import re
        from datetime import datetime
        
        section_key = field_meta.get("section", "")
        field_type = field_meta.get("type", "string")
        examples = field_meta.get("examples", None)
        patterns = field_meta.get("patterns", None)
        keywords = field_meta.get("keywords", [])
        
        # Use full_text for extraction if available
        search_text = full_text or self._find_context(section_key, clause_graph)
        
        # First, try regex-based extraction if patterns are provided
        if patterns and search_text:
            for pattern in (patterns if isinstance(patterns, list) else [patterns]):
                match = re.search(pattern, search_text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1) if match.groups() else match.group(0)
                    logger.info(f"Regex matched {field_name}: {value}")
                    return self._parse_value(value, field_type)
        
        # Try keyword-based extraction
        if keywords and search_text:
            for keyword in keywords:
                value = self._extract_by_keyword(field_name, keyword, search_text, field_type)
                if value is not None:
                    logger.info(f"Keyword matched {field_name}: {value}")
                    return value
        
        # Fallback to LLM-based extraction
        context_text = search_text or self._find_context(section_key, clause_graph)
        
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
    
    def _extract_by_keyword(self, field_name: str, keyword: str, context: str, field_type: str) -> Any:
        """Extract value by finding a keyword and capturing following text."""
        import re
        
        # Build pattern: keyword followed by optional punctuation/words, then capture value
        patterns = [
            rf"{keyword}\s*[:=]\s*([^\n\.;,]+)",  # "keyword: value" or "keyword=value"
            rf"{keyword}\s+([^\n\.;,]+)",  # "keyword value"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and value.lower() != "n/a":
                    logger.debug(f"{field_name} extracted by keyword '{keyword}': {value}")
                    return self._parse_value(value, field_type)
        
        return None
    
    def _parse_value(self, value: str, field_type: str) -> Any:
        """Parse and validate extracted value based on type."""
        import re
        from datetime import datetime
        
        if not value or value.lower() in ["none", "null", "n/a", ""]:
            return None
        
        value = value.strip()
        
        if field_type == "date":
            # Try multiple date formats
            date_patterns = [
                r'(\d{4}-\d{1,2}-\d{1,2})',  # YYYY-MM-DD
                r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
                r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, value, re.IGNORECASE)
                if match:
                    # For full matches, return as-is or normalize
                    matched_date = match.group(0)
                    logger.debug(f"Date pattern matched: {matched_date}")
                    return matched_date
            return None
        
        elif field_type == "number":
            # Extract first number from text
            match = re.search(r'(\d+(?:\.\d+)?)', value)
            if match:
                num_str = match.group(1)
                return float(num_str) if '.' in num_str else int(num_str)
            return None
        
        elif field_type == "boolean":
            return value.lower() in ['true', 'yes', '1', 'agree', 'agreed']
        
        else:  # string
            return value
    
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