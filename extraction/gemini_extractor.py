"""
Direct Gemini Vision-based form extraction.
Uses Gemini 2.0 Flash to extract all form fields from document image in one pass.
"""
import json
import logging
from typing import Dict, Any, Optional
from PIL import Image
from google import genai
from google.genai import types

from config.config import get_config

logger = logging.getLogger(__name__)


class GeminiDirectExtractor:
    """Direct extraction using Gemini Vision API."""
    
    def __init__(self):
        config = get_config()
        api_key = config.model.gemini_api_key
        
        if not api_key:
            raise ValueError("[Gemini] API key not configured")
        
        logger.info("[Gemini] Initializing direct extractor with gemini-2.0-flash")
        self.client = genai.Client(api_key=api_key)
        self.model_id = config.model.gemini_model
        self.config = config
    
    def extract(self, page_image: Image.Image, schema: Dict) -> Dict[str, Any]:
        """
        Extract all form fields directly from image using Gemini.
        
        Args:
            page_image: PIL Image of the document
            schema: Form schema definition
            
        Returns:
            Dictionary with extracted field values
        """
        logger.info("[Gemini] Starting direct extraction from image...")
        
        # Build schema description
        fields_description = self._build_schema_description(schema)
        
        # Build extraction prompt
        prompt = f"""You are a document extraction expert. Extract all the following fields from the attached document image.

Schema:
{fields_description}

Instructions:
1. Extract exact values from the document for each field
2. For dates, use YYYY-MM-DD format
3. For missing fields, use null
4. Return ONLY valid JSON, no explanations

Return a JSON object with all fields:"""
        
        try:
            logger.debug("[Gemini] Sending extraction request...")
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[page_image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1  # Low temperature for accurate extraction
                )
            )
            
            logger.debug("[Gemini] Received response")
            result = response.parsed
            
            if not result:
                logger.error("[Gemini] No parsed response from Gemini")
                return self._empty_result(schema)
            
            # Validate and clean result
            extracted = self._validate_extracted_data(result, schema)
            
            logger.info(f"[Gemini] Successfully extracted {sum(1 for v in extracted.values() if v is not None)} fields")
            return extracted
            
        except Exception as e:
            logger.error(f"[Gemini] Extraction failed: {e}")
            import traceback
            logger.debug(f"[Gemini] Traceback: {traceback.format_exc()}")
            return self._empty_result(schema)
    
    def _build_schema_description(self, schema: Dict) -> str:
        """Build human-readable schema description for the prompt."""
        lines = []
        for field_name, field_meta in schema.get("fields", {}).items():
            field_type = field_meta.get("type", "string")
            description = field_meta.get("description", field_name)
            required = field_meta.get("required", False)
            examples = field_meta.get("examples", [])
            
            line = f"- {field_name} ({field_type}): {description}"
            if required:
                line += " [REQUIRED]"
            if examples:
                line += f" (examples: {', '.join(examples)})"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _validate_extracted_data(self, data: Dict, schema: Dict) -> Dict[str, Any]:
        """Validate and clean extracted data against schema."""
        result = {}
        
        for field_name, field_meta in schema.get("fields", {}).items():
            value = data.get(field_name)
            field_type = field_meta.get("type", "string")
            
            # Clean and validate value
            cleaned_value = self._clean_value(value, field_type)
            result[field_name] = cleaned_value
            
            if cleaned_value is not None:
                logger.debug(f"[Gemini] {field_name}: {cleaned_value}")
            else:
                logger.debug(f"[Gemini] {field_name}: (null)")
        
        return result
    
    def _clean_value(self, value: Any, field_type: str) -> Optional[Any]:
        """Clean and validate a single value based on type."""
        if value is None or value == "null" or value == "":
            return None
        
        value_str = str(value).strip()
        
        if value_str.lower() in ["none", "null", "n/a", "na", "unknown"]:
            return None
        
        if field_type == "date":
            # Keep date as-is if it looks like a date
            import re
            if re.match(r'\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}', value_str):
                return value_str
            return None
        
        elif field_type == "number":
            try:
                return float(value_str) if '.' in value_str else int(value_str)
            except (ValueError, TypeError):
                return None
        
        elif field_type == "boolean":
            return value_str.lower() in ['true', 'yes', '1', 'agree']
        
        else:  # string
            return value_str if value_str else None
    
    def _empty_result(self, schema: Dict) -> Dict[str, Any]:
        """Create empty result with all fields set to null."""
        return {field_name: None for field_name in schema.get("fields", {}).keys()}
