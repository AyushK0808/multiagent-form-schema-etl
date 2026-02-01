"""
Direct Llama 3.2 Vision-based form extraction using local Ollama.
Requires: ollama pull llama2-vision
"""
import json
import logging
from typing import Dict, Any, Optional
from PIL import Image
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class LlamaDirectExtractor:
    """Direct extraction using Llama 3.2 Vision via local Ollama."""
    
    def __init__(self, model: str = "llama3.2-vision"):
        """
        Initialize Llama extractor with local Ollama.
        
        Args:
            model: Ollama model name (default: llama3.2-vision)
        """
        self._init_ollama(model)
    
    def _init_ollama(self, model: str):
        """Initialize Ollama backend."""
        try:
            import ollama
            self.client = ollama
            self.model = model
            logger.info(f"[Llama/Ollama] Using model: {model}")
            # Test connection
            logger.info("[Llama/Ollama] Testing connection to Ollama...")
            response = ollama.list()
            available_models = [m.model for m in response.models]
            logger.info(f"[Llama/Ollama] Available models: {available_models}")
            
            if model not in available_models:
                logger.warning(f"[Llama/Ollama] Model '{model}' not found. Available: {available_models}")
                logger.info(f"[Llama/Ollama] Pull the model with: ollama pull {model}")
        except ImportError:
            raise ValueError("[Llama] Ollama SDK not installed. Install with: pip install ollama")
        except Exception as e:
            raise ValueError(f"[Llama] Ollama connection failed. Ensure Ollama is running: {e}")
    
    def extract(self, page_image: Image.Image, schema: Dict) -> Dict[str, Any]:
        """
        Extract all form fields directly from image using Llama.
        
        Args:
            page_image: PIL Image of the document
            schema: Form schema definition
            
        Returns:
            Dictionary with extracted field values
        """
        logger.info("[Llama] Starting direct extraction from image...")
        
        # Build schema description
        fields_description = self._build_schema_description(schema)
        
        # Build extraction prompt
        prompt = f"""You are a document extraction expert. Extract all the following fields from the attached document image.

Schema:
{fields_description}

Instructions:
1. Extract exact values from the document for each field
2. For dates, use YYYY-MM-DD format if possible
3. For missing fields, use null
4. Return ONLY valid JSON, no explanations
5. Be precise and accurate

Return a JSON object with all fields:"""
        
        try:
            logger.debug("[Llama] Converting image to base64...")
            # Convert image to base64
            buffered = BytesIO()
            page_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            logger.debug("[Llama] Sending extraction request to Ollama...")
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                images=[img_base64],
                stream=False,
                format="json"
            )
            
            response_text = response.get("response", "")
            logger.debug(f"[Llama] Raw response: {response_text[:300]}...")
            
            # Parse JSON from response
            extracted = self._parse_json_response(response_text, schema)
            logger.info(f"[Llama] Successfully extracted {sum(1 for v in extracted.values() if v is not None)} fields")
            return extracted
            
        except Exception as e:
            logger.error(f"[Llama] Extraction failed: {e}")
            import traceback
            logger.debug(f"[Llama] Traceback: {traceback.format_exc()}")
            return self._empty_result(schema)
    
    def _parse_json_response(self, response_text: str, schema: Dict) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        import re
        
        # Try to extract JSON from response
        try:
            # Look for JSON object in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
            else:
                logger.warning("[Llama] No JSON found in response")
                return self._empty_result(schema)
        except json.JSONDecodeError as e:
            logger.error(f"[Llama] JSON parse error: {e}")
            return self._empty_result(schema)
        
        # Validate against schema
        result = {}
        for field_name, field_meta in schema.get("fields", {}).items():
            value = data.get(field_name)
            field_type = field_meta.get("type", "string")
            
            cleaned_value = self._clean_value(value, field_type)
            result[field_name] = cleaned_value
            
            if cleaned_value is not None:
                logger.debug(f"[Llama] {field_name}: {cleaned_value}")
            else:
                logger.debug(f"[Llama] {field_name}: (null)")
        
        return result
    
    def _clean_value(self, value: Any, field_type: str) -> Optional[Any]:
        """Clean and validate a single value based on type."""
        if value is None or value == "null" or value == "":
            return None
        
        value_str = str(value).strip()
        
        if value_str.lower() in ["none", "null", "n/a", "na", "unknown"]:
            return None
        
        if field_type == "date":
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
    
    def _empty_result(self, schema: Dict) -> Dict[str, Any]:
        """Create empty result with all fields set to null."""
        return {field_name: None for field_name in schema.get("fields", {}).keys()}

