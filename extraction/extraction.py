"""
Schema-guided LLM extraction with JSON-constrained output.
"""
from transformers import pipeline
import json
import re
import logging
from typing import Optional, Dict, Any
from config.config import get_config

logger = logging.getLogger(__name__)

class LLMExtractor:
    """Handles LLM-based field extraction with structured output."""
    
    def __init__(self, model_name: str = None, temperature: float = None, 
                 max_tokens: int = None):
        config = get_config()
        
        self.model_name = model_name or config.model.llm_model
        self.temperature = temperature or config.model.llm_temperature
        self.max_tokens = max_tokens or config.model.llm_max_tokens
        
        logger.info(f"Loading LLM: {self.model_name}")
        
        self.llm = pipeline(
            "text-generation",
            model=self.model_name,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            device_map=config.model.device
        )
    
    def extract_field(self, field_name: str, field_type: str, 
                     context_text: str, examples: Optional[list] = None) -> Any:
        """
        Extract a single field value from context text.
        
        Args:
            field_name: Name of the field to extract
            field_type: Expected type (date, string, number, boolean)
            context_text: Text to extract from
            examples: Optional list of example values
            
        Returns:
            Extracted value or None
        """
        if not context_text or not context_text.strip():
            logger.debug(f"Empty context for field: {field_name}")
            return None
        
        # Build type-specific prompt
        prompt = self._build_prompt(field_name, field_type, context_text, examples)
        
        # Generate response
        try:
            outputs = self.llm(prompt)
            response_text = outputs[0]["generated_text"]
            
            # Extract JSON from response
            extracted_value = self._parse_response(response_text, field_name, field_type)
            
            logger.debug(f"Extracted {field_name}: {extracted_value}")
            return extracted_value
            
        except Exception as e:
            logger.error(f"Extraction failed for {field_name}: {e}")
            return None
    
    def _build_prompt(self, field_name: str, field_type: str, 
                     context_text: str, examples: Optional[list] = None) -> str:
        """Build type-specific extraction prompt."""
        
        # Type-specific instructions
        type_instructions = {
            "date": "Extract the date in ISO format (YYYY-MM-DD). If month/day are missing, use 01.",
            "string": "Extract the exact text value.",
            "number": "Extract only the numeric value without units or formatting.",
            "boolean": "Extract as true or false.",
            "email": "Extract the email address.",
            "currency": "Extract the amount with currency code (e.g., 'USD 1000')."
        }
        
        instruction = type_instructions.get(field_type, "Extract the value.")
        
        # Add examples if provided
        example_text = ""
        if examples:
            example_text = "\n\nExamples of valid values:\n" + "\n".join(f"- {ex}" for ex in examples)
        
        # TinyLlama-specific format
        prompt = f"""<|system|>
You are a precise data extraction API. Output ONLY valid JSON.
{instruction}
If the value is not found in the text, return null.
{example_text}
</s>
<|user|>
Extract the value for "{field_name}" from this text:

TEXT:
\"\"\"{context_text[:1000]}\"\"\"

Output format (JSON only):
{{ "{field_name}": "<extracted_value>" }}
</s>
<|assistant|>
"""
        return prompt
    
    def _parse_response(self, response_text: str, field_name: str, 
                       field_type: str) -> Any:
        """Parse and validate extracted value from LLM response."""
        
        # Remove prompt from response
        if "<|assistant|>" in response_text:
            response_text = response_text.split("<|assistant|>")[-1]
        
        # Extract JSON
        json_data = self._extract_json(response_text)
        
        if json_data and field_name in json_data:
            value = json_data[field_name]
            
            # Type conversion and validation
            return self._validate_type(value, field_type)
        
        return None
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON object from text."""
        try:
            # Remove markdown code blocks
            text = re.sub(r'```json\s*|\s*```', '', text)
            
            # Find JSON object
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.debug(f"JSON extraction failed: {e}")
        
        return None
    
    def _validate_type(self, value: Any, expected_type: str) -> Any:
        """Validate and convert value to expected type."""
        if value is None or value == "null":
            return None
        
        try:
            if expected_type == "date":
                # Basic date validation (YYYY-MM-DD format)
                if isinstance(value, str) and re.match(r'\d{4}-\d{2}-\d{2}', value):
                    return value
                return None
            
            elif expected_type == "number":
                return float(value) if '.' in str(value) else int(value)
            
            elif expected_type == "boolean":
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ['true', '1', 'yes']
            
            elif expected_type in ["string", "email", "currency"]:
                return str(value).strip()
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Type validation failed: {e}")
        
        return value


def extract_field_value(field_name: str, text: str, field_type: str = "string", 
                       examples: Optional[list] = None) -> Any:
    """
    Convenience function for field extraction.
    
    Args:
        field_name: Name of field to extract
        text: Context text
        field_type: Expected type
        examples: Optional examples
        
    Returns:
        Extracted value
    """
    extractor = LLMExtractor()
    return extractor.extract_field(field_name, field_type, text, examples)