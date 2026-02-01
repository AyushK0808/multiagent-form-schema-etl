"""
Field validation and error recovery mechanisms.
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging
from google import genai
from google.genai import types

from config.config import get_config

logger = logging.getLogger(__name__)

class ValidationRule:
    """Base class for validation rules."""
    
    def validate(self, value: Any, field_meta: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate a field value.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        raise NotImplementedError


class RequiredFieldRule(ValidationRule):
    """Validates that required fields have values."""
    
    def validate(self, value: Any, field_meta: Dict) -> Tuple[bool, Optional[str]]:
        if field_meta.get("required", False) and value is None:
            return False, f"Required field is missing"
        return True, None


class TypeValidationRule(ValidationRule):
    """Validates field type."""
    
    def validate(self, value: Any, field_meta: Dict) -> Tuple[bool, Optional[str]]:
        if value is None:
            return True, None
        
        expected_type = field_meta.get("type", "string")
        
        validators = {
            "date": self._validate_date,
            "number": self._validate_number,
            "boolean": self._validate_boolean,
            "email": self._validate_email,
            "currency": self._validate_currency,
            "string": lambda v: (True, None)
        }
        
        validator = validators.get(expected_type, lambda v: (True, None))
        return validator(value)
    
    def _validate_date(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate date format (ISO 8601)."""
        try:
            datetime.fromisoformat(str(value))
            return True, None
        except ValueError:
            return False, f"Invalid date format: {value}"
    
    def _validate_number(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate numeric value."""
        if not isinstance(value, (int, float)):
            try:
                float(value)
                return True, None
            except (ValueError, TypeError):
                return False, f"Invalid number: {value}"
        return True, None
    
    def _validate_boolean(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate boolean value."""
        if not isinstance(value, bool):
            return False, f"Invalid boolean: {value}"
        return True, None
    
    def _validate_email(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate email format."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, str(value)):
            return False, f"Invalid email format: {value}"
        return True, None
    
    def _validate_currency(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate currency format (e.g., 'USD 1000')."""
        currency_pattern = r'^[A-Z]{3}\s+[\d,]+(\.\d{2})?$'
        if not re.match(currency_pattern, str(value)):
            return False, f"Invalid currency format: {value}"
        return True, None


class ConstraintValidationRule(ValidationRule):
    """Validates field constraints (min, max, pattern, enum)."""
    
    def validate(self, value: Any, field_meta: Dict) -> Tuple[bool, Optional[str]]:
        if value is None:
            return True, None
        
        constraints = field_meta.get("constraints", {})
        
        # Min/max for numbers
        if "min" in constraints and isinstance(value, (int, float)):
            if value < constraints["min"]:
                return False, f"Value {value} below minimum {constraints['min']}"
        
        if "max" in constraints and isinstance(value, (int, float)):
            if value > constraints["max"]:
                return False, f"Value {value} above maximum {constraints['max']}"
        
        # Pattern for strings
        if "pattern" in constraints and isinstance(value, str):
            if not re.match(constraints["pattern"], value):
                return False, f"Value doesn't match pattern: {constraints['pattern']}"
        
        # Enum values
        if "enum" in constraints:
            if value not in constraints["enum"]:
                return False, f"Value not in allowed list: {constraints['enum']}"
        
        return True, None


class FieldValidator:
    """Orchestrates field validation using multiple rules."""
    
    def __init__(self):
        self.rules = [
            RequiredFieldRule(),
            TypeValidationRule(),
            ConstraintValidationRule()
        ]
    
    def validate_field(self, field_name: str, value: Any, 
                      field_meta: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single field against all rules.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for rule in self.rules:
            is_valid, error_msg = rule.validate(value, field_meta)
            if not is_valid:
                errors.append(f"{field_name}: {error_msg}")
        
        return len(errors) == 0, errors
    
    def validate_form(self, form_data: Dict, schema: Dict) -> Tuple[bool, List[str]]:
        """
        Validate all fields in a form.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        all_errors = []
        
        for field_name, field_meta in schema["fields"].items():
            value = form_data.get(field_name)
            is_valid, errors = self.validate_field(field_name, value, field_meta)
            all_errors.extend(errors)
        
        return len(all_errors) == 0, all_errors


class RecoveryStrategy:
    """Handles error recovery for failed extractions."""
    
    def recover(self, field_name: str, error: str, context: Dict) -> Optional[Any]:
        """
        Attempt to recover from extraction error.
        
        Args:
            field_name: Name of the field
            error: Error message
            context: Additional context (clause_graph, schema, etc.)
            
        Returns:
            Recovered value or None
        """
        raise NotImplementedError


class DefaultValueRecovery(RecoveryStrategy):
    """Use default value from schema."""
    
    def recover(self, field_name: str, error: str, context: Dict) -> Optional[Any]:
        schema = context.get("schema", {})
        field_meta = schema.get("fields", {}).get(field_name, {})
        default = field_meta.get("default")
        
        if default is not None:
            logger.info(f"Using default value for {field_name}: {default}")
            return default
        
        return None


class RetryRecovery(RecoveryStrategy):
    """Retry extraction with relaxed constraints."""
    
    def recover(self, field_name: str, error: str, context: Dict) -> Optional[Any]:
        # This would re-attempt extraction with different parameters
        # For now, we just log and return None
        logger.info(f"Retry recovery not implemented for {field_name}")
        return None


class ValidationRecoveryManager:
    """Manages validation and recovery strategies."""
    
    def __init__(self):
        self.validator = FieldValidator()
        config = get_config()
        logger.info(f"Initializing recovery strategies. Gemini API Key available: {bool(config.model.gemini_api_key)}")
        
        self.recovery_strategies = [
            DefaultValueRecovery(),
        ]
        
        # Try to add Gemini recovery if API key is available
        if config.model.gemini_api_key:
            try:
                gemini_recovery = GeminiVisionRecovery()
                self.recovery_strategies.append(gemini_recovery)
                logger.info("[Gemini] Successfully initialized GeminiVisionRecovery")
            except ValueError as e:
                logger.warning(f"[Gemini] Could not initialize Gemini recovery: {e}")
            except Exception as e:
                logger.error(f"[Gemini] Unexpected error initializing Gemini: {e}")
        else:
            logger.info("[Gemini] No Gemini API key available, skipping Gemini recovery")
        
        # Add Retry recovery last
        self.recovery_strategies.append(RetryRecovery())
        
        logger.info(f"Loaded {len(self.recovery_strategies)} recovery strategies: {[s.__class__.__name__ for s in self.recovery_strategies]}")
    
    def validate_and_recover(self, form_data: Dict, schema: Dict, 
                            clause_graph: Dict, page_image=None) -> Tuple[Dict, List[str]]:
        """
        Validate form data and attempt recovery for errors.
        
        Args:
            form_data: Form fields to validate
            schema: Schema definition
            clause_graph: Document structure
            page_image: PIL Image for vision-based recovery
            
        Returns:
            Tuple of (recovered_data, remaining_errors)
        """
        recovered_data = form_data.copy()
        
        # Validate
        is_valid, errors = self.validator.validate_form(recovered_data, schema)
        
        if is_valid:
            return recovered_data, []
        
        # Attempt recovery for each error
        remaining_errors = []
        context = {
            "schema": schema,
            "clause_graph": clause_graph,
            "form_data": form_data,
            "page_image": page_image  # Critical for multi-modal recovery
        }
        
        for error in errors:
            # Parse field name from error message
            field_name = error.split(":")[0].strip()
            logger.debug(f"Attempting recovery for field '{field_name}': {error}")
            
            # Try recovery strategies
            recovered = False
            for strategy in self.recovery_strategies:
                logger.debug(f"  Trying {strategy.__class__.__name__}...")
                try:
                    value = strategy.recover(field_name, error, context)
                    if value is not None:
                        recovered_data[field_name] = value
                        recovered = True
                        logger.info(f"✓ Recovered '{field_name}' = {value} using {strategy.__class__.__name__}")
                        break
                except Exception as e:
                    logger.warning(f"  {strategy.__class__.__name__} failed: {e}")
                    break
            
            if not recovered:
                remaining_errors.append(error)
        
        return recovered_data, remaining_errors
    
class GeminiVisionRecovery(RecoveryStrategy):
    """Uses Gemini API to extract missing data directly from the image."""
    
    def __init__(self):
        config = get_config()
        api_key = config.model.gemini_api_key
        if not api_key:
            raise ValueError("Gemini API key is empty")
        logger.info(f"[Gemini] Initializing with model: {config.model.gemini_model}")
        self.client = genai.Client(api_key=api_key)
        self.model_id = config.model.gemini_model
        logger.info("[Gemini] Client initialized successfully")

    def recover(self, field_name: str, error: str, context: Dict) -> Optional[Any]:
        page_image = context.get("page_image")
        schema = context.get("schema", {})
        
        if not page_image:
            logger.debug(f"[Gemini] No page_image in context, skipping recovery for {field_name}")
            return None
        
        logger.info(f"[Gemini] Recovering '{field_name}' from image...")

        field_meta = schema.get("fields", {}).get(field_name, {})
        
        # Build a targeted prompt for the missing field
        prompt = f"""Extract the value for '{field_name}' from the attached document image.
Field Description: {field_meta.get('description', 'Data field')}
Expected Type: {field_meta.get('type', 'string')}
Return ONLY a JSON object: {{"{field_name}": "value"}}"""

        try:
            logger.debug(f"[Gemini] Sending request to {self.model_id} with image type: {type(page_image)}")
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[page_image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            logger.debug(f"[Gemini] Received response for {field_name}")
            result = response.parsed  # SDK automatically parses JSON if response_mime_type is set
            extracted_value = result.get(field_name) if result else None
            
            if extracted_value:
                logger.info(f"[Gemini] ✓ Successfully extracted '{field_name}': {extracted_value}")
            else:
                logger.debug(f"[Gemini] No value extracted for '{field_name}'")
            
            return extracted_value
            
        except Exception as e:
            logger.error(f"[Gemini] Vision recovery failed for {field_name}: {e}")
            import traceback
            logger.debug(f"[Gemini] Traceback: {traceback.format_exc()}")
            return None
            
class ValidationRecoveryManager:
    def __init__(self):
        self.validator = FieldValidator()
        self.recovery_strategies = [
            DefaultValueRecovery(),
            GeminiVisionRecovery(), # Add Gemini as a high-tier recovery option
            RetryRecovery()
        ]