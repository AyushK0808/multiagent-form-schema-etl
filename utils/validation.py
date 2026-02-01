"""
Field validation and error recovery mechanisms.
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging

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
        self.recovery_strategies = [
            DefaultValueRecovery(),
            RetryRecovery()
        ]
    
    def validate_and_recover(self, form_data: Dict, schema: Dict, 
                            clause_graph: Dict) -> Tuple[Dict, List[str]]:
        """
        Validate form data and attempt recovery for errors.
        
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
            "form_data": form_data
        }
        
        for error in errors:
            # Parse field name from error message
            field_name = error.split(":")[0].strip()
            
            # Try recovery strategies
            recovered = False
            for strategy in self.recovery_strategies:
                value = strategy.recover(field_name, error, context)
                if value is not None:
                    recovered_data[field_name] = value
                    recovered = True
                    logger.info(f"Recovered {field_name} using {strategy.__class__.__name__}")
                    break
            
            if not recovered:
                remaining_errors.append(error)
        
        return recovered_data, remaining_errors