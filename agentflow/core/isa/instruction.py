"""Instruction definitions and types."""

from enum import Enum
from typing import Any, Dict, Optional

class InstructionType(str, Enum):
    """Types of instructions."""
    CONTROL = "CONTROL"
    DATA = "DATA"
    COMPUTATION = "COMPUTATION"
    IO = "IO"
    SYSTEM = "SYSTEM"

class Instruction:
    """Instruction class."""
    def __init__(self, id: str, name: str, type: str, params: Dict[str, Any], description: Optional[str] = None):
        """Initialize instruction."""
        self.id = id
        self.name = name
        self.type = type
        self.params = params
        self.description = description or ""

    def validate_params(self, input_params: Dict[str, Any]) -> bool:
        """
        Validate input parameters against parameter definitions.

        Args:
            input_params: Parameters to validate

        Returns:
            True if parameters are valid
        """
        try:
            # Check required parameters
            for param_name, param_def in self.params.items():
                if param_def.get('required', False):
                    if param_name not in input_params:
                        return False

            # Check parameter types
            for param_name, param_value in input_params.items():
                if param_name in self.params:
                    param_type = self.params[param_name].get('type')
                    if param_type and not isinstance(param_value, eval(param_type)):
                        return False

            return True

        except Exception:
            return False
