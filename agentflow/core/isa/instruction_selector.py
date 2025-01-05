"""Instruction selection and optimization module."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .instruction import Instruction
from .isa_manager import Instruction

class InstructionSelector:
    """Selects and optimizes instruction sequences."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize instruction selector."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def select_instructions(self, input_data: Dict[str, Any], available_instructions: Dict[str, Instruction]) -> List[Instruction]:
        """
        Select appropriate instructions based on input data.

        Args:
            input_data: Input data to process
            available_instructions: Dictionary of available instructions

        Returns:
            Selected instruction sequence
        """
        try:
            # Extract task and parameters
            task = input_data.get('task', '')
            text = input_data.get('text', '')
            params = input_data.get('params', {})

            # Score each instruction
            scored_instructions = []
            for instruction in available_instructions.values():
                score = self._score_instruction(instruction, task, text, params)
                scored_instructions.append((score, instruction))

            # Sort by score and select top instructions
            scored_instructions.sort(reverse=True)
            selected = [instr for _, instr in scored_instructions[:3]]

            return selected

        except Exception as e:
            self.logger.error(f"Instruction selection failed: {str(e)}")
            return []

    def _score_instruction(self, instruction: Instruction, task: str, text: str, params: Dict[str, Any]) -> float:
        """Score an instruction based on task, text and parameters."""
        try:
            # Base score
            score = 0.0

            # Match instruction name with task and text
            name = instruction.name.lower()
            if name in task.lower():
                score += 1.0
            if name in text.lower():
                score += 0.5

            # Check parameter compatibility
            required_params = set(instruction.params.keys())
            provided_params = set(params.keys())
            param_match = len(required_params & provided_params) / max(len(required_params), 1)
            score += param_match

            # Add random exploration factor
            score += np.random.normal(0, 0.1)

            return score

        except Exception as e:
            self.logger.error(f"Instruction scoring failed: {str(e)}")
            return 0.0
