"""ISA Manager module."""

from typing import Dict, Any, List, Optional, Union
from ..config import AgentConfig

class Instruction:
    """Instruction class."""
    def __init__(self, id: str, name: str, type: str, params: Dict[str, Any]):
        """Initialize instruction."""
        self.id = id
        self.name = name
        self.type = type
        self.params = params

class ISAManager:
    """Manages the Instruction Set Architecture for agents."""

    def __init__(self, config: AgentConfig):
        """Initialize ISA manager."""
        self.config = config
        self.instructions = []
        self._load_instructions()

    def _load_instructions(self) -> None:
        """Load instructions from configuration."""
        if self.config.domain_config and 'instruction_set' in self.config.domain_config:
            instruction_set = self.config.domain_config['instruction_set']
            self.instructions = [
                Instruction(id=str(i), name=instr, type="default", params={})
                if isinstance(instr, str) else instr
                for i, instr in enumerate(instruction_set)
            ]

    def get_instruction(self, instruction_id: str) -> Optional[Instruction]:
        """Get instruction by ID."""
        for instruction in self.instructions:
            if instruction.id == instruction_id:
                return instruction
        return None

    def execute_instruction(self, instruction: Union[str, Instruction]) -> Any:
        """Execute an instruction."""
        if isinstance(instruction, str):
            instruction = Instruction(id=instruction, name=instruction, type="default", params={})
        if not instruction or instruction.id not in [i.id for i in self.instructions]:
            raise ValueError(f"Invalid instruction: {instruction}")
        # Execute instruction logic here
        return None 