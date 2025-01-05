"""ISA Manager Module."""

from typing import Dict, List, Any, Optional
import json
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class InstructionType(Enum):
    """Types of instructions."""
    CONTROL = "control"
    DATA = "data"
    COMPUTATION = "computation"
    IO = "io"
    VALIDATION = "validation"

@dataclass
class Instruction:
    """Formal instruction representation."""
    id: str
    name: str
    type: InstructionType
    params: Dict[str, Any]
    description: Optional[str] = None
    dependencies: Optional[List[str]] = None

class ISAManager:
    """Manages instruction set and their execution."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize ISA manager."""
        self.instructions: Dict[str, Instruction] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self._initialized = False
        
        if config_path:
            self.load_instructions(config_path)
            
    async def initialize(self):
        """Initialize ISA manager asynchronously."""
        if self._initialized:
            return self
            
        # Initialize instruction store
        self.instructions = {}
        self.execution_history = []
        
        self._initialized = True
        return self
        
    def load_instructions(self, config_path: str):
        """Load instructions from configuration file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(path) as f:
            config = json.load(f)
            
        for instr_config in config.get("instructions", []):
            instruction = Instruction(
                id=instr_config["id"],
                name=instr_config["name"],
                type=InstructionType(instr_config["type"]),
                params=instr_config.get("params", {}),
                description=instr_config.get("description"),
                dependencies=instr_config.get("dependencies", [])
            )
            self.add_instruction(instruction)
            
    def add_instruction(self, instruction: Instruction):
        """Add a new instruction."""
        if instruction.id in self.instructions:
            raise ValueError(f"Instruction with id {instruction.id} already exists")
            
        self.instructions[instruction.id] = instruction
        
    def get_instruction(self, instruction_id: str) -> Instruction:
        """Get instruction by ID.
        
        Args:
            instruction_id: ID of the instruction to get
            
        Returns:
            Instruction: The requested instruction
            
        Raises:
            ValueError: If the instruction does not exist
        """
        instruction = self.instructions.get(instruction_id)
        if instruction is None:
            raise ValueError(f"Instruction not found: {instruction_id}")
        return instruction
        
    def execute_instruction(self, instruction: Instruction) -> Dict[str, Any]:
        """Execute a single instruction.
        
        Args:
            instruction: Instruction to execute
            
        Returns:
            Execution result
            
        Raises:
            ValueError: If instruction is not found or parameters are invalid
        """
        if instruction.id not in self.instructions:
            raise ValueError(f"Unknown instruction: {instruction.id}")
            
        # Validate instruction parameters
        if instruction.type == InstructionType.CONTROL:
            if "init_param" not in instruction.params:
                raise ValueError("Control instruction requires 'init_param'")
        elif instruction.type == InstructionType.COMPUTATION:
            if "data_param" not in instruction.params:
                raise ValueError("Computation instruction requires 'data_param'")
        elif instruction.type == InstructionType.VALIDATION:
            if "threshold" not in instruction.params:
                raise ValueError("Validation instruction requires 'threshold'")
            
        # Record execution
        execution_record = {
            "instruction_id": instruction.id,
            "timestamp": datetime.now(),
            "params": instruction.params,
            "status": "completed"
        }
        self.execution_history.append(execution_record)
        
        return {"status": "success", "result": execution_record}

    def execute_instruction_with_dependencies(self, instruction: Instruction) -> Dict[str, Any]:
        """Execute an instruction and return result."""
        if instruction.id not in self.instructions:
            raise ValueError(f"Unknown instruction: {instruction.id}")
            
        # Check dependencies
        if instruction.dependencies:
            for dep_id in instruction.dependencies:
                if not self._is_dependency_satisfied(dep_id):
                    raise ValueError(f"Dependency not satisfied: {dep_id}")
                    
        # Execute instruction based on type
        result = self._execute_by_type(instruction)
        
        # Record execution
        execution_record = {
            "instruction_id": instruction.id,
            "params": instruction.params,
            "result": result
        }
        self.execution_history.append(execution_record)
        
        return result

    def _execute_by_type(self, instruction: Instruction) -> Dict[str, Any]:
        """Execute instruction based on its type."""
        try:
            if instruction.type == InstructionType.CONTROL:
                return self._execute_control_instruction(instruction)
            elif instruction.type == InstructionType.DATA:
                return self._execute_data_instruction(instruction)
            elif instruction.type == InstructionType.COMPUTATION:
                return self._execute_computation_instruction(instruction)
            elif instruction.type == InstructionType.IO:
                return self._execute_io_instruction(instruction)
            elif instruction.type == InstructionType.VALIDATION:
                return self._execute_validation_instruction(instruction)
            else:
                raise ValueError(f"Unsupported instruction type: {instruction.type}")
        except Exception as e:
            raise RuntimeError(f"Instruction execution failed: {str(e)}")

    def _execute_control_instruction(self, instruction: Instruction) -> Dict[str, Any]:
        """Execute control flow instruction."""
        params = instruction.params
        if instruction.name == "branch":
            condition = params.get("condition", False)
            if condition:
                return {"result": params.get("true_branch")}
            return {"result": params.get("false_branch")}
        elif instruction.name == "loop":
            iterations = params.get("iterations", 1)
            return {"result": f"Executed {iterations} iterations"}
        return {"result": "Control instruction executed"}

    def _execute_data_instruction(self, instruction: Instruction) -> Dict[str, Any]:
        """Execute data manipulation instruction."""
        params = instruction.params
        if instruction.name == "transform":
            data = params.get("data")
            transform_type = params.get("transform_type")
            return {"result": f"Transformed data using {transform_type}"}
        elif instruction.name == "filter":
            return {"result": "Data filtered"}
        return {"result": "Data instruction executed"}

    def _execute_computation_instruction(self, instruction: Instruction) -> Dict[str, Any]:
        """Execute computation instruction."""
        params = instruction.params
        if instruction.name == "calculate":
            operation = params.get("operation")
            return {"result": f"Calculated using {operation}"}
        return {"result": "Computation instruction executed"}

    def _execute_io_instruction(self, instruction: Instruction) -> Dict[str, Any]:
        """Execute I/O instruction."""
        params = instruction.params
        if instruction.name == "read":
            source = params.get("source")
            return {"result": f"Read from {source}"}
        elif instruction.name == "write":
            target = params.get("target")
            return {"result": f"Wrote to {target}"}
        return {"result": "I/O instruction executed"}

    def _execute_validation_instruction(self, instruction: Instruction) -> Dict[str, Any]:
        """Execute validation instruction."""
        params = instruction.params
        if instruction.name == "validate":
            rules = params.get("rules", [])
            return {"result": f"Validated against {len(rules)} rules"}
        return {"result": "Validation instruction executed"}

    def _is_dependency_satisfied(self, dep_id: str) -> bool:
        """Check if a dependency has been executed."""
        return any(record["instruction_id"] == dep_id for record in self.execution_history)

    def get_metrics(self) -> Dict[str, Any]:
        """Get ISA manager metrics.
        
        Returns:
            Metrics information
        """
        return {
            "total_instructions": len(self.instructions),
            "successful_executions": sum(1 for record in self.execution_history if record["status"] == "completed"),
            "failed_executions": len(self.execution_history) - sum(1 for record in self.execution_history if record["status"] == "completed")
        }
        
    async def cleanup(self) -> None:
        """Clean up ISA manager resources."""
        if not self._initialized:
            return
            
        # Clean up instructions
        for instruction in self.instructions.values():
            if hasattr(instruction, "cleanup") and callable(instruction.cleanup):
                await instruction.cleanup()
                
        self.instructions.clear()
        self.execution_history = []
        self._initialized = False
