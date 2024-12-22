"""ISA (Instruction Set Architecture) Manager for Agent Instructions"""

from typing import Dict, List, Any, Optional, Set
import json
from pathlib import Path
import logging
import ray
from dataclasses import dataclass
from enum import Enum
import networkx as nx

logger = logging.getLogger(__name__)

class InstructionType(Enum):
    """Types of instructions supported by the ISA"""
    BASIC = "basic"           # Single agent instruction
    COMPOSITE = "composite"   # Combination of multiple instructions
    PARALLEL = "parallel"     # Instructions that can run in parallel
    OPTIMIZED = "optimized"   # Optimized version of other instructions

@dataclass
class Instruction:
    """Represents a single instruction in the ISA"""
    name: str
    type: InstructionType
    description: str
    dependencies: List[str]
    cost: float  # Computational cost estimate
    parallelizable: bool
    agent_requirements: List[str]  # Required agent capabilities

@ray.remote
class ParallelInstruction:
    """Ray actor for parallel instruction execution"""
    def __init__(self, instruction: Instruction):
        self.instruction = instruction
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the instruction in parallel"""
        # Implementation will be provided by specific instruction types
        raise NotImplementedError

class ISAManager:
    """Manages the Instruction Set Architecture for agents"""
    
    def __init__(self, isa_config_path: Optional[str] = None):
        self.instructions: Dict[str, Instruction] = {}
        self.dependency_graph = nx.DiGraph()
        self.execution_history = []
        self._initialized = False
        self._config_path = isa_config_path
    
    async def initialize(self):
        """Initialize the ISA manager"""
        if self._initialized:
            return self
            
        # Initialize Ray if not already initialized
        try:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
        except Exception as e:
            logger.warning(f"Failed to initialize Ray: {e}. Some parallel features may be unavailable.")
        
        if self._config_path:
            self.load_instructions(self._config_path)
            
        self._initialized = True
        return self
    
    def load_instructions(self, config_path: str):
        """Load instructions from configuration file"""
        try:
            with open(config_path) as f:
                config = json.load(f)
                
            for instr_config in config.get("instructions", []):
                instruction = Instruction(
                    name=instr_config["name"],
                    type=InstructionType[instr_config["type"].upper()],
                    description=instr_config["description"],
                    dependencies=instr_config.get("dependencies", []),
                    cost=instr_config.get("cost", 1.0),
                    parallelizable=instr_config.get("parallelizable", False),
                    agent_requirements=instr_config.get("agent_requirements", [])
                )
                self.register_instruction(instruction)
        except Exception as e:
            logger.error(f"Failed to load ISA configuration: {e}")
    
    def register_instruction(self, instruction: Instruction):
        """Register a new instruction and update dependency graph"""
        self.instructions[instruction.name] = instruction
        
        # Update dependency graph
        self.dependency_graph.add_node(instruction.name)
        for dep in instruction.dependencies:
            self.dependency_graph.add_edge(dep, instruction.name)
    
    def optimize_instruction_sequence(self, instructions: List[str]) -> List[List[str]]:
        """Optimize a sequence of instructions for parallel execution"""
        if not instructions:
            return []
            
        # Build execution levels based on dependencies
        levels: List[Set[str]] = []
        remaining = set(instructions)
        
        while remaining:
            # Find instructions with no remaining dependencies
            level = {
                instr for instr in remaining
                if not any(dep in remaining for dep in self.instructions[instr].dependencies)
            }
            
            if not level:
                # Circular dependency detected
                logger.warning("Circular dependency detected in instruction sequence")
                break
                
            levels.append(level)
            remaining -= level
        
        # Group parallelizable instructions within each level
        optimized_sequence = []
        for level in levels:
            parallel_group = []
            sequential_group = []
            
            for instr in level:
                if self.instructions[instr].parallelizable:
                    parallel_group.append(instr)
                else:
                    sequential_group.append(instr)
            
            if parallel_group:
                optimized_sequence.append(parallel_group)
            if sequential_group:
                optimized_sequence.extend([[instr] for instr in sequential_group])
        
        return optimized_sequence
    
    async def execute_instruction_sequence(self, sequence: List[List[str]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute an optimized instruction sequence"""
        results = []
        
        for group in sequence:
            if len(group) > 1:  # Parallel execution
                # Create Ray actors for parallel execution
                actors = [
                    ParallelInstruction.remote(self.instructions[instr])
                    for instr in group
                ]
                
                # Execute instructions in parallel
                futures = [actor.execute.remote(context) for actor in actors]
                group_results = await ray.get(futures)
                results.extend(group_results)
                
                # Update execution history
                for instr, result in zip(group, group_results):
                    self.execution_history.append({
                        "instruction": instr,
                        "type": "parallel",
                        "status": "completed",
                        "result": result
                    })
            else:  # Sequential execution
                instr = group[0]
                try:
                    # Execute single instruction
                    result = await self._execute_single_instruction(instr, context)
                    results.append(result)
                    
                    self.execution_history.append({
                        "instruction": instr,
                        "type": "sequential",
                        "status": "completed",
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Failed to execute instruction {instr}: {e}")
                    self.execution_history.append({
                        "instruction": instr,
                        "type": "sequential",
                        "status": "failed",
                        "error": str(e)
                    })
                    raise
        
        return results
    
    async def _execute_single_instruction(self, instruction_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single instruction"""
        instruction = self.instructions[instruction_name]
        # Implementation will be provided by specific instruction types
        raise NotImplementedError
    
    async def cleanup(self):
        """Clean up ISA manager state"""
        # Clear instructions
        self.instructions = {}
        
        # Clear dependency graph
        self.dependency_graph.clear()
        
        # Clear history
        self.execution_history = []
        
        # Reset initialization flag
        self._initialized = False
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history with tracing information"""
        return self.execution_history
    
    def get_instruction_stats(self) -> Dict[str, Any]:
        """Get statistics about instruction execution"""
        stats = {
            "total_executions": len(self.execution_history),
            "successful_executions": sum(1 for h in self.execution_history if h["status"] == "completed"),
            "failed_executions": sum(1 for h in self.execution_history if h["status"] == "failed"),
            "instruction_usage": {},
            "average_parallel_group_size": 0
        }
        
        # Calculate instruction usage
        for history in self.execution_history:
            instr = history["instruction"]
            stats["instruction_usage"][instr] = stats["instruction_usage"].get(instr, 0) + 1
        
        # Calculate average parallel group size
        parallel_groups = [h for h in self.execution_history if h["type"] == "parallel"]
        if parallel_groups:
            stats["average_parallel_group_size"] = len(parallel_groups) / len(set(h["instruction"] for h in parallel_groups))
        
        return stats
