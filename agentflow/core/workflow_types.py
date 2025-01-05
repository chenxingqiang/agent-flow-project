"""Workflow types and execution logic."""

from enum import Enum
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import asyncio
import networkx as nx

from .exceptions import WorkflowExecutionError

class WorkflowStepType(Enum):
    """Types of workflow steps."""
    TRANSFORM = "transform"
    ANALYZE = "analyze"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    CUSTOM = "custom"
    RESEARCH_EXECUTION = "research_execution"

class WorkflowStatus(Enum):
    """Status of a workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"  # Waiting for dependencies

@dataclass
class StepConfig:
    """Configuration for a workflow step."""
    strategy: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowStep:
    """A step in a workflow."""
    id: str
    name: str
    type: WorkflowStepType
    config: StepConfig
    dependencies: List[str] = field(default_factory=list)
    
    def validate(self) -> None:
        """Validate step configuration."""
        if not isinstance(self.type, WorkflowStepType):
            raise WorkflowExecutionError(f"Invalid step type: {self.type}")
            
        # Add more validation as needed
        if self.type == WorkflowStepType.TRANSFORM:
            valid_strategies = {"feature_engineering", "outlier_removal", "standard"}
            if self.config.strategy not in valid_strategies:
                raise WorkflowExecutionError(
                    f"Invalid strategy for transform step: {self.config.strategy}"
                )

@dataclass
class WorkflowConfig:
    """Configuration for a workflow."""
    id: str
    name: str
    max_iterations: int = 10
    timeout: float = 300.0
    steps: List[WorkflowStep] = field(default_factory=list)
    
    def _validate_dependencies(self) -> None:
        """Validate step dependencies."""
        # Build dependency graph
        graph = nx.DiGraph()
        step_ids = {step.id for step in self.steps}
        
        for step in self.steps:
            graph.add_node(step.id)
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise WorkflowExecutionError(
                        f"Missing dependency '{dep}' for step '{step.id}'"
                    )
                graph.add_edge(dep, step.id)
                
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                raise WorkflowExecutionError(
                    f"Circular dependencies detected: {cycles}"
                )
        except nx.NetworkXNoCycle:
            pass
    
    def _get_execution_order(self) -> List[str]:
        """Get step execution order based on dependencies."""
        graph = nx.DiGraph()
        for step in self.steps:
            graph.add_node(step.id)
            for dep in step.dependencies:
                graph.add_edge(dep, step.id)
                
        try:
            return list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible as e:
            raise WorkflowExecutionError("Invalid step dependencies") from e
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps.
        
        Args:
            data: Input data for workflow
            
        Returns:
            Workflow execution results
            
        Raises:
            WorkflowExecutionError: If workflow execution fails
        """
        # Handle empty workflow
        if not self.steps:
            return {}
            
        # Validate steps
        for step in self.steps:
            step.validate()
            
        # Validate dependencies
        self._validate_dependencies()
        
        # Get execution order
        try:
            execution_order = self._get_execution_order()
        except WorkflowExecutionError as e:
            raise WorkflowExecutionError(f"Failed to determine execution order: {str(e)}")
            
        # Execute steps
        results = {}
        step_map = {step.id: step for step in self.steps}
        
        try:
            for step_id in execution_order:
                step = step_map[step_id]
                # Execute step (mock implementation)
                results[step_id] = {
                    "status": "success",
                    "output": data  # Just pass through data for now
                }
        except Exception as e:
            raise WorkflowExecutionError(f"Step execution failed: {str(e)}")
            
        return results