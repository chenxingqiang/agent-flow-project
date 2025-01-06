"""Workflow types and execution logic."""

from enum import Enum
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import asyncio
import networkx as nx
import uuid

from .exceptions import WorkflowExecutionError

class WorkflowStepType(Enum):
    """Types of workflow steps."""
    TRANSFORM = "transform"
    ANALYZE = "analyze"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    CUSTOM = "custom"
    RESEARCH_EXECUTION = "research_execution"
    DOCUMENT_GENERATION = "document_generation"
    CODE_GENERATION = "code_generation"
    FIGURE_GENERATION = "figure_generation"
    VIDEO_GENERATION = "video_generation"
    VOICE_GENERATION = "voice_generation"

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
        # Validate step type
        if not isinstance(self.type, WorkflowStepType):
            try:
                self.type = WorkflowStepType(self.type)
            except (ValueError, TypeError):
                raise WorkflowExecutionError(f"Invalid step type: {self.type}")
            
        # Validate strategy based on step type
        if self.type == WorkflowStepType.TRANSFORM:
            valid_strategies = {"feature_engineering", "outlier_removal", "standard"}
            if self.config.strategy not in valid_strategies:
                raise WorkflowExecutionError(
                    f"Invalid strategy for transform step: {self.config.strategy}. "
                    f"Valid strategies are: {', '.join(valid_strategies)}"
                )
        elif self.type == WorkflowStepType.ANALYZE:
            valid_strategies = {"statistical", "exploratory", "diagnostic"}
            if self.config.strategy not in valid_strategies:
                raise WorkflowExecutionError(
                    f"Invalid strategy for analyze step: {self.config.strategy}. "
                    f"Valid strategies are: {', '.join(valid_strategies)}"
                )
        elif self.type == WorkflowStepType.VALIDATE:
            valid_strategies = {"schema", "data_quality", "business_rules"}
            if self.config.strategy not in valid_strategies:
                raise WorkflowExecutionError(
                    f"Invalid strategy for validate step: {self.config.strategy}. "
                    f"Valid strategies are: {', '.join(valid_strategies)}"
                )
        elif self.type == WorkflowStepType.AGGREGATE:
            valid_strategies = {"sum", "mean", "custom"}
            if self.config.strategy not in valid_strategies:
                raise WorkflowExecutionError(
                    f"Invalid strategy for aggregate step: {self.config.strategy}. "
                    f"Valid strategies are: {', '.join(valid_strategies)}"
                )
        elif self.type == WorkflowStepType.RESEARCH_EXECUTION:
            valid_strategies = {"literature_review", "data_collection", "analysis"}
            if self.config.strategy not in valid_strategies:
                raise WorkflowExecutionError(
                    f"Invalid strategy for research execution step: {self.config.strategy}. "
                    f"Valid strategies are: {', '.join(valid_strategies)}"
                )
        elif self.type == WorkflowStepType.DOCUMENT_GENERATION:
            valid_strategies = {"markdown", "latex", "html", "pdf"}
            if self.config.strategy not in valid_strategies:
                raise WorkflowExecutionError(
                    f"Invalid strategy for document generation step: {self.config.strategy}. "
                    f"Valid strategies are: {', '.join(valid_strategies)}"
                )
        elif self.type == WorkflowStepType.CODE_GENERATION:
            valid_strategies = {"python", "javascript", "typescript", "java", "cpp", "custom"}
            if self.config.strategy not in valid_strategies:
                raise WorkflowExecutionError(
                    f"Invalid strategy for code generation step: {self.config.strategy}. "
                    f"Valid strategies are: {', '.join(valid_strategies)}"
                )
        elif self.type == WorkflowStepType.FIGURE_GENERATION:
            valid_strategies = {"matplotlib", "plotly", "seaborn", "custom"}
            if self.config.strategy not in valid_strategies:
                raise WorkflowExecutionError(
                    f"Invalid strategy for figure generation step: {self.config.strategy}. "
                    f"Valid strategies are: {', '.join(valid_strategies)}"
                )
        elif self.type == WorkflowStepType.VIDEO_GENERATION:
            valid_strategies = {"animation", "slideshow", "screen_recording", "custom"}
            if self.config.strategy not in valid_strategies:
                raise WorkflowExecutionError(
                    f"Invalid strategy for video generation step: {self.config.strategy}. "
                    f"Valid strategies are: {', '.join(valid_strategies)}"
                )
        elif self.type == WorkflowStepType.VOICE_GENERATION:
            valid_strategies = {"text_to_speech", "voice_cloning", "custom"}
            if self.config.strategy not in valid_strategies:
                raise WorkflowExecutionError(
                    f"Invalid strategy for voice generation step: {self.config.strategy}. "
                    f"Valid strategies are: {', '.join(valid_strategies)}"
                )

@dataclass
class WorkflowConfig:
    """Configuration for a workflow."""
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_iterations: int = 10
    max_retries: int = 3
    timeout: float = 300.0
    steps: List[WorkflowStep] = field(default_factory=list)
    use_ell2a: bool = False
    ell2a_mode: str = "simple"
    ell2a_config: Dict[str, Any] = field(default_factory=lambda: {
        "model": "gpt-4",
        "max_tokens": 2000,
        "temperature": 0.7,
        "tools": [],
        "stream": False,
        "simple": {
            "max_retries": 3,
            "retry_delay": 1.0,
            "timeout": 30.0
        },
        "complex": {
            "max_retries": 3,
            "retry_delay": 1.0,
            "timeout": 60.0,
            "track_performance": True,
            "track_memory": True
        }
    })
    
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
        if not self.steps:
            return []
            
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
            try:
                step.validate()
            except WorkflowExecutionError as e:
                raise WorkflowExecutionError(f"Step validation failed for {step.id}: {str(e)}")
            
        # Validate dependencies
        try:
            self._validate_dependencies()
        except WorkflowExecutionError as e:
            raise WorkflowExecutionError(f"Dependency validation failed: {str(e)}")
        
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