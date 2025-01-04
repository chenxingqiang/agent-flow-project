"""Workflow type definitions."""
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from typing_extensions import Annotated
import uuid
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
import yaml

from .agent_types import AgentConfig

class StepStatus(str, Enum):
    """Step status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    """Workflow status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowStepType(str, Enum):
    """Workflow step type."""
    RESEARCH_PLANNING = "research_planning"
    RESEARCH_EXECUTION = "research_execution"
    DOCUMENT_GENERATION = "document_generation"
    DATA_COLLECTION = "data_collection"
    DATA_ANALYSIS = "data_analysis"
    RESULT_SYNTHESIS = "result_synthesis"
    VALIDATION = "validation"
    REVIEW = "review"
    CUSTOM = "custom"
    DEFAULT = "default"

class StepConfig(BaseModel):
    """Step configuration."""
    model_config = ConfigDict(frozen=True)
    
    depth: Optional[str] = Field(default="standard")
    sources: Optional[List[str]] = Field(default_factory=list)
    parallel_sources: Optional[bool] = Field(default=False)
    max_source_depth: Optional[int] = Field(default=3)
    format: Optional[str] = Field(default="standard")
    citation_style: Optional[str] = Field(default="APA")
    params: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return yaml.dump(self.model_dump())

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.params.get(key, default)

class WorkflowStep(BaseModel):
    """Workflow step."""
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    type: str = Field(...)
    config: Dict[str, Any] = Field(default_factory=dict)
    dependencies: Optional[List[str]] = Field(default_factory=list)
    enabled: bool = Field(default=True)
    retry_policy: Optional[Dict[str, Any]] = Field(default_factory=dict)
    error_handling: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timeout: Optional[int] = Field(default=3600)
    
    @field_validator('type')
    @classmethod
    def validate_step_type(cls, v: str) -> str:
        """Validate step type."""
        valid_types = [
            "research_planning", "research_execution", "document_generation",
            "data_collection", "data_analysis", "result_synthesis",
            "validation", "review", "custom"
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid step type: {v}")
        return v

class WorkflowResult(BaseModel):
    """Workflow execution result."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: WorkflowStatus = WorkflowStatus.COMPLETED
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    steps_completed: int = 0
    steps_failed: int = 0
    total_steps: int = 0
    retries: int = 0
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow'
    ) 

class WorkflowConfig(BaseModel):
    """Workflow configuration."""
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    max_iterations: int = Field(default=10, gt=0)
    timeout: int = Field(default=3600, gt=0)
    logging_level: str = Field(default="INFO")
    required_fields: List[str] = Field(default_factory=list)
    error_handling: Dict[str, Any] = Field(default_factory=dict)
    retry_policy: Optional[Dict[str, Any]] = Field(default=None)
    error_policy: Optional[Dict[str, Any]] = Field(default=None)
    is_distributed: bool = Field(default=False)
    distributed: bool = Field(default=False)  # Alias for is_distributed
    steps: List[WorkflowStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    agents: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('max_iterations')
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        """Validate max iterations."""
        if v <= 0:
            raise ValueError("max_iterations must be greater than 0")
        return v
        
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout."""
        if v <= 0:
            raise ValueError("timeout must be greater than 0")
        return v
        
    @field_validator('logging_level')
    @classmethod
    def validate_logging_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid logging level: {v}")
        return v.upper()
        
    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v: List[WorkflowStep]) -> List[WorkflowStep]:
        """Validate workflow steps."""
        if not v:
            return v
            
        # Validate step dependencies
        step_types = {step.type for step in v}
        for step in v:
            if step.dependencies:
                for dep in step.dependencies:
                    if dep not in step_types:
                        raise ValueError(f"Invalid dependency {dep} for step {step.type}")
        return v
        
    def __str__(self) -> str:
        return yaml.dump(self.model_dump())
        
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow with the given data.
        
        Args:
            data: Input data for the workflow.
            
        Returns:
            Dict[str, Any]: The result of the workflow execution, with step IDs as keys.
        """
        from .workflow_executor import WorkflowExecutor
        executor = WorkflowExecutor(self)
        result = await executor.execute(data)
        return result

# Register custom YAML constructors
def workflow_step_constructor(loader, node):
    return WorkflowStep(**loader.construct_mapping(node))

def step_config_constructor(loader, node):
    return StepConfig(**loader.construct_mapping(node))

yaml.add_constructor('tag:yaml.org,2002:python/object:agentflow.core.workflow_types.WorkflowStep', workflow_step_constructor)
yaml.add_constructor('tag:yaml.org,2002:python/object:agentflow.core.workflow_types.StepConfig', step_config_constructor)

class WorkflowMetrics(BaseModel):
    """Workflow execution metrics."""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    steps_completed: int = 0
    steps_failed: int = 0
    total_steps: int = 0
    retries: int = 0
    latency: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)