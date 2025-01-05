"""Workflow types and execution logic."""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class WorkflowStepType(str, Enum):
    """Types of workflow steps."""
    TRANSFORM = "transform"
    ANALYZE = "analyze"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    CUSTOM = "custom"

class StepConfig(BaseModel):
    """Configuration for a workflow step."""
    strategy: str
    params: Dict[str, Any] = Field(default_factory=dict)

class WorkflowStep(BaseModel):
    """A step in a workflow."""
    id: str
    name: str
    type: WorkflowStepType
    config: StepConfig
    description: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class WorkflowConfig(BaseModel):
    """Configuration for a workflow."""
    id: Optional[str] = None
    name: str
    max_iterations: int = Field(default=5)
    timeout: int = Field(default=3600)
    steps: List[WorkflowStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)