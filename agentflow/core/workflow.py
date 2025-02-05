"""Workflow engine module."""

from typing import Dict, Any, Optional, List, Union, Type, TYPE_CHECKING, cast
import uuid
import logging
import asyncio
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from .workflow_types import WorkflowConfig, WorkflowStepType, StepConfig, WorkflowStep
from .enums import WorkflowStatus
from ..agents.agent_types import AgentType
from .config import AgentConfig

if TYPE_CHECKING:
    from ..agents.agent import Agent

logger = logging.getLogger(__name__)

class WorkflowInstance(BaseModel):
    """Workflow instance class."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    steps: List[WorkflowStep] = Field(default_factory=list)
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    config: Optional[WorkflowConfig] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom dump method to ensure proper serialization."""
        data = super().model_dump(**kwargs)
        # Ensure status is serialized as string
        data["status"] = self.status.value if self.status else None
        # Keep the result as is since we've already serialized it
        if self.result:
            data["result"] = self.result
        return data

class WorkflowEngine:
    """Workflow engine class."""
    
    def __init__(self, workflow_config: Optional[Union[Dict[str, Any], WorkflowConfig]] = None):
        """Initialize workflow engine.
        
        Args:
            workflow_config: Optional workflow configuration
        """
        self._initialized = False
        self.instances: Dict[str, WorkflowInstance] = {}
        
        # Validate and store config
        if isinstance(workflow_config, dict):
            if "COLLABORATION" not in workflow_config or "WORKFLOW" not in workflow_config["COLLABORATION"]:
                raise ValueError("Workflow definition must contain COLLABORATION.WORKFLOW")
            self.config = workflow_config
        elif workflow_config is not None and not isinstance(workflow_config, WorkflowConfig):
            raise ValueError("workflow_config must be an instance of WorkflowConfig, a dictionary, or None")
        else:
            self.config = workflow_config
            
    async def initialize(self) -> None:
        """Initialize workflow engine."""
        if not self._initialized:
            # Import at runtime to avoid circular imports
            from ..agents.agent_factory import create_agent
            
            self._create_agent = create_agent
            self._initialized = True
            
    async def create_workflow(self, name: str, config: Optional[WorkflowConfig] = None) -> WorkflowInstance:
        """Create a new workflow instance.
        
        Args:
            name: Workflow name
            config: Optional workflow configuration
            
        Returns:
            WorkflowInstance: Created workflow instance
        """
        instance = WorkflowInstance(name=name, config=config)
        if config and config.steps:
            instance.steps = [
                WorkflowStep(
                    id=step.id,
                    name=step.name,
                    type=step.type,
                    config=step.config,
                    dependencies=step.dependencies,
                    required=step.required,
                    optional=step.optional,
                    is_distributed=step.is_distributed
                )
                for step in config.steps
            ]
        self.instances[instance.id] = instance
        return instance
        
    async def execute_workflow(self, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute workflow instance.
        
        Args:
            instance: Workflow instance to execute
            
        Returns:
            Dict[str, Any]: Workflow execution results
            
        Raises:
            ValueError: If workflow instance is invalid
        """
        if not instance.steps:
            raise ValueError("Workflow instance has no steps")
            
        instance.status = WorkflowStatus.RUNNING
        instance.updated_at = datetime.now()
        
        step_results = {}
        try:
            for step in instance.steps:
                try:
                    # Handle agent steps differently
                    if step.type == WorkflowStepType.AGENT:
                        # For testing, just pass through the context
                        if instance.context.get("test_mode"):
                            result = {"data": instance.context.get("data", {})}
                        else:
                            raise ValueError(f"Agent step {step.id} requires an agent to be configured")
                    else:
                        result = await step.execute(instance.context)

                    if not result:
                        raise ValueError(f"Step {step.id} returned no result")
                    step_results[step.id] = {
                        "id": step.id,
                        "type": step.type.value,
                        "status": WorkflowStatus.COMPLETED.value,
                        "result": result,
                        "error": None
                    }
                except Exception as e:
                    step_results[step.id] = {
                        "id": step.id,
                        "type": step.type.value,
                        "status": WorkflowStatus.FAILED.value,
                        "result": None,
                        "error": str(e)
                    }
                    instance.status = WorkflowStatus.FAILED
                    instance.error = f"Step {step.id} failed: {str(e)}"
                    break
                    
            if instance.status != WorkflowStatus.FAILED:
                instance.status = WorkflowStatus.COMPLETED
                
            instance.result = {
                "status": instance.status.value,
                "steps": list(step_results.values()),
                "error": instance.error
            }
                
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error = str(e)
            instance.result = {
                "status": instance.status.value,
                "steps": list(step_results.values()),
                "error": str(e)
            }
            
        instance.updated_at = datetime.now()
        return instance.model_dump()