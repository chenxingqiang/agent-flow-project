"""Base workflow module."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union
from enum import Enum
import logging
from datetime import datetime
from dataclasses import dataclass
from pydantic import Field, ConfigDict, BaseModel
from .types import AgentStatus
from ..ell2a.integration import ELL2AIntegration
from ..ell2a.types.message import Message, MessageRole
from .isa.isa_manager import ISAManager
from .instruction_selector import InstructionSelector
from .config import WorkflowConfig, StepConfig, WorkflowStepType, WorkflowStep

logger = logging.getLogger(__name__)

class WorkflowStepType(str, Enum):
    """Workflow step type enumeration."""
    TRANSFORM = "transform"
    ANALYZE = "analyze"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    CUSTOM = "custom"

class StepConfig(BaseModel):
    """Step configuration."""
    strategy: str
    params: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True)

class WorkflowStep(BaseModel):
    """Workflow step configuration."""
    id: Optional[str] = None
    name: Optional[str] = None
    type: WorkflowStepType
    config: StepConfig
    model_config = ConfigDict(frozen=True)

class WorkflowConfig(BaseModel):
    """Workflow configuration."""
    id: Optional[str] = None
    name: Optional[str] = None
    max_iterations: int = Field(default=10)
    timeout: float = Field(default=3600)
    logging_level: str = Field(default="INFO")
    required_fields: List[str] = Field(default_factory=list)
    error_handling: Dict[str, str] = Field(default_factory=dict)
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {"max_retries": 3, "retry_delay": 1.0})
    error_policy: Dict[str, bool] = Field(default_factory=lambda: {"ignore_warnings": False, "fail_fast": True})
    steps: List[WorkflowStep] = Field(default_factory=list)
    ell2a_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    model_config = ConfigDict(
        frozen=True, 
        extra='allow'  # Allow extra fields
    )

@dataclass
class WorkflowInstance:
    """Workflow instance class."""
    id: str
    agent: 'Agent'
    status: str = "initialized"
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None

class WorkflowEngine:
    """Workflow engine class."""
    
    def __init__(self, workflow_def: Dict[str, Any], workflow_config: Union[Dict[str, Any], WorkflowConfig, None] = None):
        """Initialize workflow engine.
        
        Args:
            workflow_def: Workflow definition
            workflow_config: Workflow configuration
            
        Raises:
            ValueError: If workflow_config is invalid
            ValueError: If workflow definition is invalid
        """
        # Validate workflow definition
        if not workflow_def or not isinstance(workflow_def, dict) or \
           "COLLABORATION" not in workflow_def or \
           "WORKFLOW" not in workflow_def["COLLABORATION"]:
            raise ValueError("Workflow definition must contain COLLABORATION.WORKFLOW")
            
        self.workflow_def = workflow_def
        self._initialized = False
        self._ell2a = None
        self._isa_manager = None
        self._instruction_selector = None
        
        # Convert workflow_config to WorkflowConfig if needed
        if workflow_config is None:
            self.workflow_config = WorkflowConfig()
        elif isinstance(workflow_config, dict):
            # Ensure steps are converted to WorkflowStep instances
            steps = []
            for step in workflow_config.get('steps', []):
                if not isinstance(step, WorkflowStep):
                    step_dict = step if isinstance(step, dict) else dict(step)
                    # Ensure required fields are present
                    if 'id' not in step_dict:
                        step_dict['id'] = f"step-{len(steps) + 1}"
                    if 'name' not in step_dict:
                        step_dict['name'] = step_dict['id']
                    if 'type' not in step_dict:
                        step_dict['type'] = WorkflowStepType.TRANSFORM
                    if 'config' not in step_dict:
                        step_dict['config'] = {"strategy": "default"}
                    elif isinstance(step_dict['config'], StepConfig):
                        step_dict['config'] = step_dict['config'].__dict__
                    elif not isinstance(step_dict['config'], dict):
                        step_dict['config'] = {"strategy": "default"}
                    
                    step = WorkflowStep(**step_dict)
                steps.append(step)
            
            workflow_config['steps'] = steps
            self.workflow_config = WorkflowConfig(**workflow_config)
        elif isinstance(workflow_config, WorkflowConfig):
            # Create a new WorkflowConfig with converted steps
            steps = []
            for step in workflow_config.steps:
                steps.append(WorkflowStep(
                    id=step.id or f"step-{len(steps) + 1}",
                    name=step.name or step.id,
                    type=step.type,
                    config=step.config
                ))
            
            self.workflow_config = WorkflowConfig(
                name=workflow_config.name,
                max_iterations=workflow_config.max_iterations,
                timeout=workflow_config.timeout,
                logging_level=workflow_config.logging_level,
                required_fields=workflow_config.required_fields,
                error_policy=workflow_config.error_policy,
                steps=steps
            )
        elif isinstance(workflow_config, str):
            # Validate that the input is a valid workflow configuration name or path
            if not workflow_config:
                raise ValueError("workflow_config must be an instance of WorkflowConfig, a dictionary, or None")
            
            # Attempt to load configuration from a file or configuration manager
            try:
                from agentflow.core.config_manager import ConfigManager
                if hasattr(ConfigManager, 'load_workflow_config'):
                    self.workflow_config = ConfigManager.load_workflow_config(workflow_config)
                else:
                    raise ValueError("ConfigManager does not have load_workflow_config method")
            except Exception as e:
                raise ValueError(f"workflow_config must be an instance of WorkflowConfig, a dictionary, or None: {str(e)}")
        else:
            # Try converting to dictionary if it has __dict__ attribute
            try:
                workflow_dict = {}
                for k, v in workflow_config.__dict__.items():
                    if not k.startswith('_'):
                        if k == 'steps':
                            # Ensure steps are converted to WorkflowStep instances
                            steps = []
                            for step in v:
                                if not isinstance(step, WorkflowStep):
                                    step_dict = step if isinstance(step, dict) else dict(step)
                                    # Ensure required fields are present
                                    if 'id' not in step_dict:
                                        step_dict['id'] = f"step-{len(steps) + 1}"
                                    if 'name' not in step_dict:
                                        step_dict['name'] = step_dict['id']
                                    if 'type' not in step_dict:
                                        step_dict['type'] = WorkflowStepType.TRANSFORM
                                    if 'config' not in step_dict:
                                        step_dict['config'] = {"strategy": "default"}
                                    elif isinstance(step_dict['config'], StepConfig):
                                        step_dict['config'] = step_dict['config'].__dict__
                                    elif not isinstance(step_dict['config'], dict):
                                        step_dict['config'] = {"strategy": "default"}
                                    
                                    step = WorkflowStep(**step_dict)
                                steps.append(step)
                            workflow_dict[k] = steps
                        else:
                            workflow_dict[k] = v
                
                self.workflow_config = WorkflowConfig(**workflow_dict)
            except Exception as e:
                raise ValueError(f"workflow_config must be an instance of WorkflowConfig, a dictionary, or None: {str(e)}")
            
        self.workflows = {}
        self.state_manager = {}
        
    async def initialize(self):
        """Initialize workflow engine."""
        self._initialized = True
        
    async def cleanup(self):
        """Clean up workflow resources."""
        self._initialized = False
        
        # Cleanup mocked components if they exist
        if self._ell2a and hasattr(self._ell2a, 'cleanup'):
            await self._ell2a.cleanup()
        
        if self._isa_manager and hasattr(self._isa_manager, 'cleanup'):
            await self._isa_manager.cleanup()
        
        if self._instruction_selector and hasattr(self._instruction_selector, 'cleanup'):
            await self._instruction_selector.cleanup()
        
        self.workflows.clear()
        
    async def register_workflow(self, agent):
        """Register a workflow with an agent."""
        workflow_id = f"workflow_{len(self.workflows) + 1}"
        self.workflows[workflow_id] = WorkflowInstance(
            id=workflow_id,
            agent=agent
        )
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered workflow.
        
        Args:
            workflow_id: Unique identifier for the workflow
            input_data: Input data for the workflow
        
        Returns:
            Dictionary with workflow execution result
        
        Raises:
            ValueError: If workflow is not registered or input is invalid
        """
        if not self._initialized:
            raise ValueError("Workflow engine not initialized")
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not registered")
        
        if input_data is None:
            raise ValueError("Input data cannot be None")
        
        workflow_instance = self.workflows[workflow_id]
        agent = workflow_instance.agent
        
        try:
            # Execute workflow steps
            result = await agent.process_message(input_data)
            
            # Standardize the result format
            return {
                "status": "success", 
                "content": result,
                "workflow_id": workflow_id
            }
        except Exception as e:
            # Handle workflow execution errors
            workflow_instance.status = "failed"
            workflow_instance.error = str(e)
            
            return {
                "status": "error", 
                "error": str(e),
                "workflow_id": workflow_id
            }