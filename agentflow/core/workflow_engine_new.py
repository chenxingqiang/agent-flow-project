"""Workflow engine module."""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple, TYPE_CHECKING
import logging
import ray
import time

from .workflow_types import (
    WorkflowConfig as WorkflowConfigType,
    WorkflowStep,
    WorkflowStepType,
    StepConfig,
    WorkflowStatus
)
from .workflow_state import WorkflowStateManager
from .metrics import MetricsManager
from .exceptions import WorkflowExecutionError
from .workflow_executor import WorkflowExecutor
from ..agents.agent import Agent
from ..agents.agent_types import AgentType
from .config import AgentConfig, ModelConfig

if TYPE_CHECKING:
    from .workflow import WorkflowInstance

logger = logging.getLogger(__name__)

class WorkflowEngine:
    """Workflow engine for executing workflows."""

    def __init__(self, workflow_config: Optional[Union[WorkflowConfigType, Dict[str, Any]]] = None):
        """Initialize workflow engine.
        
        Args:
            workflow_config: Optional workflow configuration
        """
        self._initialized = False
        self.workflows: Dict[str, WorkflowConfigType] = {}
        self.agents = {}  # Map of agent IDs to agent instances
        self.state_manager = WorkflowStateManager()
        self.metrics = MetricsManager()
        self.status = "initialized"
        self.start_time = None
        self.end_time = None
        self._instruction_selector = None
        self._isa_manager = None
        self._ell2a = None
        self.is_distributed = False  # Add is_distributed attribute
        self._default_agent_id: Optional[str] = None  # Store the default agent ID
        
        # Store workflow configuration
        if workflow_config:
            if not isinstance(workflow_config, (WorkflowConfigType, dict)):
                raise ValueError("workflow_config must be an instance of WorkflowConfig, a dictionary, or None")
            if isinstance(workflow_config, dict):
                workflow_config = WorkflowConfigType.model_validate(workflow_config)
            self.workflow_config = workflow_config
            self._pending_registration = None
        else:
            self.workflow_config = None
            self._pending_registration = None
            
        # Initialize workflow definition as empty
        self.workflow_def = None

    @property
    def default_agent_id(self) -> Optional[str]:
        """Get the default agent ID."""
        return self._default_agent_id
        
    @default_agent_id.setter
    def default_agent_id(self, value: str) -> None:
        """Set the default agent ID."""
        self._default_agent_id = value
    
    async def initialize(self, workflow_def: Optional[Dict[str, Any]] = None, workflow_config: Optional[Union[WorkflowConfigType, Dict[str, Any]]] = None) -> None:
        """Initialize workflow engine."""
        if workflow_def:
            self.workflow_def = workflow_def
        else:
            self.workflow_def = {}

        if workflow_config:
            if not isinstance(workflow_config, (WorkflowConfigType, dict)):
                raise ValueError("workflow_config must be an instance of WorkflowConfig, a dictionary, or None")
            if isinstance(workflow_config, dict):
                workflow_config = WorkflowConfigType.model_validate(workflow_config)
            self.workflow_config = workflow_config

            # Create a default agent for the workflow
            agent_config = AgentConfig(
                name="default_agent",
                type=AgentType.RESEARCH,
                model=ModelConfig.model_validate({"name": "gpt-4", "provider": "openai"}),
                workflow=workflow_config.model_dump()
            )
            agent = Agent(config=agent_config)
            
            # Store the registration task to be awaited during initialization
            self._pending_registration = (agent, workflow_config)
        else:
            self.workflow_config = None

        # Initialize components
        from ..ell2a.integration import ELL2AIntegration
        self._ell2a = ELL2AIntegration()
        
        from .isa.isa_manager import ISAManager
        self._isa_manager = ISAManager()
        await self._isa_manager.initialize()
        
        from .instruction_selector import InstructionSelector
        self._instruction_selector = InstructionSelector()
        await self._instruction_selector.initialize()

        self._initialized = True

    async def register_workflow(self, agent: 'Agent') -> str:
        """Register workflow for an agent.
        
        Args:
            agent: Agent instance to register workflow for
            
        Returns:
            str: ID of the registered workflow
        """
        if not agent.config or not isinstance(agent.config, (dict, AgentConfig)):
            raise ValueError("Agent must have a configuration")
            
        if not agent.config.workflow:
            raise ValueError("Agent must have a workflow configuration")
            
        workflow_id = str(uuid.uuid4())
        
        # Convert to WorkflowConfig if needed
        workflow = agent.config.workflow
        if isinstance(workflow, dict):
            workflow = WorkflowConfigType.model_validate(workflow)
        elif not isinstance(workflow, WorkflowConfigType):
            raise ValueError("Invalid workflow configuration type")
            
        self.workflows[workflow_id] = workflow
        self.agents[workflow_id] = agent
        
        return workflow_id

    async def execute_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
            context: Execution context
            
        Returns:
            dict: Execution result
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"No workflow found with ID {workflow_id}")
            
        workflow = self.workflows[workflow_id]
        agent = self.agents[workflow_id]
        
        try:
            # Validate workflow configuration
            if not workflow.steps:
                raise WorkflowExecutionError("Empty workflow: no steps defined")

            # Convert Message object to dictionary if needed
            if not isinstance(context, dict) and hasattr(context, 'model_dump'):
                context = context.model_dump()
            elif not isinstance(context, dict):
                context = {"data": context}

            result = {
                "status": "success",
                "content": "",
                "steps": []
            }

            for step in workflow.steps:
                step_result = await self._execute_step(step, context, agent)
                result["steps"].append(step_result)
                if isinstance(step_result, dict) and "content" in step_result:
                    result["content"] = step_result["content"]

            return result
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)

    async def cleanup(self) -> None:
        """Cleanup workflow engine resources.
        
        Raises:
            WorkflowExecutionError: If cleanup fails
        """
        try:
            # Clear workflows
            self.workflows.clear()
            
            # Cleanup instruction selector
            if self._instruction_selector:
                await self._instruction_selector.cleanup()
                
            # Cleanup ISA manager
            if self._isa_manager:
                await self._isa_manager.cleanup()
                
            # Cleanup ELL2A integration
            if self._ell2a:
                self._ell2a.cleanup()  # Not async
                
            # Clear component references
            self._instruction_selector = None
            self._isa_manager = None
            self._ell2a = None
            self._initialized = False
            
            # Update status
            self.status = "cleaned"
            self.end_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error during workflow engine cleanup: {str(e)}")
            raise WorkflowExecutionError(f"Failed to cleanup workflow engine: {str(e)}")

    async def _execute_step(self, step: WorkflowStep, context: Dict[str, Any], agent: 'Agent') -> Dict[str, Any]:
        """Execute a single workflow step.
        
        Args:
            step: Step to execute
            context: Execution context
            agent: Agent instance
            
        Returns:
            dict: Step execution result
        """
        try:
            # For testing, just pass through the context
            if context.get("test_mode"):
                return {"data": context.get("data", {})}
                
            # Execute step based on type
            if step.type == WorkflowStepType.TRANSFORM:
                if step.config.strategy == "custom" and "execute" in step.config.params:
                    execute_func = step.config.params["execute"]
                    if asyncio.iscoroutinefunction(execute_func):
                        result = await execute_func(context.get("data", {}))
                    else:
                        result = execute_func(context.get("data", {}))
                else:
                    result = context.get("data", {})
            else:
                result = await agent.process_message(context.get("message", ""))
                
            return {"content": result} if isinstance(result, str) else result
            
        except Exception as e:
            error_msg = f"Step {step.id} failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)

    def get_workflow(self, agent_id: str) -> Optional[WorkflowConfigType]:
        """Get workflow configuration for an agent.

        Args:
            agent_id: ID of the agent.

        Returns:
            Workflow configuration if found, None otherwise.
        """
        return self.workflows.get(agent_id)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent instance.

        Args:
            agent_id: ID of the agent.

        Returns:
            Agent instance if found, None otherwise.
        """
        return self.agents.get(agent_id) 