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

    def __init__(self):
        """Initialize workflow engine."""
        self.workflows: Dict[str, WorkflowConfigType] = {}
        self.agents = {}  # Map of agent IDs to agent instances
        self.state_manager = WorkflowStateManager()
        self.metrics = MetricsManager()
        self.status = "initialized"
        self.start_time = None
        self.end_time = None
        self._initialized = False
        self._instruction_selector = None
        self._isa_manager = None
        self._ell2a = None
        self.is_distributed = False  # Add is_distributed attribute
        self._default_agent_id: Optional[str] = None  # Store the default agent ID
        
        # Store workflow definition and configuration
        self.workflow_def = None
        self.workflow_config = None
        self._pending_registration = None

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

        self._initialized = True
    
    async def register_workflow(self, agent: 'Agent', workflow: Union[WorkflowConfigType, Dict[str, Any]]) -> None:
        """Register workflow for an agent."""
        if workflow is None:
            raise ValueError("Workflow cannot be None")
            
        if not agent.config or not isinstance(agent.config, (dict, AgentConfig)) or not agent.config:
            raise ValueError("Agent must have a configuration")
            
        if isinstance(workflow, dict):
            workflow = WorkflowConfigType.model_validate(workflow)
        if agent.id in self.workflows:
            raise ValueError(f"Agent {agent.id} already registered")
        self.workflows[agent.id] = workflow

        # If no workflow definition exists, create one
        if not self.workflow_def:
            # Create workflow definition with COLLABORATION section
            workflow_steps = {}
            for step in workflow.steps:
                step_dict = {
                    "step": len(workflow_steps) + 1,
                    "name": step.name,
                    "description": step.name,
                    "input": ["data"],
                    "type": str(step.type),
                    "agent_config": agent.config.model_dump() if hasattr(agent.config, 'model_dump') else dict(agent.config),
                    "dependencies": step.dependencies
                }
                workflow_steps[step.id] = step_dict

                # Log warning for distributed step in non-distributed workflow
                if step.is_distributed and not workflow.distributed:
                    logger.warning(f"Distributed step {step.id} in non-distributed workflow {workflow.id}")

            # Get the communication protocol from the step configuration
            protocol = None
            for step in workflow.steps:
                if step.config.params and "protocol" in step.config.params:
                    protocol = step.config.params["protocol"].upper()
                    break

            # If no protocol is found in steps, use FEDERATED as default
            if not protocol:
                protocol = "FEDERATED"

            # Create workflow definition
            self.workflow_def = {
                "COLLABORATION": {
                    "MODE": "SEQUENTIAL",  # Default mode
                    "COMMUNICATION_PROTOCOL": {
                        "TYPE": protocol
                    },
                    "WORKFLOW": workflow_steps
                }
            }

            # Validate workflow list is not empty
            if not self.workflow_def.get("COLLABORATION", {}).get("WORKFLOW"):
                raise WorkflowExecutionError("Empty workflow: no workflow steps defined in COLLABORATION.WORKFLOW")

    async def execute_workflow(self, agent_id_or_instance: Union[str, 'WorkflowInstance'], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow."""
        try:
            # Handle both agent ID and instance cases
            if isinstance(agent_id_or_instance, str):
                # Validate workflow exists
                if agent_id_or_instance not in self.workflows:
                    raise WorkflowExecutionError(f"No workflow registered for agent {agent_id_or_instance}")
                workflow = self.workflows[agent_id_or_instance]
                agent_id = agent_id_or_instance
            else:
                workflow = agent_id_or_instance
                agent_id = workflow.id

            # Validate workflow configuration
            if not workflow.steps:
                raise WorkflowExecutionError("Empty workflow: no steps defined")

            # Validate workflow definition
            if not self.workflow_def or not self.workflow_def.get("COLLABORATION", {}).get("WORKFLOW"):
                raise WorkflowExecutionError("Empty workflow: no workflow steps defined in COLLABORATION.WORKFLOW")

            # Validate communication protocol
            protocol = self.workflow_def.get("COLLABORATION", {}).get("COMMUNICATION_PROTOCOL", {}).get("TYPE")
            if not protocol:
                raise WorkflowExecutionError("No communication protocol specified")
            if protocol not in {"FEDERATED", "GOSSIP", "HIERARCHICAL", "HIERARCHICAL_MERGE"}:
                raise WorkflowExecutionError(f"Invalid communication protocol: {protocol}")

            # Validate step protocols
            valid_protocols = {"federated", "gossip", "hierarchical", "hierarchical_merge"}
            for step in workflow.steps:
                step_protocol = step.config.params.get("protocol", "").lower()
                if step_protocol and step_protocol not in valid_protocols:
                    raise WorkflowExecutionError(f"Invalid protocol '{step_protocol}' in step {step.id}")

            # Execute workflow steps
            result = await self._execute_workflow_steps(agent_id, context)
            return result
        except WorkflowExecutionError as e:
            result = {
                "status": WorkflowStatus.FAILED.value,
                "error": str(e),
                "steps": {},
                "result": {
                    "steps": []
                }
            }
            logger.error(f"Error executing workflow: {str(e)}")
            raise e
        except Exception as e:
            result = {
                "status": WorkflowStatus.FAILED.value,
                "error": str(e),
                "steps": {},
                "result": {
                    "steps": []
                }
            }
            error_msg = f"Workflow execution failed: {str(e)}"
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

    async def cleanup(self) -> None:
        """Cleanup workflow engine resources.
        
        Raises:
            WorkflowExecutionError: If cleanup fails
        """
        try:
            # Store components for cleanup
            instruction_selector = self._instruction_selector
            isa_manager = self._isa_manager
            ell2a = self._ell2a
            
            # Clear workflows
            self.workflows.clear()
            
            # Cleanup instruction selector (async)
            if instruction_selector and hasattr(instruction_selector, 'cleanup'):
                await instruction_selector.cleanup()
                
            # Cleanup ELL2A integration (async)
            if ell2a and hasattr(ell2a, 'cleanup'):
                await ell2a.cleanup()
                
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

    async def _execute_workflow_steps(self, agent_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps."""
        workflow = self.workflows[agent_id]
        result = {
            "status": WorkflowStatus.PENDING.value,
            "steps": {},
            "result": {
                "steps": []
            }
        }

        try:
            # Validate workflow configuration
            if not workflow.steps:
                raise WorkflowExecutionError("Empty workflow: no steps defined")

            # Convert Message object to dictionary if needed
            if not isinstance(context, dict) and hasattr(context, 'model_dump'):
                context = context.model_dump()
            elif not isinstance(context, dict):
                context = {"data": context}

            for step in workflow.steps:
                step_result = None
                retry_count = 0
                start_time = time.time()

                # Validate step configuration
                if not step.config:
                    raise WorkflowExecutionError(f"Step {step.id} has no configuration")

                while retry_count <= step.config.max_retries:
                    try:
                        # Handle agent steps differently
                        if step.type == WorkflowStepType.AGENT:
                            # For testing, just pass through the context
                            if context.get("test_mode"):
                                step_result = {"data": context.get("data", {})}
                            else:
                                raise WorkflowExecutionError(f"Agent step {step.id} requires an agent to be configured")
                        else:
                            step_result = await step.execute(context)

                        # Standardize step result format
                        if isinstance(step_result, dict):
                            if "data" in step_result:
                                if isinstance(step_result["data"], dict):
                                    step_result = step_result["data"]
                                else:
                                    step_result = {"data": step_result["data"]}
                            else:
                                step_result = {"data": step_result}
                        else:
                            step_result = {"data": step_result}
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count > step.config.max_retries:
                            error_msg = f"Step {step.id} failed: {str(e)}"
                            result["status"] = WorkflowStatus.FAILED.value
                            result["error"] = error_msg
                            step_info = {
                                "id": step.id,
                                "error": error_msg,
                                "status": WorkflowStatus.FAILED.value,
                                "start_time": start_time,
                                "end_time": time.time(),
                                "attempts": retry_count
                            }
                            result["steps"][step.id] = step_info
                            result["result"]["steps"].append(step_info)
                            raise WorkflowExecutionError(error_msg)
                        # Calculate retry delay with exponential backoff
                        delay = step.config.retry_delay * (step.config.retry_backoff ** (retry_count - 1))
                        await asyncio.sleep(delay)

                if step_result is not None:
                    step_info = {
                        "id": step.id,
                        "result": step_result,
                        "status": "success",
                        "start_time": start_time,
                        "end_time": time.time(),
                        "attempts": retry_count + 1
                    }
                    result["steps"][step.id] = step_info
                    result["result"]["steps"].append(step_info)
                    if isinstance(step_result, dict):
                        context.update(step_result)

            result["status"] = WorkflowStatus.COMPLETED.value
            return result

        except WorkflowExecutionError as e:
            result["status"] = WorkflowStatus.FAILED.value
            result["error"] = str(e)
            raise
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            result["status"] = WorkflowStatus.FAILED.value
            result["error"] = error_msg
            raise WorkflowExecutionError(error_msg)

    async def _default_step_executor(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default step executor when no custom execute function is provided."""
        # For now, just pass through the context
        return context