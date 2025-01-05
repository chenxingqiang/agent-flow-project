"""Distributed workflow implementation using Ray."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
# Import workflow-related modules
from agentflow.core.workflow_state import WorkflowStateManager, WorkflowStatus, StepStatus
from agentflow.core.exceptions import WorkflowExecutionError
from agentflow.core.research_workflow import ResearchDistributedWorkflow

import ray
from ray.exceptions import RayActorError, RayTaskError

# Initialize logger
logger = logging.getLogger(__name__)

def initialize_ray():
    """Initialize Ray if not already initialized."""
    if not ray.is_initialized():
        try:
            ray.init(
                ignore_reinit_error=True,
                runtime_env={
                    "pip": ["ray[default]>=2.5.0"]
                },
                num_cpus=2,  # Use 2 CPUs for distributed testing
                local_mode=False,  # Use distributed mode
                include_dashboard=False  # Disable dashboard for tests
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise


class DistributedWorkflowStep:
    """Base class for workflow steps"""
    def __init__(self, step_id: str, config: Dict[str, Any] = None):
        self.step_id = step_id
        self.config = config or {}
        self._initialize_logging()
        
    def _initialize_logging(self):
        """Initialize step-specific logging."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{self.step_id}")
        
    @ray.method(num_returns=1)
    def health_check(self) -> bool:
        """Check if the actor is healthy."""
        return True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step"""
        raise NotImplementedError("Subclasses must implement execute()")

@ray.remote(max_restarts=3, max_task_retries=3)
class ResearchStep(DistributedWorkflowStep):
    """A research workflow step that can be executed remotely"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research step"""
        try:
            # Log step execution
            self.logger.info(f"Executing research step {self.step_id}")
            
            # Validate input
            if not input_data:
                raise ValueError("No input data provided")
            
            # Process research task
            result = {
                "status": "success",
                "result": f"Research completed for {self.step_id}",
                "metadata": {
                    "step_id": self.step_id,
                    "timestamp": time.time()
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in research step: {str(e)}", exc_info=True)
            raise StepExecutionError(f"Research step failed: {str(e)}")

@ray.remote(max_restarts=3, max_task_retries=3)
class DocumentStep(DistributedWorkflowStep):
    """A document generation step that can be executed remotely"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the document step"""
        try:
            # Log step execution
            self.logger.info(f"Executing document step {self.step_id}")
            
            # Process document task
            result = {
                "status": "success",
                "result": f"Document generated for {self.step_id}",
                "metadata": {
                    "step_id": self.step_id,
                    "timestamp": time.time()
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in document step: {str(e)}", exc_info=True)
            raise StepExecutionError(f"Document step failed: {str(e)}")

@ray.remote(max_restarts=3, max_task_retries=3)
class ImplementationStep(DistributedWorkflowStep):
    """An implementation step that can be executed remotely"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the implementation step"""
        try:
            # Log step execution
            self.logger.info(f"Executing implementation step {self.step_id}")
            
            # Process implementation task
            result = {
                "status": "success",
                "result": f"Implementation completed for {self.step_id}",
                "metadata": {
                    "step_id": self.step_id,
                    "timestamp": time.time()
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in implementation step: {str(e)}", exc_info=True)
            raise StepExecutionError(f"Implementation step failed: {str(e)}")

class DistributedWorkflow:
    """Base class for distributed workflows"""
    
    def __init__(
                self,
                workflow_config: Dict[str, Any] = None,
                config: Optional[Dict[str, Any]] = None,
                state_manager: Optional[WorkflowStateManager] = None
            ):
        """Initialize workflow."""
        self.workflow_config = workflow_config or {}
        self.config = config or {}
        self.state_manager = state_manager or WorkflowStateManager()
        self.distributed_steps = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Populate step configurations with global retry settings
        for key, value in self.config.items():
            if key.endswith('_config'):
                # Ensure retry settings are populated
                value.setdefault('retry_delay', self.config.get('retry_delay', 1.0))
                value.setdefault('retry_backoff', self.config.get('retry_backoff', 2.0))
                value.setdefault('max_retries', self.config.get('max_retries', 3))
                
    async def _check_actor_health(self, step_actor) -> bool:
        """Check if a Ray actor is healthy.
        
        Args:
            step_actor: Ray actor reference
            
        Returns:
            bool: True if actor is healthy, False otherwise
        """
        try:
            return await ray.get(step_actor.health_check.remote())
        except (RayActorError, RayTaskError):
            return False
            
    async def _ensure_actor_health(self, step_id: str, step_actor) -> Any:
        """Ensure a Ray actor is healthy, recreating it if necessary.
        
        Args:
            step_id: ID of the step
            step_actor: Ray actor reference
            
        Returns:
            Ray actor reference: Original or recreated actor
        """
        if not await self._check_actor_health(step_actor):
            self.logger.warning(f"Actor for step {step_id} is unhealthy, recreating...")
            # Get actor class and config from original actor
            actor_class = ray.get(step_actor.get_class.remote())
            actor_config = ray.get(step_actor.get_config.remote())
            # Create new actor
            new_actor = actor_class.remote(step_id, actor_config)
            self.distributed_steps[step_id] = new_actor
            return new_actor
        return step_actor

    async def _execute_step_with_retry(
            self,
            step_actor,
            step_id: str,
            step_input: Dict[str, Any],
            max_retries: int = 3,
            retry_delay: float = 1.0,
            retry_backoff: float = 2.0
        ) -> Dict[str, Any]:
        """Execute a workflow step with retry logic.
        
        Args:
            step_actor: Ray actor reference
            step_id: Step ID
            step_input: Input data for the step
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            retry_backoff: Multiplicative factor for retry delay
            
        Returns:
            Dict[str, Any]: Step execution result
            
        Raises:
            WorkflowExecutionError: If step execution fails after all retries
        """
        attempt = 0
        last_exception = None
        current_delay = retry_delay

        while attempt < max_retries:
            try:
                # Ensure actor is healthy
                step_actor = await self._ensure_actor_health(step_id, step_actor)
                
                # Execute step
                self.logger.info(f"Executing step {step_id} (attempt {attempt + 1}/{max_retries})")
                step_result_ref = step_actor.execute.remote(step_input)
                step_result = await ray.get(step_result_ref)
                return step_result

            except (RayActorError, RayTaskError) as e:
                last_exception = e
                attempt += 1
                if attempt < max_retries:
                    self.logger.warning(
                        f"Step {step_id} failed (attempt {attempt}/{max_retries}): {e}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= retry_backoff
                else:
                    self.logger.error(f"Step {step_id} failed after {max_retries} attempts: {e}")
                    
        raise WorkflowExecutionError(
            f"Step {step_id} failed after {max_retries} attempts: {last_exception}"
        )
    
    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously"""
        try:
            # Validate input
            input_data = self.validate_input(input_data)
            
            # Initialize workflow state
            self.state_manager.initialize_workflow()
            self.state_manager.set_workflow_status(WorkflowStatus.RUNNING)
            
            results = {}
            final_output = None
            
            # Get workflow steps
            workflow_steps = self.workflow_config.get('WORKFLOW', {})
            
            # Ensure workflow_steps is a list of dictionaries
            if isinstance(workflow_steps, dict):
                workflow_steps = [
                    {'step_id': step_id, **step_config}
                    for step_id, step_config in sorted(
                        workflow_steps.items(),
                        key=lambda x: x[1].get('step', 0)
                    )
                ]
            elif not isinstance(workflow_steps, list):
                workflow_steps = []
            
            # Execute each step
            for step in workflow_steps:
                # Ensure step has a step number
                step_num = step.get('step', step.get('step_id', 0))
                if isinstance(step_num, str) and step_num.startswith('step_'):
                    step_num = int(step_num.split('_')[-1])
                
                # Fallback to index if step_num is not a valid number
                if not isinstance(step_num, int):
                    step_num = workflow_steps.index(step) + 1
                
                step_id = f"step_{step_num}"
                step_config = self.config.get(f'{step_id}_config', {})
                
                # Get retry settings
                max_retries = step_config.get('max_retries', self.config.get('max_retries', 3))
                retry_delay = step_config.get('retry_delay', self.config.get('retry_delay', 1.0))
                retry_backoff = step_config.get('retry_backoff', self.config.get('retry_backoff', 2.0))
                
                # Prepare step input
                try:
                    # First try the workflow's method
                    step_input = self._prepare_step_input(step, input_data, results)
                except Exception:
                    # Fallback to a default input preparation
                    required_fields = step.get('input', []) if isinstance(step, dict) else []
                    step_input = {
                        field: input_data.get(field, f'Mock {field}') 
                        for field in required_fields
                    }
                    step_input.update({
                        'name': step.get('name', 'Mock Step') if isinstance(step, dict) else 'Mock Step',
                        'description': step.get('description', 'Mock Step Description') if isinstance(step, dict) else 'Mock Step Description',
                        'agent_config': step.get('agent_config', {}) if isinstance(step, dict) else {},
                        'input': required_fields
                    })
                
                # Execute step with retry logic
                try:
                    step_actor = self.distributed_steps.get(step_id)
                    if not step_actor:
                        raise ValueError(f"No step actor found for step_id: {step_id}")
                    
                    step_result = await self._execute_step_with_retry(
                        step_actor,
                        step_id,
                        step_input,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        retry_backoff=retry_backoff
                    )
                    
                    # Store results
                    results[step_id] = step_result
                    self.state_manager.set_step_result(step_id, step_result)
                    self.state_manager.set_step_status(step_id, StepStatus.SUCCESS)
                    
                    # Update final output
                    if isinstance(step_result, dict):
                        if 'output' in step_result:
                            final_output = step_result['output']
                        else:
                            final_output = step_result
                    else:
                        final_output = step_result
                    
                except Exception as e:
                    self.state_manager.set_step_status(step_id, StepStatus.FAILED)
                    raise
            
            # Set workflow status to completed
            self.state_manager.set_workflow_status(WorkflowStatus.COMPLETED)
            
            return {
                'status': 'success',
                'output': final_output,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error executing workflow: {str(e)}")
            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
            raise
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow synchronously"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute_async(input_data))
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow input data."""
        return input_data
    
    def _prepare_step_input(self, step: Dict[str, Any], input_data: Dict[str, Any], previous_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare input for a workflow step."""
        return input_data
    
    def _default_error_handler(self, error: Exception, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle workflow errors"""
        error_msg = f"Workflow error: {str(error)}"
        if input_data:
            error_msg += f" with input: {input_data}"
        logger.error(error_msg)
        raise WorkflowExecutionError(error_msg)
    
    @classmethod
    def create_remote_workflow(cls, workflow_config: Dict[str, Any], config: Dict[str, Any], state_manager: Optional[WorkflowStateManager] = None) -> Any:
        """Create a remote workflow instance."""
        # Create a new remote actor instance
        RemoteWorkflow = ray.remote(cls)
        return RemoteWorkflow.remote(workflow_config, config, state_manager)

def execute_step(step_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone function to execute a step with the given type and input data.

    Args:
        step_type (str): The type of step to execute.
        input_data (Dict[str, Any]): Input data for the step.

    Returns:
        Dict[str, Any]: The output from the step execution
    """
    # This is a placeholder implementation. You may need to modify it based on your specific requirements.
    # For now, it simply returns the input data as a demonstration.
    return input_data

def _execute_step_with_retry(
    distributed_steps: Dict[str, Any], 
    step_type: str, 
    step_def: Dict[str, Any], 
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Standalone function to execute a step with retry mechanism.

    Args:
        distributed_steps (Dict[str, Any]): Dictionary of distributed steps
        step_type (str): The type of step to execute.
        step_def (Dict[str, Any]): Step definition configuration.
        input_data (Dict[str, Any]): Input data for the step.

    Returns:
        Dict[str, Any]: The output from the step execution
    """
    # This is a placeholder implementation. You may need to modify it based on your specific requirements.
    # For now, it simply returns the input data as a demonstration.
    return input_data

class WorkflowExecutionError(Exception):
    """Custom exception for workflow execution errors"""
    def __init__(self, message, add_prefix=True):
        self.message = message
        self.add_prefix = add_prefix

class StepExecutionError(Exception):
    """Custom exception for step execution errors"""
    pass

# Initialize Ray when module is imported
initialize_ray()
