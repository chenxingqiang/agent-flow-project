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

import ray

# Initialize Ray with proper configuration
if not ray.is_initialized():
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "pip": ["ray[default]>=2.5.0"]
        }
    )

# Configure logging for the entire module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import workflow-related modules
from agentflow.core.workflow import WorkflowExecutionError, WorkflowStatus, StepStatus
from agentflow.core.workflow_state import WorkflowStateManager

class DistributedWorkflowStep:
    """Base class for workflow steps"""
    def __init__(self, step_id: str, config: Dict[str, Any] = None):
        self.step_id = step_id
        self.config = config or {}

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step"""
        raise NotImplementedError("Subclasses must implement execute()")

@ray.remote
class ResearchStep(DistributedWorkflowStep):
    """A research workflow step that can be executed remotely"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research step"""
        try:
            # Log step execution
            logger.info(f"Executing research step {self.step_id}")
            
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
            logger.error(f"Error in research step: {str(e)}")
            raise

@ray.remote
class DocumentStep(DistributedWorkflowStep):
    """A document generation step that can be executed remotely"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the document step"""
        try:
            # Log step execution
            logger.info(f"Executing document step {self.step_id}")
            
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
            logger.error(f"Error in document step: {str(e)}")
            raise

@ray.remote
class ImplementationStep(DistributedWorkflowStep):
    """An implementation step that can be executed remotely"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the implementation step"""
        try:
            # Log step execution
            logger.info(f"Executing implementation step {self.step_id}")
            
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
            logger.error(f"Error in implementation step: {str(e)}")
            raise

class DistributedWorkflow:
    """Base class for distributed workflows"""
    def __init__(
        self,
        workflow_config: Dict[str, Any] = None,
        config: Optional[Dict[str, Any]] = None,
        state_manager: Optional[WorkflowStateManager] = None
    ):
        self.workflow_config = workflow_config or {}
        self.config = config or {}
        self.state_manager = state_manager or WorkflowStateManager()
        self.distributed_steps = {}

        # Populate step configurations with global retry settings
        for key, value in self.config.items():
            if key.endswith('_config'):
                # Ensure retry settings are populated
                value.setdefault('retry_delay', self.config.get('retry_delay', 1.0))
                value.setdefault('retry_backoff', self.config.get('retry_backoff', 2.0))
                value.setdefault('max_retries', self.config.get('max_retries', 3))

    def _execute_step_with_retry(
        self,
        distributed_steps: Dict[str, Any],
        step_id: str,
        step_input: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0
    ) -> Dict[str, Any]:
        """Execute a workflow step with retry logic."""
        attempt = 0
        last_exception = None
        current_delay = retry_delay

        while attempt < max_retries:
            try:
                step_actor = distributed_steps.get(step_id)
                if not step_actor:
                    raise ValueError(f"No step actor found for step_id: {step_id}")
                
                result = ray.get(step_actor.execute.remote(step_input))
                return result

            except Exception as e:
                last_exception = e
                attempt += 1
                if attempt < max_retries:
                    time.sleep(current_delay)
                    current_delay *= retry_backoff
                logger.warning(f"Retry attempt {attempt} for step {step_id} after error: {str(e)}")

        error_msg = f"Step {step_id} failed after {max_retries} attempts. Last error: {str(last_exception)}"
        logger.error(error_msg)
        raise WorkflowExecutionError(error_msg)

    def _execute_remote_step(self, step_actor, step_input):
        """Execute a single step remotely."""
        try:
            return ray.get(step_actor.execute.remote(step_input))
        except Exception as e:
            logger.error(f"Error executing remote step: {str(e)}")
            raise

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow synchronously"""
        try:
            # Validate input data
            self.validate_input(input_data)

            # Initialize workflow state
            self.state_manager.initialize_workflow()
            self.state_manager.set_workflow_status(WorkflowStatus.RUNNING)

            # Execute steps in order
            results = {}
            final_output = None

            # Normalize workflow steps to a list
            workflow_steps = self.workflow_config.get('WORKFLOW', [])
            if isinstance(workflow_steps, dict):
                workflow_steps = [
                    {'step_id': step_id, **step_config} 
                    for step_id, step_config in workflow_steps.items()
                ]

            for step in workflow_steps:
                # Normalize step to ensure it has a step_id
                if isinstance(step, dict):
                    step_id = step.get('step_id', f"step_{step.get('step', 0)}")
                else:
                    step_id = f"step_{step}"
                
                step_config = self.config.get(f'{step_id}_config', {})

                # Prepare step input
                step_input = self._prepare_step_input(step, input_data, results)

                try:
                    # Execute step with retry and await the result
                    step_result_ref = self._execute_step_with_retry(
                        distributed_steps=self.distributed_steps,
                        step_id=step_id,
                        step_input=step_input,
                        max_retries=step_config.get('max_retries', self.config.get('max_retries', 2)),
                        retry_delay=step_config.get('retry_delay', self.config.get('retry_delay', 1.0)),
                        retry_backoff=step_config.get('retry_backoff', self.config.get('retry_backoff', 2.0))
                    )
                    step_result = ray.get(step_result_ref)

                    # Store results
                    results[step_id] = step_result
                    self.state_manager.set_step_result(step_id, step_result)
                    self.state_manager.set_step_status(step_id, StepStatus.COMPLETED)

                    # Format output based on step configuration
                    step_output = {}
                    if isinstance(step, dict):
                        step_output = step.get('output', {})
                    
                    if step_output:
                        if isinstance(step_result, dict):
                            # Extract result content
                            result_content = step_result.get('result', step_result)
                            if isinstance(result_content, dict):
                                final_output = {
                                    'output': {
                                        **step_output,
                                        **result_content
                                    }
                                }
                            else:
                                final_output = {
                                    'output': {
                                        **step_output,
                                        'result': result_content
                                    }
                                }
                        else:
                            final_output = {
                                'output': {
                                    **step_output,
                                    'result': step_result
                                }
                            }
                    else:
                        final_output = step_result if isinstance(step_result, dict) else {'output': step_result}

                except Exception as e:
                    # Handle step failure
                    self.state_manager.set_step_status(step_id, StepStatus.FAILED)
                    self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
                    error_msg = f"Step {step_id} failed: {str(e)}"
                    logger.error(error_msg)
                    raise WorkflowExecutionError(error_msg)

            # Set final workflow status and return results
            self.state_manager.set_workflow_status(WorkflowStatus.COMPLETED)
            return final_output if final_output is not None else {'output': results}

        except Exception as e:
            # Handle workflow failure
            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)

    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously"""
        try:
            # Validate input data
            self.validate_input(input_data)

            # Initialize workflow state
            self.state_manager.initialize_workflow()
            self.state_manager.set_workflow_status(WorkflowStatus.RUNNING)

            # Execute steps in order
            results = {}
            final_output = None

            # Normalize workflow steps to a list
            workflow_steps = self.workflow_config.get('WORKFLOW', [])
            if isinstance(workflow_steps, dict):
                workflow_steps = [
                    {'step_id': step_id, **step_config} 
                    for step_id, step_config in workflow_steps.items()
                ]

            for step in workflow_steps:
                # Normalize step to ensure it has a step_id
                if isinstance(step, dict):
                    step_id = step.get('step_id', f"step_{step.get('step', 0)}")
                else:
                    step_id = f"step_{step}"
                
                step_config = self.config.get(f'{step_id}_config', {})

                # Prepare step input
                step_input = self._prepare_step_input(step, input_data, results)

                try:
                    # Execute step with retry and await the result
                    step_result_ref = self._execute_step_with_retry.remote(
                        distributed_steps=self.distributed_steps,
                        step_id=step_id,
                        step_input=step_input,
                        max_retries=step_config.get('max_retries', self.config.get('max_retries', 2)),
                        retry_delay=step_config.get('retry_delay', self.config.get('retry_delay', 1.0)),
                        retry_backoff=step_config.get('retry_backoff', self.config.get('retry_backoff', 2.0))
                    )
                    step_result = await ray.get(step_result_ref)

                    # Store results
                    results[step_id] = step_result
                    self.state_manager.set_step_result(step_id, step_result)
                    self.state_manager.set_step_status(step_id, StepStatus.COMPLETED)

                    # Format output based on step configuration
                    step_output = {}
                    if isinstance(step, dict):
                        step_output = step.get('output', {})
                    
                    if step_output:
                        if isinstance(step_result, dict):
                            # Extract result content
                            result_content = step_result.get('result', step_result)
                            if isinstance(result_content, dict):
                                final_output = {
                                    'output': {
                                        **step_output,
                                        **result_content
                                    }
                                }
                            else:
                                final_output = {
                                    'output': {
                                        **step_output,
                                        'result': result_content
                                    }
                                }
                        else:
                            final_output = {
                                'output': {
                                    **step_output,
                                    'result': step_result
                                }
                            }
                    else:
                        final_output = step_result if isinstance(step_result, dict) else {'output': step_result}

                except Exception as e:
                    # Handle step failure
                    self.state_manager.set_step_status(step_id, StepStatus.FAILED)
                    self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
                    error_msg = f"Step {step_id} failed: {str(e)}"
                    logger.error(error_msg)
                    raise WorkflowExecutionError(error_msg)

            # Set final workflow status and return results
            self.state_manager.set_workflow_status(WorkflowStatus.COMPLETED)
            return final_output if final_output is not None else {'output': results}

        except Exception as e:
            # Handle workflow failure
            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)

    def validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate the input data for the workflow.
        
        Raises:
            ValueError: If required inputs are missing or input data is empty.
        """
        # Check if input_data is None or empty
        if not input_data:
            raise ValueError("Missing required inputs")

        # Get required input fields
        required_fields = self.get_required_input_fields()

        # Convert non-dictionary inputs to dictionary
        for key in required_fields:
            if key not in input_data or input_data[key] is None:
                # If the field is missing, create a default dictionary
                input_data[key] = {}
            
            # Ensure the input is a dictionary
            if not isinstance(input_data[key], dict):
                input_data[key] = {"value": input_data[key]}

        # Check for missing required inputs
        missing_inputs = [
            key for key in required_fields 
            if not input_data.get(key)
        ]

        # Raise error if any required inputs are missing
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {', '.join(missing_inputs)}")

    def _prepare_step_input(self, step: Dict[str, Any], input_data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for a step based on its configuration and previous results"""
        # Initialize input data
        processed_input = {}
        previous_results = previous_results or {}

        # Handle input references
        for input_key in step.get('input', []):
            if isinstance(input_key, str):
                # Check for workflow references
                if input_key.startswith('WORKFLOW.'):
                    try:
                        # Split reference into parts
                        ref_parts = input_key.split('.')
                        if len(ref_parts) < 2:
                            raise ValueError(f"Invalid workflow reference format: {input_key}")

                        # Normalize step identifier
                        step_ref = ref_parts[1]
                        step_id = f"step_{step_ref}" if not step_ref.startswith('step_') else step_ref

                        # Get result from previous steps
                        if step_id in previous_results:
                            result = previous_results[step_id]

                            # Handle nested references
                            for part in ref_parts[2:]:
                                if isinstance(result, dict):
                                    # Try to get nested value, fallback to 'result' key
                                    result = result.get(part, result.get('result', {}))
                                else:
                                    # If not a dict, use the value as is
                                    break

                            # Store the processed result
                            processed_input[input_key] = result
                        else:
                            # If step result not found, keep reference as placeholder
                            processed_input[input_key] = input_key
                            logger.warning(f"Step result not found for reference: {input_key}")
                    except Exception as e:
                        logger.warning(f"Error processing workflow reference {input_key}: {str(e)}")
                        processed_input[input_key] = input_key
                else:
                    # Handle direct input references
                    processed_input[input_key] = input_data.get(input_key, input_key)
            else:
                # Non-string inputs passed through as-is
                processed_input[input_key] = input_key

        # Add agent configuration
        agent_config = step.get('agent_config', {})
        if isinstance(agent_config, dict):
            processed_input.update(agent_config)

        # Log processed input for debugging
        logger.debug(f"Processed input for step {step.get('step', 'unknown')}: {processed_input}")

        return processed_input

class ResearchDistributedWorkflow(DistributedWorkflow):
    """A specialized distributed workflow for research-related tasks."""
    
    def __init__(
        self,
        workflow_config: Dict[str, Any] = None,
        config: Optional[Dict[str, Any]] = None,
        state_manager: Optional[WorkflowStateManager] = None
    ):
        super().__init__(workflow_config, config, state_manager)
        
        # Standardize workflow configuration
        workflow_config = workflow_config or {}
        if 'steps' in workflow_config:
            workflow_config['WORKFLOW'] = workflow_config.pop('steps')
        
        # Initialize workflow structure
        if 'WORKFLOW' not in workflow_config:
            workflow_config['WORKFLOW'] = {}
        
        # Initialize distributed steps
        self.distributed_steps = {}
        for step_id, step_config in workflow_config['WORKFLOW'].items():
            # Ensure step_config is a dictionary
            if not isinstance(step_config, dict):
                step_config = {'step': 1}
            
            # Get step number from config or extract from step_id
            if 'step' in step_config:
                step_num = step_config['step']
            else:
                try:
                    # Try to extract number from step_id (e.g., "step_1" -> 1)
                    step_num = int(step_id.split('_')[-1])
                except (ValueError, IndexError):
                    # If extraction fails, use position in workflow
                    step_num = len(self.distributed_steps) + 1
                step_config['step'] = step_num
            
            # Create appropriate step actor based on type
            step_type = step_config.get('type', 'research').lower()
            step_key = f"step_{step_num}"
            
            if step_type == 'document':
                step_actor = DocumentStep.remote(
                    step_id=step_key,
                    config=config.get(f"{step_key}_config", {})
                )
            elif step_type == 'implementation':
                step_actor = ImplementationStep.remote(
                    step_id=step_key,
                    config=config.get(f"{step_key}_config", {})
                )
            else:  # Default to research step
                step_actor = ResearchStep.remote(
                    step_id=step_key,
                    config=config.get(f"{step_key}_config", {})
                )
            
            self.distributed_steps[step_key] = step_actor
        
        # Set workflow attributes
        self.steps = self.distributed_steps
        self.required_fields = workflow_config.get('ENVIRONMENT', {}).get(
            'INPUT', {"STUDENT_NEEDS", "LANGUAGE", "TEMPLATE"}
        )
        self.default_status = WorkflowStatus.PENDING
        self.error_handling = {
            'missing_input_error': 'Missing or empty inputs',
            'missing_field_error': 'Missing required fields: {}',
            'handler': self._default_error_handler
        }
        
        # Store configuration
        self.original_workflow_config = workflow_config
        self.workflow_config = workflow_config

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow synchronously"""
        try:
            # Validate input data
            self.validate_input(input_data)
            
            # Initialize workflow state
            self.state_manager.initialize_workflow()
            self.state_manager.set_workflow_status(WorkflowStatus.RUNNING)
            
            # Execute steps in order
            results = {}
            final_output = None
            
            # Get workflow steps
            workflow_steps = self.workflow_config.get('WORKFLOW', {})
            if isinstance(workflow_steps, dict):
                workflow_steps = [
                    {'step_id': step_id, **step_config}
                    for step_id, step_config in sorted(
                        workflow_steps.items(),
                        key=lambda x: x[1].get('step', 0)
                    )
                ]
            
            # Execute each step
            for step in workflow_steps:
                step_num = step.get('step', 0)
                step_id = f"step_{step_num}"
                step_config = self.config.get(f'{step_id}_config', {})
                
                # Prepare step input
                step_input = self._prepare_step_input(step, input_data, results)
                
                try:
                    # Execute step with retry logic
                    step_result = self._execute_step_with_retry(
                        distributed_steps=self.distributed_steps,
                        step_id=step_id,
                        step_input=step_input,
                        max_retries=step_config.get('max_retries', 3),
                        retry_delay=step_config.get('retry_delay', 1.0),
                        retry_backoff=step_config.get('retry_backoff', 2.0)
                    )
                    
                    # Store results
                    results[step_id] = step_result
                    self.state_manager.set_step_result(step_id, step_result)
                    self.state_manager.set_step_status(step_id, StepStatus.COMPLETED)
                    
                    # Update final output
                    if isinstance(step_result, dict):
                        final_output = {
                            'output': step_result.get('result', step_result)
                        }
                    else:
                        final_output = {'output': step_result}
                    
                except Exception as e:
                    self.state_manager.set_step_status(step_id, StepStatus.FAILED)
                    self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
                    error_msg = f"Step {step_id} failed: {str(e)}"
                    logger.error(error_msg)
                    raise WorkflowExecutionError(error_msg)
            
            # Set success status
            self.state_manager.set_workflow_status(WorkflowStatus.COMPLETED)
            return final_output if final_output is not None else {'output': results}
            
        except Exception as e:
            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)

    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.execute, input_data
        )

    def validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate workflow input data"""
        if not input_data:
            error_msg = self.error_handling.get(
                'missing_input_error', 'Missing or empty inputs'
            )
            raise ValueError(error_msg)
        
        missing_fields = [
            field for field in self.required_fields
            if field not in input_data
        ]
        
        if missing_fields:
            error_msg = self.error_handling.get(
                'missing_field_error', 'Missing required fields: {}'
            ).format(missing_fields)
            raise ValueError(error_msg)

    def _prepare_step_input(
        self,
        step: Dict[str, Any],
        input_data: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare input for a workflow step"""
        processed_input = {}
        
        # Add original input data
        for field in self.required_fields:
            if field in input_data:
                processed_input[field] = input_data[field]
        
        # Add previous results
        if previous_results:
            processed_input['previous_results'] = previous_results
        
        # Add step-specific configuration
        if isinstance(step, dict):
            for key, value in step.items():
                if key not in ['step_id', 'step', 'output']:
                    processed_input[key] = value
        
        return processed_input

    def _default_error_handler(
        self,
        error: Exception,
        input_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle workflow errors"""
        error_msg = f"Workflow error: {str(error)}"
        if input_data:
            error_msg += f" with input: {input_data}"
        logger.error(error_msg)
        raise WorkflowExecutionError(error_msg)

    @classmethod
    def create_remote_workflow(
        cls,
        workflow_config: Dict[str, Any] = None,
        config: Optional[Dict[str, Any]] = None,
        state_manager: Optional[WorkflowStateManager] = None
    ):
        """Create a remote workflow instance"""
        return ray.remote(cls).remote(workflow_config, config, state_manager)

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
