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

    async def _execute_step_with_retry(
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
                
                # Get future and return it directly
                return step_actor.execute.remote(step_input)

            except Exception as e:
                last_exception = e
                attempt += 1
                if attempt < max_retries:
                    await asyncio.sleep(current_delay)  # Use await with asyncio.sleep
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

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
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
                    self.state_manager.set_step_status(step_id, StepStatus.SUCCESS)

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
            input_data = self.validate_input(input_data)

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

                # Get retry settings
                max_retries = step_config.get('max_retries', self.config.get('max_retries', 3))
                retry_delay = step_config.get('retry_delay', self.config.get('retry_delay', 0.1))
                retry_backoff = step_config.get('retry_backoff', self.config.get('retry_backoff', 2.0))

                # Prepare step input
                step_input = self._prepare_step_input(step, input_data, results)

                # Initialize retry counter
                retry_count = 0
                last_exception = None

                while retry_count < max_retries:
                    try:
                        # Execute step with retry logic
                        step_actor = self.distributed_steps.get(step_id)
                        if not step_actor:
                            raise ValueError(f"No step actor found for step_id: {step_id}")

                        # Execute step and get result
                        step_result_ref = step_actor.execute.remote(step_input)
                        loop = asyncio.get_running_loop()
                        
                        try:
                            step_result = await loop.run_in_executor(None, ray.get, step_result_ref)
                        except Exception as ray_error:
                            # Extract the original error message from the Ray error
                            error_msg = str(ray_error)
                            if "StepExecutionError" in error_msg:
                                # Extract the actual error message from the Ray error
                                error_msg = error_msg.split("StepExecutionError: ")[-1].split("\n")[0]
                            raise StepExecutionError(error_msg)

                        # Store results
                        results[step_id] = step_result
                        self.state_manager.set_step_result(step_id, step_result)
                        self.state_manager.set_step_status(step_id, StepStatus.SUCCESS)

                        # Update final output
                        if isinstance(step_result, dict):
                            final_output = step_result.get('result', step_result)
                        else:
                            final_output = step_result

                        # Success - break retry loop
                        break

                    except StepExecutionError as e:
                        last_exception = e
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            # Log retry attempt
                            logger.warning(
                                f"Step {step_id} failed (attempt {retry_count + 1}/{max_retries}). "
                                f"Retrying in {retry_delay} seconds..."
                            )
                            # Wait before retry with exponential backoff
                            await asyncio.sleep(retry_delay)
                            retry_delay *= retry_backoff
                        else:
                            # Max retries exceeded
                            self.state_manager.set_step_status(step_id, StepStatus.FAILED)
                            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
                            error_msg = f"Step {step_id} failed after {max_retries} retries: {str(last_exception)}"
                            logger.error(error_msg)
                            raise WorkflowExecutionError(error_msg)

            # Set final workflow status and return results
            self.state_manager.set_workflow_status(WorkflowStatus.COMPLETED)
            return {'output': final_output} if final_output is not None else {'output': results}

        except WorkflowExecutionError as e:
            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
            error_msg = str(e)
            if not error_msg.startswith("Workflow execution failed:") and getattr(e, "add_prefix", True):
                error_msg = f"Workflow execution failed: {error_msg}"
            logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)
        except Exception as e:
            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)

    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow input data.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            Validated and normalized input data
            
        Raises:
            WorkflowExecutionError: If required fields are missing
        """
        # If input_data is None, create an empty dictionary
        if input_data is None:
            input_data = {}

        # Debug print statements
        logger.debug(f"Required fields: {self.required_fields}")
        logger.debug(f"Input data: {input_data}")
    
        # Normalize input data to ensure it's a dictionary
        if not isinstance(input_data, dict):
            try:
                input_data = dict(input_data)
            except (TypeError, ValueError):
                input_data = {"value": input_data}

        # Validate each required field
        missing_fields = []
        for field in self.required_fields:
            if field not in input_data:
                missing_fields.append(field)

        logger.debug(f"Missing fields: {missing_fields}")

        # If any fields are missing, raise an error
        if missing_fields:
            error_msg = f"Missing required input: {missing_fields[0]}"
            logger.error(f"Workflow validation failed: {error_msg}")
            raise WorkflowExecutionError(error_msg, add_prefix=False)

        return input_data

    def _prepare_step_input(self, step: Dict[str, Any], input_data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for a workflow step"""
        # Ensure input_data is not None
        if input_data is None:
            input_data = {}

        # Normalize input_data to a dictionary if it's not already
        if not isinstance(input_data, dict):
            try:
                input_data = dict(input_data)
            except (TypeError, ValueError):
                input_data = {"value": input_data}

        processed_input = {}

        # Add original input data
        for field in self.required_fields:
            if field in input_data and input_data[field] is not None:
                processed_input[field] = input_data[field]

        # Add previous step results to the input
        if previous_results:
            processed_input['previous_results'] = previous_results

        # Add any step-specific configuration
        step_config = step.get('config', {})
        processed_input.update(step_config)

        # Ensure the input is not empty
        if not processed_input:
            processed_input = {"value": "Default step input"}

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
        
        # Convert list workflow to dict if needed
        if isinstance(workflow_config['WORKFLOW'], list):
            workflow_dict = {}
            for step in workflow_config['WORKFLOW']:
                step_num = step.get('step', len(workflow_dict) + 1)
                workflow_dict[f'step_{step_num}'] = step
            workflow_config['WORKFLOW'] = workflow_dict
        
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
            'missing_field_error': 'Missing required inputs',
            'handler': self._default_error_handler
        }
        
        # Store configuration
        self.original_workflow_config = workflow_config
        self.workflow_config = workflow_config

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow synchronously"""
        try:
            # Validate input data
            input_data = self.validate_input(input_data)
            
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
                    step_actor = self.distributed_steps.get(step_id)
                    if not step_actor:
                        raise ValueError(f"No step actor found for step_id: {step_id}")

                    # Execute step and get result
                    step_result_ref = step_actor.execute.remote(step_input)
                    loop = asyncio.get_running_loop()
                    
                    try:
                        step_result = await loop.run_in_executor(None, ray.get, step_result_ref)
                    except Exception as ray_error:
                        # Extract the original error message from the Ray error
                        error_msg = str(ray_error)
                        if "StepExecutionError" in error_msg:
                            # Extract the actual error message from the Ray error
                            error_msg = error_msg.split("StepExecutionError: ")[-1].split("\n")[0]
                        raise StepExecutionError(error_msg)

                    # Store results
                    results[step_id] = step_result
                    self.state_manager.set_step_result(step_id, step_result)
                    self.state_manager.set_step_status(step_id, StepStatus.SUCCESS)
                    
                    # Update final output
                    if isinstance(step_result, dict):
                        final_output = step_result.get('result', step_result)
                    else:
                        final_output = step_result
                    
                except StepExecutionError as e:
                    self.state_manager.set_step_status(step_id, StepStatus.FAILED)
                    self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
                    error_msg = f"Step {step_id} failed: {str(e)}"
                    logger.error(error_msg)
                    raise WorkflowExecutionError(error_msg)
            
            # Set success status
            self.state_manager.set_workflow_status(WorkflowStatus.COMPLETED)
            return {'output': final_output} if final_output is not None else {'output': results}
            
        except Exception as e:
            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
            error_msg = str(e)
            logger.error(f"Workflow execution failed: {error_msg}")
            raise WorkflowExecutionError(error_msg)

    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously"""
        try:
            # Validate input data
            input_data = self.validate_input(input_data)

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

                # Get retry settings
                max_retries = step_config.get('max_retries', self.config.get('max_retries', 3))
                retry_delay = step_config.get('retry_delay', self.config.get('retry_delay', 0.1))
                retry_backoff = step_config.get('retry_backoff', self.config.get('retry_backoff', 2.0))

                # Prepare step input
                step_input = self._prepare_step_input(step, input_data, results)

                # Initialize retry counter
                retry_count = 0
                last_exception = None

                while retry_count < max_retries:
                    try:
                        # Execute step with retry logic
                        step_actor = self.distributed_steps.get(step_id)
                        if not step_actor:
                            raise ValueError(f"No step actor found for step_id: {step_id}")

                        # Execute step and get result
                        step_result_ref = step_actor.execute.remote(step_input)
                        loop = asyncio.get_running_loop()
                        
                        try:
                            step_result = await loop.run_in_executor(None, ray.get, step_result_ref)
                        except Exception as ray_error:
                            # Extract the original error message from the Ray error
                            error_msg = str(ray_error)
                            if "StepExecutionError" in error_msg:
                                # Extract the actual error message from the Ray error
                                error_msg = error_msg.split("StepExecutionError: ")[-1].split("\n")[0]
                            raise StepExecutionError(error_msg)

                        # Store results
                        results[step_id] = step_result
                        self.state_manager.set_step_result(step_id, step_result)
                        self.state_manager.set_step_status(step_id, StepStatus.SUCCESS)

                        # Update final output
                        if isinstance(step_result, dict):
                            final_output = step_result.get('result', step_result)
                        else:
                            final_output = step_result

                        # Success - break retry loop
                        break

                    except StepExecutionError as e:
                        last_exception = e
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            # Log retry attempt
                            logger.warning(
                                f"Step {step_id} failed (attempt {retry_count + 1}/{max_retries}). "
                                f"Retrying in {retry_delay} seconds..."
                            )
                            # Wait before retry with exponential backoff
                            await asyncio.sleep(retry_delay)
                            retry_delay *= retry_backoff
                        else:
                            # Max retries exceeded
                            self.state_manager.set_step_status(step_id, StepStatus.FAILED)
                            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
                            error_msg = f"Step {step_id} failed after {max_retries} retries: {str(last_exception)}"
                            logger.error(error_msg)
                            raise WorkflowExecutionError(error_msg)

            # Set final workflow status and return results
            self.state_manager.set_workflow_status(WorkflowStatus.COMPLETED)
            return {'output': final_output} if final_output is not None else {'output': results}

        except WorkflowExecutionError as e:
            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
            error_msg = str(e)
            if not error_msg.startswith("Workflow execution failed:") and getattr(e, "add_prefix", True):
                error_msg = f"Workflow execution failed: {error_msg}"
            logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)
        except Exception as e:
            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)

    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow input data"""
        # If input_data is None, create an empty dictionary
        if input_data is None:
            input_data = {}

        # Debug print statements
        logger.debug(f"Required fields: {self.required_fields}")
        logger.debug(f"Input data: {input_data}")
    
        # Normalize input data to ensure it's a dictionary
        if not isinstance(input_data, dict):
            try:
                input_data = dict(input_data)
            except (TypeError, ValueError):
                input_data = {"value": input_data}

        # Validate each required field
        missing_fields = []
        for field in self.required_fields:
            if field not in input_data:
                missing_fields.append(field)

        logger.debug(f"Missing fields: {missing_fields}")

        # If any fields are missing, raise an error
        if missing_fields:
            error_msg = f"Missing required input: {missing_fields[0]}"
            logger.error(f"Workflow validation failed: {error_msg}")
            raise WorkflowExecutionError(error_msg, add_prefix=False)

        return input_data

    def _prepare_step_input(
        self,
        step: Dict[str, Any],
        input_data: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare input for a workflow step"""
        # Ensure input_data is not None
        if input_data is None:
            input_data = {}

        # Normalize input_data to a dictionary if it's not already
        if not isinstance(input_data, dict):
            try:
                input_data = dict(input_data)
            except (TypeError, ValueError):
                input_data = {"value": input_data}

        processed_input = {}

        # Add original input data
        for field in self.required_fields:
            if field in input_data and input_data[field] is not None:
                processed_input[field] = input_data[field]

        # Add previous step results to the input
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

class WorkflowExecutionError(Exception):
    """Custom exception for workflow execution errors"""
    def __init__(self, message, add_prefix=True):
        self.message = message
        self.add_prefix = add_prefix

class StepExecutionError(Exception):
    """Custom exception for step execution errors"""
    pass
