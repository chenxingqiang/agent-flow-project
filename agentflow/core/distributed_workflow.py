import ray
import logging
import time
from typing import Dict, Any, List, Callable, Optional, Union
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from .workflow import BaseWorkflow
from .workflow_state import WorkflowStateManager, StepStatus
from .retry import RetryConfig, with_retry
import asyncio
import copy

logger = logging.getLogger(__name__)

@ray.remote
class DistributedWorkflowStep:
    """Represents a distributed workflow step"""
    
    def __init__(self, step_config: Dict[str, Any]):
        """Initialize a distributed workflow step.
        
        Args:
            step_config: Configuration dictionary containing:
                - step_function: The function to execute (optional)
                - step_number: Step number in workflow
                - input: List of required input keys
                - output_type: Type of output produced
                - preprocessors: List of preprocessing functions (optional)
                - postprocessors: List of postprocessing functions (optional)
        """
        self.step_config = step_config
        self.step_number = step_config['step_number']
        # Use 'input' instead of 'input_keys'
        self.input_keys = step_config.get('input', [])
        self.output_type = step_config['output_type']
        
        # Provide a default step function if not specified
        self.step_function = step_config.get('step_function', self._default_step_function)
        self.preprocessors = step_config.get('preprocessors', [])
        self.postprocessors = step_config.get('postprocessors', [])
        
    def _default_step_function(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default step function that passes through input data."""
        return {
            'result': input_data,
            'step_number': self.step_number,
            'output_type': self.output_type
        }
        
    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step asynchronously."""
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Apply preprocessors
            processed_input = self._apply_preprocessors(input_data)
            
            # Execute step function
            if asyncio.iscoroutinefunction(self.step_function):
                result = await self.step_function(processed_input)
            else:
                result = self.step_function(processed_input)
                
            # Apply postprocessors
            processed_result = self._apply_postprocessors(result)
            
            return {
                "step_num": self.step_number,
                "result": processed_result,
                "metadata": {
                    "timestamp": time.time(),
                    "worker_id": ray.get_runtime_context().worker_id,
                    "output_type": self.output_type
                }
            }
            
        except Exception as e:
            raise ValueError(f"Step {self.step_number} execution failed: {str(e)}")
            
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step."""
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Apply preprocessors
            processed_input = self._apply_preprocessors(input_data)
            
            # Execute step function
            if self.step_function is None:
                raise ValueError(f"No step function defined for step {self.step_number}")
                
            result = self.step_function(processed_input)
            
            # Apply postprocessors
            processed_result = self._apply_postprocessors(result)
            
            return {
                "step_num": self.step_number,
                "result": processed_result,
                "metadata": {
                    "timestamp": time.time(),
                    "worker_id": ray.get_runtime_context().get_worker_id(),
                    "output_type": self.output_type
                }
            }
            
        except Exception as e:
            # Wrap the exception to ensure it propagates correctly through Ray
            raise ValueError(f"Step {self.step_number} execution failed: {str(e)}")
        
    def _validate_input(self, input_data: Dict[str, Any]):
        """Validate input data against required keys."""
        missing_keys = [key for key in self.input_keys if key not in input_data]
        if missing_keys:
            raise ValueError(f"Missing required inputs: {', '.join(missing_keys)}")
            
    def _apply_preprocessors(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all preprocessors in sequence."""
        result = input_data.copy()
        for preprocessor in self.preprocessors:
            result = preprocessor(result)
        return result
        
    def _apply_postprocessors(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all postprocessors in sequence."""
        processed = result.copy()
        for postprocessor in self.postprocessors:
            processed = postprocessor(processed)
        return processed

class DistributedWorkflow(BaseWorkflow, ABC):
    """
    Advanced distributed workflow management system
    
    Supports:
    - Multi-step workflows
    - Distributed execution
    - Dynamic input preparation
    - Flexible step-to-step data passing
    - Robust error handling
    """
    
    def __init__(self, config: Dict[str, Any], workflow_def: Dict[str, Any]):
        """
        Initialize the distributed workflow with configuration and workflow definition.
        
        Args:
            config (Dict[str, Any]): Configuration for the workflow
            workflow_def (Dict[str, Any]): Workflow definition
        """
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(config.get('logging_level', logging.INFO))
        
        # Deep copy to prevent modification of original input
        self.config = copy.deepcopy(config)
        self.workflow_def = copy.deepcopy(workflow_def)
        
        # Add required_fields attribute
        self.required_fields = []
        
        # Ensure WORKFLOW key exists and is a list
        if 'WORKFLOW' not in self.workflow_def:
            self.workflow_def['WORKFLOW'] = []
        
        # Validate workflow definition
        if not isinstance(self.workflow_def['WORKFLOW'], list):
            raise ValueError("Workflow definition must contain a list of workflow steps")
        
        # Ensure each step has required attributes
        for step in self.workflow_def['WORKFLOW']:
            if 'step' not in step:
                raise ValueError(f"Invalid workflow step: missing 'step' attribute - {step}")
            
            # Ensure step configuration exists
            step_config_key = f'step_{step["step"]}_config'
            if step_config_key not in self.config:
                self.config[step_config_key] = {
                    'preprocessors': [],
                    'postprocessors': []
                }
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # Initialize distributed steps
        self.distributed_steps = {}
        for step_def in self.workflow_def['WORKFLOW']:
            step_num = step_def['step']
            
            # Create distributed step actor
            @ray.remote
            class DistributedStep:
                async def execute(self, input_data):
                    # Default implementation, can be overridden
                    return {'result': input_data, 'step_num': step_num}
            
            self.distributed_steps[step_num] = DistributedStep.remote()
    
    def _get_step_function(self, step: Dict[str, Any]) -> Callable:
        """Default implementation of step function."""
        return lambda x: x

    def execute_step(self, step_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default implementation of step execution."""
        return input_data

    def process_step(self, step_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default implementation of step processing."""
        return input_data
        
    def _prepare_step_input(self, step_def: Dict[str, Any], input_data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input for a specific workflow step.
        
        Args:
            step_def (Dict[str, Any]): Definition of the current workflow step
            input_data (Dict[str, Any]): Current input data
            previous_results (Dict[str, Any]): Results from previous steps
        
        Returns:
            Dict[str, Any]: Prepared input for the current step
        """
        # Create a copy of input data to avoid modifying the original
        step_input = input_data.copy()
        
        # Handle workflow references in input
        if step_def.get('input', []):
            for input_ref in step_def['input']:
                if input_ref.startswith('WORKFLOW.'):
                    # Extract step number from reference
                    ref_step_num = int(input_ref.split('.')[1])
                    
                    # Find the corresponding previous result
                    previous_result = None
                    for key, result in previous_results.items():
                        if isinstance(key, int) and key == ref_step_num:
                            previous_result = result
                            break
                        elif isinstance(key, str) and (f'step_{ref_step_num}' in key or str(ref_step_num) == key):
                            previous_result = result
                            break
                    
                    if previous_result is not None:
                        # Prefer 'result' key if available
                        result_to_add = previous_result.get('result', previous_result)
                        
                        # Add multiple representations for compatibility
                        step_input['previous_step_result'] = previous_result
                        step_input[f'step_{ref_step_num}_result'] = result_to_add
                        step_input['WORKFLOW.1'] = result_to_add
                        
                        # Ensure original keys are preserved
                        if isinstance(previous_result, dict):
                            for key, value in previous_result.items():
                                if key not in step_input:
                                    step_input[key] = value
                
                # Validate required inputs
                if input_ref not in step_input and not input_ref.startswith('WORKFLOW.'):
                    raise ValueError(f"Missing or empty input: {input_ref}")
        
        return step_input
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps

        Args:
            input_data (Dict[str, Any]): Initial input data

        Returns:
            Dict[int, Dict[str, Any]]: Results from each executed step
        """
        # Set max iterations from config, default to total number of workflow steps
        max_iterations = self.config.get('max_iterations', len(self.workflow_def['WORKFLOW']))
        max_concurrent = self.config.get('max_concurrent_steps', 1)
        
        # Initialize results dictionary and semaphore for concurrency control
        results = {}
        current_input = input_data.copy()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Group steps by their dependencies
        step_groups = self._group_steps_by_dependencies()
        
        # Track if any step failed
        workflow_failed = False
        workflow_error = None
        
        # Process each group of steps
        for step_group in step_groups:
            # Create tasks for all steps in the current group
            tasks = []
            for step_def in step_group:
                if len(results) >= max_iterations:
                    break
                    
                step_num = step_def['step']
                
                # Skip if workflow has already failed and not configured to continue
                if workflow_failed and not self.config.get('continue_on_error', False):
                    break
                
                # Create a task for the step
                task = asyncio.create_task(
                    self._execute_step_with_semaphore(
                        semaphore,
                        step_num,
                        step_def,
                        current_input,
                        results
                    )
                )
                tasks.append(task)
            
            if tasks:
                try:
                    # Wait for all tasks in the current group to complete
                    step_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results and handle any exceptions
                    for step_def, result in zip(step_group, step_results):
                        step_num = step_def['step']
                        if isinstance(result, Exception):
                            self.logger.error(f"Error executing step {step_num}: {result}")
                            
                            # If not continuing on error, raise the first error
                            if not self.config.get('continue_on_error', False):
                                workflow_failed = True
                                workflow_error = result
                                break
                        else:
                            results[step_num] = result
                            current_input.update(result)
                    
                    # If workflow failed, break out of step groups
                    if workflow_failed:
                        break
                
                except Exception as e:
                    # Unexpected error in task gathering
                    workflow_failed = True
                    workflow_error = e
                    break
        
        # If workflow failed, raise the first error
        if workflow_failed and workflow_error:
            raise ValueError(str(workflow_error))
        
        return results
        
    def execute_async(self, 
                      input_data: Dict[str, Any], 
                      max_concurrent_steps: int = 2) -> ray.ObjectRef:
        """
        Execute workflow steps asynchronously
        
        Args:
            input_data: Initial input data
            max_concurrent_steps: Maximum number of steps to run concurrently
        
        Returns:
            Ray object reference for async execution
        """
        @ray.remote
        def async_workflow_executor(workflow_instance, input_data):
            return workflow_instance.execute(input_data)
        
        return async_workflow_executor.remote(self, input_data)

class ResearchDistributedWorkflow(DistributedWorkflow):
    """
    Distributed research workflow implementation
    """
    
    def __init__(self, config: Dict[str, Any], workflow_def: Dict[str, Any]):
        """
        Initialize the distributed workflow with configuration and workflow definition.
        
        Args:
            config (Dict[str, Any]): Configuration for the workflow
            workflow_def (Dict[str, Any]): Workflow definition
        """
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(config.get('logging_level', logging.INFO))
        
        # Deep copy to prevent modification of original input
        self.config = copy.deepcopy(config)
        self.workflow_def = copy.deepcopy(workflow_def)
        
        # Add required_fields attribute
        self.required_fields = []
        
        # Ensure WORKFLOW key exists and is a list
        if 'WORKFLOW' not in self.workflow_def:
            self.workflow_def['WORKFLOW'] = []
        
        # Validate workflow definition
        if not isinstance(self.workflow_def['WORKFLOW'], list):
            raise ValueError("Workflow definition must contain a list of workflow steps")
        
        # Ensure each step has required attributes
        for step in self.workflow_def['WORKFLOW']:
            if 'step' not in step:
                raise ValueError(f"Invalid workflow step: missing 'step' attribute - {step}")
            
            # Ensure step configuration exists
            step_config_key = f'step_{step["step"]}_config'
            if step_config_key not in self.config:
                self.config[step_config_key] = {
                    'preprocessors': [],
                    'postprocessors': []
                }
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # Initialize distributed steps
        self.distributed_steps = {}
        for step_def in self.workflow_def['WORKFLOW']:
            step_num = step_def['step']
            
            # Create distributed step actor
            @ray.remote
            class DistributedStep:
                async def execute(self, input_data):
                    # Default implementation, can be overridden
                    return {'result': input_data, 'step_num': step_num}
            
            self.distributed_steps[step_num] = DistributedStep.remote()
        
        self.state_manager = WorkflowStateManager()
        self.retry_config = RetryConfig(
            max_retries=config.get('max_retries', 3),
            delay=config.get('retry_delay', 1.0),
            backoff_factor=config.get('retry_backoff', 2.0)
        )
        
    def _initialize_steps(self) -> Dict[int, Any]:
        """Initialize workflow steps from workflow definition"""
        steps = {}
        for step_def in self.workflow_def.get('WORKFLOW', []):
            step_number = step_def.get('step', len(steps) + 1)
            step_config = {
                'input': step_def.get('input', []),
                'output_type': step_def.get('output', {}).get('type', 'default'),
                'step_number': step_number,
                **self.config.get(f'step_{step_number}_config', {})
            }
            steps[step_number] = DistributedWorkflowStep.remote(step_config)
        return steps
        
    @with_retry()
    async def execute_step(self, step_num: int, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step with retry"""
        step_actor = self.distributed_steps.get(step_num)
        if not step_actor:
            raise ValueError(f"Step {step_num} not initialized")
        
        # Ensure step is initialized before starting
        if step_num not in self.state_manager.states:
            self.state_manager.initialize_step(step_num)
        
        # Check if we've exceeded max retries
        retry_count = self.state_manager.get_step_retry_count(step_num)
        max_retries = self.retry_config.max_retries
        
        if retry_count > max_retries:
            raise ValueError(f"Step {step_num} exceeded maximum retry attempts")
        
        # If this is a retry, increment the retry count
        if retry_count > 0:
            self.state_manager.retry_step(step_num)
        
        self.state_manager.start_step(step_num)
        
        try:
            # Apply preprocessors before execution
            step_config = self.config.get(f'step_{step_num}_config', {})
            preprocessors = step_config.get('preprocessors', [])
            processed_input = step_input.copy()
            
            # Apply preprocessors asynchronously if they are async functions
            for preprocessor in preprocessors:
                if asyncio.iscoroutinefunction(preprocessor):
                    processed_input = await preprocessor(processed_input)
                else:
                    processed_input = preprocessor(processed_input)

            # Execute the step
            try:
                # Get the future from the remote call
                future = step_actor.execute.remote(processed_input)
                
                # Create an asyncio task to run ray.get in the background
                loop = asyncio.get_running_loop()
                timeout = self.config.get('step_timeout', 60)  # Default 60 seconds timeout
                
                async def get_result():
                    return await loop.run_in_executor(None, ray.get, future)
                
                try:
                    result = await asyncio.wait_for(get_result(), timeout=timeout)
                except asyncio.TimeoutError:
                    # Cancel the Ray task if it times out
                    ray.cancel(future)
                    self.state_manager.fail_step(step_num, "Step execution timed out")
                    raise ValueError(f"Step {step_num} execution timed out")
                
            except Exception as e:
                self.state_manager.fail_step(step_num, str(e))
                raise ValueError(f"Step {step_num} execution failed: {str(e)}")
            
            # Apply postprocessors if any
            postprocessors = step_config.get('postprocessors', [])
            processed_result = result.copy() if isinstance(result, dict) else {'result': result}
            
            # Apply postprocessors asynchronously if they are async functions
            for postprocessor in postprocessors:
                if asyncio.iscoroutinefunction(postprocessor):
                    processed_result = await postprocessor(processed_result)
                else:
                    processed_result = postprocessor(processed_result)
                
            # Add metadata to the result
            if isinstance(processed_result, dict):
                if 'metadata' not in processed_result:
                    processed_result['metadata'] = {}
                processed_result['metadata'].update({
                    'timestamp': time.time(),
                    'step_num': step_num
                })
                
            self.state_manager.complete_step(step_num, processed_result)
            return processed_result
            
        except Exception as e:
            # Increment retry count on failure
            self.state_manager.increment_step_retry_count(step_num)
            
            # If we've reached max retries, fail the step
            retry_count = self.state_manager.get_step_retry_count(step_num)
            if retry_count > max_retries:
                self.state_manager.fail_step(step_num, str(e))
            
            self.logger.error(f"Step {step_num} execution failed: {str(e)}")
            raise
            
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps

        Args:
            input_data (Dict[str, Any]): Initial input data

        Returns:
            Dict[int, Dict[str, Any]]: Results from each executed step
        """
        # Set max iterations from config, default to total number of workflow steps
        max_iterations = self.config.get('max_iterations', len(self.workflow_def['WORKFLOW']))
        max_concurrent = self.config.get('max_concurrent_steps', 1)
        
        # Initialize results dictionary and semaphore for concurrency control
        results = {}
        current_input = input_data.copy()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Group steps by their dependencies
        step_groups = self._group_steps_by_dependencies()
        
        # Track if any step failed
        workflow_failed = False
        workflow_error = None
        
        # Process each group of steps
        for step_group in step_groups:
            # Create tasks for all steps in the current group
            tasks = []
            for step_def in step_group:
                if len(results) >= max_iterations:
                    break
                    
                step_num = step_def['step']
                
                # Skip if workflow has already failed and not configured to continue
                if workflow_failed and not self.config.get('continue_on_error', False):
                    break
                
                # Create a task for the step
                task = asyncio.create_task(
                    self._execute_step_with_semaphore(
                        semaphore,
                        step_num,
                        step_def,
                        current_input,
                        results
                    )
                )
                tasks.append(task)
            
            if tasks:
                try:
                    # Wait for all tasks in the current group to complete
                    step_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results and handle any exceptions
                    for step_def, result in zip(step_group, step_results):
                        step_num = step_def['step']
                        if isinstance(result, Exception):
                            self.logger.error(f"Error executing step {step_num}: {result}")
                            
                            # If not continuing on error, raise the first error
                            if not self.config.get('continue_on_error', False):
                                workflow_failed = True
                                workflow_error = result
                                break
                        else:
                            results[step_num] = result
                            current_input.update(result)
                    
                    # If workflow failed, break out of step groups
                    if workflow_failed:
                        break
                
                except Exception as e:
                    # Unexpected error in task gathering
                    workflow_failed = True
                    workflow_error = e
                    break
        
        # If workflow failed, raise the first error
        if workflow_failed and workflow_error:
            raise ValueError(str(workflow_error))
        
        return results
        
    async def _execute_step_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        step_num: int, 
        step_def: Dict[str, Any], 
        input_data: Dict[str, Any], 
        results: Dict[int, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step with concurrency control and error handling

        Args:
            semaphore (asyncio.Semaphore): Concurrency control semaphore
            step_num (int): Step number
            step_def (Dict[str, Any]): Step definition
            input_data (Dict[str, Any]): Input data for the step
            results (Dict[int, Any]): Previous step results

        Returns:
            Dict[str, Any]: Step execution result
        """
        # Ensure step state is initialized
        self.state_manager.initialize_step_state(step_num)
        
        # Retrieve step-specific configuration
        step_config = self.config.get(f'step_{step_num}_config', {})
        
        # Prepare input data
        step_input = input_data.copy()
        
        # Track if preprocessors were applied
        preprocessors_applied = False
        
        # Apply preprocessors
        preprocessors = step_config.get('preprocessors', [])
        for preprocessor in preprocessors:
            try:
                # Check if preprocessor is async
                if asyncio.iscoroutinefunction(preprocessor):
                    step_input = await preprocessor(step_input)
                else:
                    step_input = preprocessor(step_input)
                preprocessors_applied = True
            except Exception as e:
                self.logger.error(f"Preprocessor error in step {step_num}: {e}")
                self.state_manager.fail_step(step_num, str(e))
                raise
        
        # Retry mechanism
        max_retries = self.config.get('max_retries', 3)
        retry_delay = self.config.get('retry_delay', 1.0)
        retry_backoff = self.config.get('retry_backoff', 2.0)
        
        original_error = None
        
        async with semaphore:
            for attempt in range(max_retries + 1):
                try:
                    # Set step timeout
                    step_timeout = step_config.get('step_timeout', self.config.get('step_timeout', None))
                    
                    # Get the actual step object
                    step_obj = self.distributed_steps[step_num]
                    
                    # Determine if this is a Ray actor
                    is_ray_actor = hasattr(step_obj, 'execute') and hasattr(step_obj.execute, 'remote')
                    
                    # Execute the step
                    try:
                        if is_ray_actor:
                            # Use .remote() for Ray actors
                            if step_timeout:
                                # Use asyncio.wait_for to implement timeout
                                step_result = await asyncio.wait_for(
                                    step_obj.execute.remote(step_input), 
                                    timeout=step_timeout
                                )
                            else:
                                # Execute without timeout
                                step_result = await step_obj.execute.remote(step_input)
                        else:
                            # Use regular async method for non-Ray actors
                            if step_timeout:
                                # Use asyncio.wait_for to implement timeout
                                step_result = await asyncio.wait_for(
                                    step_obj.execute(step_input), 
                                    timeout=step_timeout
                                )
                            else:
                                # Execute without timeout
                                step_result = await step_obj.execute(step_input)
                    except asyncio.TimeoutError:
                        raise ValueError(f"Step {step_num} execution timed out")
                    
                    # Apply postprocessors
                    postprocessors = step_config.get('postprocessors', [])
                    for postprocessor in postprocessors:
                        try:
                            # Check if postprocessor is async
                            if asyncio.iscoroutinefunction(postprocessor):
                                step_result = await postprocessor(step_result)
                            else:
                                step_result = postprocessor(step_result)
                        except Exception as e:
                            self.logger.error(f"Postprocessor error in step {step_num}: {e}")
                            self.state_manager.fail_step(step_num, str(e))
                            raise
                    
                    # Ensure result has metadata
                    if 'metadata' not in step_result:
                        step_result['metadata'] = {}
                    
                    # Add metadata to result
                    step_result['metadata'].update({
                        'timestamp': time.time(),
                        'step_num': step_num,
                        'preprocessed': preprocessors_applied  # Use the flag
                    })
                    
                    # Add preprocessed flag directly to result if preprocessors were applied
                    if preprocessors_applied:
                        step_result['preprocessed'] = True
                    
                    # Increment step success count
                    self.state_manager.increment_step_success_count(step_num)
                    self.state_manager.complete_step(step_num, step_result)
                    
                    return step_result
                
                except Exception as e:
                    # Log the error
                    self.logger.error(f"Step {step_num} execution failed: {e}")
                    
                    # Store the original error for final error reporting
                    if original_error is None:
                        original_error = e
                    
                    # Increment retry count only if max retries has not been reached
                    if attempt < max_retries:
                        self.state_manager.increment_step_retry_count(step_num)
                    
                    # Check if max retries are exhausted
                    if attempt >= max_retries:
                        # Mark step as failed
                        self.state_manager.fail_step(step_num, str(original_error))
                        
                        # Raise a custom error that includes the original error context
                        raise ValueError(f"Step {step_num} execution failed: {original_error}")
            
            # This should never be reached, but added for completeness
            raise ValueError(f"Step {step_num} exceeded maximum retry attempts")
        
    def _group_steps_by_dependencies(self) -> List[List[Dict[str, Any]]]:
        """Group steps that can be executed in parallel based on their dependencies"""
        steps = self.workflow_def['WORKFLOW']
        groups = []
        remaining_steps = steps.copy()
        
        while remaining_steps:
            current_group = []
            next_remaining = []
            
            for step in remaining_steps:
                # Check if all dependencies are satisfied
                dependencies = self._get_step_dependencies(step)
                if all(dep in [s['step'] for group in groups for s in group] for dep in dependencies):
                    current_group.append(step)
                else:
                    next_remaining.append(step)
            
            if not current_group:
                # If no steps can be added, there might be a circular dependency
                raise ValueError("Circular dependency detected in workflow steps")
                
            groups.append(current_group)
            remaining_steps = next_remaining
            
        return groups
        
    def _get_step_dependencies(self, step: Dict[str, Any]) -> List[int]:
        """Get the step numbers that this step depends on"""
        dependencies = []
        input_keys = step.get('input', [])
        
        for key in input_keys:
            # Check for workflow step reference (e.g., "WORKFLOW.1")
            if isinstance(key, str) and key.startswith('WORKFLOW.'):
                try:
                    step_num = int(key.split('.')[1])
                    dependencies.append(step_num)
                except (IndexError, ValueError):
                    continue
                    
        return dependencies
        
    def validate_step_input(self, step_num: int, input_data: Dict[str, Any]):
        """Validate input data for a specific step"""
        step_def = next((s for s in self.workflow_def.get('WORKFLOW', []) if s.get('step') == step_num), None)
        if not step_def:
            return
        
        # Check required inputs for the step
        required_inputs = step_def.get('input', [])
        for input_key in required_inputs:
            # Skip workflow references
            if input_key.startswith('WORKFLOW.'):
                continue
            
            if input_key not in input_data or not input_data[input_key]:
                raise ValueError(f"Missing or empty inputs: {input_key}")
        
    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow asynchronously."""
        if not input_data:
            raise ValueError("Empty input data")

        results = {}
        futures = {}

        # First, submit all remote tasks
        for step_num, step_actor in self.distributed_steps.items():
            step_def = next((s for s in self.workflow_def.get('WORKFLOW', []) if s.get('step') == step_num), None)
            if not step_def:
                raise ValueError(f"Step definition for step {step_num} not found")

            step_input = self._prepare_step_input(
                step_def,
                input_data,
                results
            )

            # Apply preprocessors before execution
            step_config = self.config.get(f'step_{step_num}_config', {})
            preprocessors = step_config.get('preprocessors', [])
            for preprocessor in preprocessors:
                step_input = preprocessor(step_input)

            # Submit remote task
            futures[step_num] = step_actor.execute.remote(step_input)

        # Await and process results
        for step_num, future in futures.items():
            try:
                # Await the result
                result = ray.get(future)
                
                # Apply postprocessors
                step_config = self.config.get(f'step_{step_num}_config', {})
                postprocessors = step_config.get('postprocessors', [])
                for postprocessor in postprocessors:
                    result = postprocessor(result)

                results[step_num] = result

            except Exception as e:
                raise ValueError(f"Step {step_num} execution failed: {str(e)}")

        return results
        
    def _prepare_step_input(self, step_config: Dict[str, Any], 
                           input_data: Dict[str, Any], 
                           previous_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare input for a specific workflow step.
        
        Args:
            step_config (Dict[str, Any]): Definition of the current workflow step
            input_data (Dict[str, Any]): Current input data
            previous_results (Dict[int, Dict[str, Any]]): Results from previous steps
        
        Returns:
            Dict[str, Any]: Prepared input for the current step
        """
        # Create a copy of input data to avoid modifying the original
        step_input = input_data.copy()
        
        # Handle workflow references in input
        if step_config.get('input', []):
            for input_ref in step_config['input']:
                if input_ref.startswith('WORKFLOW.'):
                    # Extract step number from reference
                    ref_step_num = int(input_ref.split('.')[1])
                    
                    # Find the corresponding previous result
                    previous_result = None
                    for key, result in previous_results.items():
                        if isinstance(key, int) and key == ref_step_num:
                            previous_result = result
                            break
                        elif isinstance(key, str) and (f'step_{ref_step_num}' in key or str(ref_step_num) == key):
                            previous_result = result
                            break
                    
                    if previous_result is not None:
                        # Prefer 'result' key if available
                        result_to_add = previous_result.get('result', previous_result)
                        
                        # Add multiple representations for compatibility
                        step_input['previous_step_result'] = previous_result
                        step_input[f'step_{ref_step_num}_result'] = result_to_add
                        step_input['WORKFLOW.1'] = result_to_add
                        
                        # Ensure original keys are preserved
                        if isinstance(previous_result, dict):
                            for key, value in previous_result.items():
                                if key not in step_input:
                                    step_input[key] = value
                
                # Validate required inputs
                if input_ref not in step_input and not input_ref.startswith('WORKFLOW.'):
                    raise ValueError(f"Missing or empty input: {input_ref}")
        
        return step_input
        
    def _get_step_function(self, step: Dict[str, Any]) -> Callable:
        """
        Get the function to execute for a specific step
        
        Args:
            step: Step definition
        
        Returns:
            Callable step function
        """
        step_type = step['output']['type']
        return lambda input_data: self.process_step(step_type, input_data)
    
    def process_step(self, step_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a specific step type
        
        Args:
            step_type: Type of workflow step
            input_data: Input data for the step
        
        Returns:
            Processed step results
        """
        if step_type == 'research':
            return self._process_research(input_data)
        elif step_type == 'document':
            return self._process_document(input_data)
        else:
            raise ValueError(f"Unknown step type: {step_type}")
    
    def _process_research(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Distributed research processing
        
        Args:
            input_data: Input data for research
        
        Returns:
            Research results
        """
        topic = input_data.get('research_topic', '')
        if not topic:
            raise ValueError("Empty research topic")
        
        # Simulate distributed research processing
        return {
            "result": f"Distributed research findings for {topic}",
            "summary": f"Comprehensive analysis of {topic}",
            "recommendations": ["Recommendation 1", "Recommendation 2"]
        }
    
    def _process_document(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Distributed document generation
        
        Args:
            input_data: Input data for document
        
        Returns:
            Generated document
        """
        research_data = input_data.get('research_data', {})
        if not research_data:
            raise ValueError("Empty research data for document generation")
        
        return {
            "content": f"Distributed document content based on {research_data}",
            "format": "markdown"
        }
