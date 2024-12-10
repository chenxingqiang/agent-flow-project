import logging
import asyncio
import random
import time
import re

# Configure logging for the entire module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a module-level logger
logger = logging.getLogger(__name__)

import ray
from typing import Dict, Any, List, Callable, Optional, Union, Set  
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from .workflow import BaseWorkflow
from .workflow_state import WorkflowStateManager, StepStatus, WorkflowStatus
from .retry import RetryConfig, with_retry
import copy
import json

class WorkflowExecutionError(Exception):
    """Exception raised for errors during workflow execution"""
    pass

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
        self.step_number = step_config.get('step_number', step_config.get('step', 1))
        # Use 'input' instead of 'input_keys'
        self.input_keys = step_config.get('input', [])
        self.output_type = step_config.get('output_type', {})
        
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
            processed_input = input_data
            for preprocessor in self.preprocessors:
                if asyncio.iscoroutinefunction(preprocessor):
                    processed_input = await preprocessor(processed_input)
                else:
                    processed_input = preprocessor(processed_input)
                processed_input['preprocessed'] = True
            
            # Execute step function
            if asyncio.iscoroutinefunction(self.step_function):
                result = await self.step_function(processed_input)
            else:
                result = self.step_function(processed_input)
                
            # Apply postprocessors
            processed_result = result
            for postprocessor in self.postprocessors:
                if asyncio.iscoroutinefunction(postprocessor):
                    processed_result = await postprocessor(processed_result)
                else:
                    processed_result = postprocessor(processed_result)
                processed_result['postprocessed'] = True
            
            # Ensure result is a dictionary with proper metadata
            if not isinstance(processed_result, dict):
                processed_result = {'result': processed_result}
            
            # Add metadata
            metadata = {
                "timestamp": time.time(),
            }
            
            if 'preprocessed' in processed_input:
                metadata['preprocessed'] = True
            if 'postprocessed' in processed_result:
                metadata['postprocessed'] = True
            
            # Ensure format compliance if specified in workflow config
            if hasattr(self, 'workflow_config') and self.workflow_config:
                step_config = next((step for step in self.workflow_config.get('WORKFLOW', []) 
                                if step.get('step') == self.step_number), None)
                if step_config and 'output' in step_config:
                    output_format = step_config['output'].get('format')
                    if output_format:
                        if 'result' in processed_result and isinstance(processed_result['result'], dict):
                            processed_result['result']['format'] = output_format
                        elif 'output' in processed_result and isinstance(processed_result['output'], dict):
                            processed_result['output']['format'] = output_format
            
            # Combine all metadata and results
            full_result = {
                **processed_result,
                **metadata
            }
            
            return full_result
            
        except Exception as e:
            raise ValueError(f"Step {self.step_number} failed after {str(e)}")
            
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step synchronously."""
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Apply preprocessors
            processed_input = input_data
            for preprocessor in self.preprocessors:
                processed_input = preprocessor(processed_input)
                processed_input['preprocessed'] = True
            
            # Execute step function
            result = self.step_function(processed_input)
                
            # Apply postprocessors
            processed_result = result
            for postprocessor in self.postprocessors:
                processed_result = postprocessor(processed_result)
                processed_result['postprocessed'] = True
            
            # Ensure result is a dictionary with proper metadata
            if not isinstance(processed_result, dict):
                processed_result = {'result': processed_result}
            
            # Add metadata
            metadata = {
                "timestamp": time.time(),
            }
            
            if 'preprocessed' in processed_input:
                metadata['preprocessed'] = True
            if 'postprocessed' in processed_result:
                metadata['postprocessed'] = True
            
            return {
                "step_num": self.step_number,
                "result": processed_result,
                "metadata": metadata
            }
            
        except Exception as e:
            raise ValueError(f"Step {self.step_number} failed after {str(e)}")
        
    def _validate_input(self, input_data: Dict[str, Any]):
        """Validate input data against required keys."""
        missing_keys = [key for key in self.input_keys if key not in input_data]
        if missing_keys:
            raise ValueError(f"Missing required inputs: {', '.join(missing_keys)}")
            
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
    
    def __init__(
        self, 
        workflow_config: Dict[str, Any], 
        config: Optional[Dict[str, Any]] = None, 
        state_manager: Optional[WorkflowStateManager] = None
    ):
        # Validate required inputs during initialization
        self._validate_workflow_config(workflow_config)

        self.workflow_config = workflow_config
        self.config = config or {}
        self.distributed_steps = {}
        self.state_manager = state_manager or WorkflowStateManager()
        
        # Extract required fields from workflow steps
        self.required_fields = []
        for step in workflow_config.get('WORKFLOW', []):
            # Collect input fields from each step, excluding workflow references
            self.required_fields.extend([
                field for field in step.get('input', []) 
                if not field.startswith('WORKFLOW.')
            ])
        
        # Remove duplicates while preserving order
        self.required_fields = list(dict.fromkeys(self.required_fields))
        
        # Initialize step configurations
        self._initialize_step_configs()
        
    def _get_step_function(self, step_def: Dict[str, Any]) -> Callable:
        """Get the step function from the step definition

        Args:
            step_def (Dict[str, Any]): Step definition

        Returns:
            Callable: Step function
        """
        step_id = step_def.get('id')
        step_type = step_def.get('type', 'default')
        
        # Get the step function from the registry
        step_func = self.step_registry.get(step_type)
        if not step_func:
            # If no specific function is found, use the default function
            step_func = self.step_registry['default']
        
        return step_func

    def _initialize_steps(self):
        """Initialize workflow steps with Ray actors"""
        if not self.workflow_config:
            raise ValueError("No workflow configuration provided")

        # Get workflow steps
        workflow_steps = self.workflow_config.get('WORKFLOW', [])
        if not workflow_steps:
            raise ValueError("No workflow steps found in configuration")

        # Initialize distributed steps
        for step in workflow_steps:
            step_id = f"step_{step['step']}"
            step_type = step.get('type', 'research')
            step_config = {
                'type': step_type,
                'name': step.get('name', f'Step {step["step"]}'),
                'description': step.get('description', ''),
                'input': step.get('input', []),
                'output': step.get('output', {}),
                'agent_config': step.get('agent_config', {})
            }
            step_config.update(self.config.get(f'{step_id}_config', {}))

            # Create Ray actor for this step
            try:
                step_actor = ray.remote(ResearchStep).remote(
                    step_id=step_id,
                    config=step_config
                )
                self.distributed_steps[step_id] = step_actor
                logger.debug(f"Initialized step actor for {step_id}")
            except Exception as e:
                logger.error(f"Failed to initialize step {step_id}: {e}")
                raise WorkflowExecutionError(f"Step initialization failed: {e}")

    def execute_step(self, step_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default implementation of step execution."""
        return input_data

    def process_step(self, step_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a specific step type
        
        Args:
            step_type: Type of workflow step
            input_data: Input data for the step
        
        Returns:
            Processed step results
        """
        # Normalize step type
        step_type = step_type.lower()
        
        # Flexible step type handling
        if step_type in ['research', 'research_step', 'step_1']:
            return self._process_research(input_data)
        elif step_type in ['document', 'document_step', 'step_2']:
            return self._process_document(input_data)
        elif step_type in ['summary', 'summary_step', 'step_3']:
            # Add a default summary processing
            return {
                'result': {
                    'summary': f"Summary of input data: {input_data}"
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
        else:
            # Return input data as-is for unknown step types
            return {
                'result': input_data,
                'metadata': {
                    'step_type': step_type,
                    'timestamp': time.time()
                }
            }
        
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
            "output": {
                "research_findings": f"Distributed research findings for {topic}",
                "summary": f"Comprehensive analysis of {topic}",
                "recommendations": ["Recommendation 1", "Recommendation 2"],
                "metadata": {
                    "timestamp": time.time()
                }
            }
        }
    
    def _process_document(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Distributed document generation
        
        Args:
            input_data: Input data for document
        
        Returns:
            Generated document
        """
        # Try multiple ways to get research data
        research_data = None
        if 'WORKFLOW.1' in input_data:
            research_data = input_data['WORKFLOW.1']
            if isinstance(research_data, dict):
                research_data = research_data.get('output', research_data)
        
        if research_data is None:
            research_data = (
                input_data.get('research_data', {}) or 
                input_data.get('output', {}) or 
                input_data
            )
        
        # Extract research findings from the research data
        research_findings = (
            research_data.get('research_findings', '') or
            research_data.get('content', 'No research findings available')
        )
        
        return {
            "output": {
                "document": f"Document based on research: {research_findings}",
                "format": "markdown",
                "metadata": {
                    "generated_at": time.time()
                }
            }
        }

    def _prepare_step_input(self, step_def: Dict[str, Any], input_data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input for a workflow step.

        Args:
            step_def (Dict[str, Any]): Step definition from workflow
            input_data (Dict[str, Any]): Initial input data
            previous_results (Dict[str, Any]): Results from previous steps

        Returns:
            Dict[str, Any]: Prepared input for the current step
        """
        # Initialize step input with a copy of input data
        step_input = input_data.copy()

        # Handle input field references
        input_fields = step_def.get('input', [])
        for field in input_fields:
            if field.startswith('WORKFLOW.'):
                # Extract step number from reference
                ref_step_num = field.split('.')[1]
                ref_step_key = f"step_{ref_step_num}"

                # Try to get result from previous steps
                if ref_step_key in previous_results:
                    step_result = previous_results[ref_step_key]
                    # Extract the actual result value
                    if isinstance(step_result, dict):
                        if 'output' in step_result:
                            # If result has 'output' structure, get the result from there
                            output = step_result['output']
                            if isinstance(output, dict):
                                # If output is a dict, try to get 'result' or use as is
                                step_input[field] = output.get('result', output)
                            else:
                                step_input[field] = output
                        elif 'result' in step_result:
                            # If result has 'result' key directly, use that
                            step_input[field] = step_result['result']
                        else:
                            # Otherwise use the entire result dict
                            step_input[field] = step_result
                    else:
                        # If result is not a dict, use it as is
                        step_input[field] = step_result
                else:
                    # Raise an error if the referenced step is not found
                    raise ValueError(f"Referenced step {ref_step_key} not found in previous results")
            elif field in input_data:
                # Add the field from input_data
                step_input[field] = input_data[field]

        # Add step-specific configuration if it exists
        step_config = self.config.get(f"step_{step_def.get('step')}_config", {})
        if step_config:
            additional_inputs = step_config.get('additional_inputs', {})
            if isinstance(additional_inputs, dict):
                step_input.update(additional_inputs)

        return step_input
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Execute workflow steps

        Args:
            input_data (Dict[str, Any]): Initial input data

        Returns:
            Dict[str, Dict[str, Any]]: Results from each executed step
        """
        try:
            # Validate required input fields
            self._validate_input(input_data)

            # Reset workflow state
            self.state_manager.reset_workflow_state()

            # Group steps by dependencies for potential parallel execution
            step_groups = self._group_steps_by_dependencies()

            # Track results and updated input data
            results: Dict[str, Dict[str, Any]] = {}

            # Maximum concurrent steps from config
            max_concurrent_steps = self.config.get('max_concurrent_steps', 4)
            semaphore = asyncio.Semaphore(max_concurrent_steps)

            # Execute steps in order of dependency groups
            for group in step_groups:
                group_tasks = []
                for step_def in group:
                    step_id = f"step_{step_def['step']}"
                    step_number = step_def['step']

                    # Check if step is in distributed steps
                    if step_id not in self.distributed_steps:
                        logger.warning(f"Step {step_id} not found in distributed steps")
                        continue

                    # Create task for this step
                    task = asyncio.create_task(
                        self._execute_step_with_semaphore(
                            semaphore, 
                            step_def, 
                            input_data, 
                            results, 
                            step_id
                        )
                    )
                    group_tasks.append(task)

                # Wait for all tasks in this group to complete
                group_results = await asyncio.gather(*group_tasks, return_exceptions=True)

                # Process results and handle exceptions
                for task_result, step_def in zip(group_results, group):
                    step_id = f"step_{step_def['step']}"
                    step_number = step_def['step']

                    if isinstance(task_result, Exception):
                        # Handle step failure
                        if not self.error_handling.get('continue_on_error', False):
                            raise task_result
                        
                        # Update step status to failed
                        self.state_manager.update_step_status(step_id, StepStatus.FAILED)
                    else:
                        # Format the step result
                        formatted_result = task_result
                        if isinstance(task_result, dict):
                            if 'output' in task_result:
                                formatted_result = task_result
                            else:
                                formatted_result = {'output': task_result}
                        else:
                            formatted_result = {'output': {'result': task_result}}
                        
                        # Update results and input data
                        results[step_id] = formatted_result
                        # Store the result in the state manager
                        self.state_manager.set_step_result(step_id, formatted_result)
                        
                        # Extract the actual result for the next step's input
                        if isinstance(formatted_result, dict):
                            output = formatted_result.get('output', {})
                            if isinstance(output, dict):
                                next_step_input = output.get('result', output)
                            else:
                                next_step_input = output
                        else:
                            next_step_input = formatted_result
                        
                        input_data[f'WORKFLOW.{step_number}'] = next_step_input
                        
                        # Mark step as completed
                        self.state_manager.update_step_status(step_id, StepStatus.COMPLETED)

            return results
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            raise WorkflowExecutionError(f"Workflow execution failed: {str(e)}")

    async def _execute_step_with_semaphore(
            self, 
            semaphore: asyncio.Semaphore, 
            step_def: Dict[str, Any], 
            input_data: Dict[str, Any], 
            results: Dict[str, Any],
            step_id: str
        ) -> Dict[str, Any]:
        """Execute a single step with concurrency control"""
        async with semaphore:
            # Prepare input for the current step
            step_input = self._prepare_step_input(step_def, input_data, results)
            
            # Apply timeout if configured
            timeout = self.config.get('step_timeout')
            if timeout:
                try:
                    async with asyncio.timeout(timeout):
                        return await self._execute_step_with_retry(step_id, step_input)
                except asyncio.TimeoutError:
                    raise ValueError(f"Step {step_id} timed out after {timeout} seconds")
            else:
                return await self._execute_step_with_retry(step_id, step_input)

    async def _execute_step_with_retry(
        self, 
        step_id: str, 
        step_input: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0
    ) -> Dict[str, Any]:
        """Execute a workflow step with retry mechanism.
        
        Args:
            step_id: Identifier for the current step
            step_input: Input data for the step
            max_retries: Maximum number of retries
            retry_delay: Initial delay between retries
            retry_backoff: Exponential backoff factor
            
        Returns:
            Dict containing step execution results
            
        Raises:
            WorkflowExecutionError: If step execution fails after all retries
        """
        # Reset retry count at the start of execution
        self.state_manager.reset_step_retry_count(step_id)
        
        # Get the step function
        step_func = self.distributed_steps.get(step_id)
        if not step_func:
            raise WorkflowExecutionError(f"Step {step_id} not found in distributed steps")
        
        # Jitter to prevent thundering herd problem
        jitter = lambda x: x * (1 + random.uniform(-0.1, 0.1))
        
        for attempt in range(max_retries + 1):
            try:
                # Preprocess input
                preprocessed_input = await self._preprocess_input(step_input, step_id)
                
                # Execute the step with Ray or standard async method
                if hasattr(step_func, 'execute') and hasattr(step_func.execute, 'remote'):
                    # Ray actor
                    raw_result = await step_func.execute.remote(preprocessed_input)
                else:
                    # Regular object or function
                    raw_result = await step_func.execute(preprocessed_input)

                # Structure the result properly
                result = {
                    'output': raw_result,
                    'step_id': step_id,
                    'status': 'success'
                }
                
                # If successful, return the result
                return result
            
            except Exception as e:
                current_retry = attempt + 1
                # If it's the last attempt, raise a persistent failure
                if attempt == max_retries:
                    # Mark step as failed if all retries exhausted
                    self.state_manager.update_step_status(step_id, StepStatus.FAILED)
                    raise WorkflowExecutionError(f"Persistent failure in step {step_id}: {str(e)}")
                
                # Update retry status for this attempt
                self.state_manager.retry_step(step_id)
                
                # Update step metadata with last error
                error_metadata = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'timestamp': time.time()
                }
                self.state_manager.update_step_metadata(step_id, {'last_error': error_metadata})

                # Increment retry count
                self.state_manager.increment_retry_count(step_id)

                # Exponential backoff with jitter
                delay = jitter(retry_delay * (retry_backoff ** attempt))
                logger.warning(f"Retry attempt {current_retry} for step {step_id}. Waiting {delay:.2f} seconds.")
                
                # Wait before next retry
                await asyncio.sleep(delay)
                
    def _group_steps_by_dependencies(self) -> List[List[Dict[str, Any]]]:
        """Group steps that can be executed in parallel based on their dependencies"""
        steps = sorted(self.workflow_config.get('WORKFLOW', []), key=lambda x: x.get('step', 0))
        groups = []
        current_group = []
        
        for step in steps:
            # If step has no dependencies, add to current group
            dependencies = self._get_step_dependencies(step)
            if not dependencies:
                current_group.append(step)
            else:
                # If step depends on previous steps, start new group
                if current_group:
                    groups.append(current_group)
                current_group = [step]
        
        if current_group:
            groups.append(current_group)
        
        return groups

    def _get_step_dependencies(self, step: Dict[str, Any]) -> frozenset:
        """Get the step numbers that this step depends on"""
        dependencies = set()
        for input_key in step.get('input', []):
            if input_key.startswith('WORKFLOW.'):
                try:
                    step_num = int(input_key.split('.')[1])
                    dependencies.add(step_num)
                except (IndexError, ValueError):
                    pass
        # Add explicit dependencies
        explicit_deps = step.get('depends_on', [])
        if isinstance(explicit_deps, (list, tuple)):
            dependencies.update(int(dep) for dep in explicit_deps if str(dep).isdigit())
        
        return frozenset(dependencies)

    def _validate_input(self, input_data: Dict[str, Any]):
        """Validate input data against workflow requirements.
        
        Args:
            input_data: Dictionary containing input data for the workflow
            
        Raises:
            ValueError: If required inputs are missing or invalid
            TypeError: If input_data is not a dictionary
        """
        # Skip validation if no workflow steps
        if not self.workflow_config.get('WORKFLOW', []):
            return
            
        # Skip validation if input_data is None
        if input_data is None:
            return
            
        # Validate input_data is a dictionary
        if not isinstance(input_data, dict):
            raise TypeError(f"Input data must be a dictionary, got {type(input_data)}")
            
        # Get required fields from workflow steps
        required_fields = []
        workflow_refs = []
        for step in self.workflow_config['WORKFLOW']:
            for field in step.get('input', []):
                # Handle nested field references
                if field.startswith('WORKFLOW.'):
                    # Ignore workflow references during initial validation
                    continue
                elif '.' in field:
                    base_field = field.split('.')[0]
                    if base_field not in required_fields:
                        required_fields.append(base_field)
                else:
                    if field not in required_fields and not field.startswith('WORKFLOW.'):
                        required_fields.append(field)
        
        # Remove duplicates while preserving order
        required_fields = list(dict.fromkeys(required_fields))
            
        # Validate required fields
        missing_inputs = []
        empty_inputs = []
        
        for req_input in required_fields:
            if req_input not in input_data:
                missing_inputs.append(req_input)
            elif input_data[req_input] is None or str(input_data[req_input]).strip() == '':
                empty_inputs.append(req_input)
        
        # Build error message if validation fails
        error_msgs = []
        if missing_inputs:
            error_msgs.append(f"Missing required inputs: {', '.join(missing_inputs)}")
        if empty_inputs:
            error_msgs.append(f"Required inputs cannot be None or empty: {', '.join(empty_inputs)}")
        
        if error_msgs:
            raise ValueError(' '.join(error_msgs))

class DistributedWorkflow:
    def __init__(
        self, 
        workflow_config: Dict[str, Any] = None, 
        config: Optional[Dict[str, Any]] = None, 
        state_manager: Optional[WorkflowStateManager] = None,
        workflow_def: Dict[str, Any] = None  
    ):
        # Prefer workflow_def if provided, otherwise use workflow_config
        if workflow_def is not None:
            # Convert workflow_def to workflow_config format
            workflow_config = {
                'WORKFLOW': workflow_def.get('execution_policies', {}).get('steps', [])
            }
        
        # Validate that we have a workflow configuration
        if workflow_config is None:
            raise ValueError("Either workflow_config or workflow_def must be provided")

        # Convert workflow_config to standard format if needed
        if isinstance(workflow_config, dict) and 'WORKFLOW' in workflow_config:
            steps = workflow_config['WORKFLOW']
            if isinstance(steps, list):
                # Normalize each step to ensure consistent format
                normalized_steps = []
                for step in steps:
                    normalized_step = {
                        'step': step.get('step', len(normalized_steps) + 1),
                        'type': step.get('type', 'research'),
                        'name': step.get('title', f'Step {step.get("step", len(normalized_steps) + 1)}'),
                        'description': step.get('description', ''),
                        'input': step.get('input', []),
                        'output': step.get('output', {}),
                        'agent_config': step.get('agent_config', {})
                    }
                    normalized_steps.append(normalized_step)
                workflow_config['WORKFLOW'] = normalized_steps

        self.workflow_config = workflow_config
        self.config = config or {}
        self.distributed_steps = {}
        self.state_manager = state_manager or WorkflowStateManager()
        
        # Extract required fields from workflow steps
        self.required_fields = []
        for step in workflow_config.get('WORKFLOW', []):
            # Collect input fields from each step, excluding workflow references
            self.required_fields.extend([
                field for field in step.get('input', []) 
                if not field.startswith('WORKFLOW.')
            ])
        
        # Remove duplicates while preserving order
        self.required_fields = list(dict.fromkeys(self.required_fields))
        
        # Initialize step configurations
        self._initialize_step_configs()

    def _validate_workflow_config(self, workflow_config):
        """Validate workflow configuration during initialization"""
        required_keys = ['WORKFLOW']
        optional_keys = ['AGENT', 'CONTEXT', 'OBJECTIVE']
        
        # Check for required keys
        for key in required_keys:
            if key not in workflow_config:
                raise ValueError(f"Missing required workflow configuration key: {key}")
        
        # Validate workflow steps
        if not isinstance(workflow_config['WORKFLOW'], list):
            raise ValueError("WORKFLOW must be a list of step configurations")
        
        if not workflow_config['WORKFLOW']:
            raise ValueError("WORKFLOW cannot be empty")
    
    def _initialize_step_configs(self):
        """Initialize step configurations from workflow definition"""
        for step in self.workflow_config.get('WORKFLOW', []):
            step_id = f"step_{step.get('step', 0)}"
            step_config = {
                'timeout': self.config.get(f'{step_id}_config', {}).get('timeout', 30),
                'preprocessors': self.config.get(f'{step_id}_config', {}).get('preprocessors', []),
                'postprocessors': self.config.get(f'{step_id}_config', {}).get('postprocessors', [])
            }
            self.config[f'{step_id}_config'] = step_config
    
    async def _preprocess_input(self, input_data, step_id):
        """Apply preprocessors to input data"""
        preprocessors = self.config.get(f'{step_id}_config', {}).get('preprocessors', [])
        for preprocessor in preprocessors:
            input_data = await preprocessor(input_data)
        return input_data
    
    async def _postprocess_result(self, result, step_id):
        """Apply postprocessors to result"""
        postprocessors = self.config.get(f'{step_id}_config', {}).get('postprocessors', [])
        for postprocessor in postprocessors:
            result = await postprocessor(result)
        return result
    
    async def _execute_step_with_retry(
        self, 
        step_id: str, 
        step_func, 
        input_data=None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0
    ):
        """Execute a step with retry mechanism"""
        # Reset retry count at the start of execution
        self.state_manager.reset_step_retry_count(step_id)
        
        # Jitter to prevent thundering herd problem
        jitter = lambda x: x * (1 + random.uniform(-0.1, 0.1))
        
        current_retry = 0
        current_delay = retry_delay
        last_exception = None

        while current_retry < max_retries + 1:
            try:
                # Execute the step function
                if input_data is not None:
                    raw_result = await step_func(input_data)
                else:
                    raw_result = await step_func()
                
                # Structure the result properly
                result = {
                    'result': raw_result,
                    'step_id': step_id,
                    'status': 'success'
                }
                
                # If successful, return the result
                return result
            
            except Exception as e:
                current_retry += 1
                last_exception = e
                logger.warning(f"Step {step_id} failed (attempt {current_retry}/{max_retries + 1}): {str(e)}")

                # Update retry status
                self.state_manager.retry_step(step_id)
                
                # Update step metadata with last error
                error_metadata = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'timestamp': time.time()
                }
                self.state_manager.update_step_metadata(step_id, {'last_error': error_metadata})

                # Increment retry count
                self.state_manager.increment_retry_count(step_id)

                # If all retries exhausted, raise the final error
                if current_retry == max_retries + 1:
                    logger.error(f"Step {step_id} failed after {max_retries} retries: {str(last_exception)}")
                    self.state_manager.update_step_status(step_id, StepStatus.FAILED)
                    raise WorkflowExecutionError(f"Persistent failure in step {step_id}: {str(last_exception)}")
                
                # Wait before retrying with exponential backoff
                await asyncio.sleep(current_delay)
                current_delay *= retry_backoff
                
    async def execute(self, input_data):
        """Execute workflow with input validation"""
        # Validate required inputs
        self._validate_input(input_data)
        
        # Track overall workflow execution
        results = {}
        parallel_steps = []
        sequential_steps = []
        
        # Identify and categorize steps
        for step_config in sorted(self.workflow_config.get('WORKFLOW', []), key=lambda x: x.get('step', 0)):
            step_id = f"step_{step_config.get('step', 0)}"
            
            # Prepare step input based on dependencies
            step_input = self._prepare_step_input(input_data, step_config, results)
            
            # Get the step function
            step_func = self.distributed_steps.get(step_id)
            if not step_func:
                continue
            
            # Categorize steps
            if step_config.get('parallel', False):
                parallel_steps.append((step_id, step_func, step_input))
            else:
                sequential_steps.append((step_id, step_func, step_input))
        
        # Execute sequential steps first
        for step_id, step_func, step_input in sequential_steps:
            step_result = await self._execute_step_with_retry(step_id, step_func, step_input)
            results[step_id] = step_result
        
        # Execute parallel steps concurrently
        if parallel_steps:
            parallel_results = await asyncio.gather(
                *[self._execute_step_with_retry(step_id, step_func, step_input) 
                  for step_id, step_func, step_input in parallel_steps]
            )
            
            # Add parallel step results
            for (step_id, _, _), result in zip(parallel_steps, parallel_results):
                results[step_id] = result
        
        return results
    
    def _validate_input(self, input_data):
        """Validate input data against workflow requirements.
        
        Args:
            input_data: Dictionary containing input data for the workflow
            
        Raises:
            ValueError: If required inputs are missing or invalid
            TypeError: If input_data is not a dictionary
        """
        # Skip validation if no workflow steps
        if not self.workflow_config.get('WORKFLOW', []):
            return
            
        # Skip validation if input_data is None
        if input_data is None:
            return
            
        # Validate input_data is a dictionary
        if not isinstance(input_data, dict):
            raise TypeError(f"Input data must be a dictionary, got {type(input_data)}")
            
        # Get required fields from workflow steps
        required_fields = []
        workflow_refs = []
        for step in self.workflow_config['WORKFLOW']:
            for field in step.get('input', []):
                # Handle nested field references
                if field.startswith('WORKFLOW.'):
                    # Ignore workflow references during initial validation
                    continue
                elif '.' in field:
                    base_field = field.split('.')[0]
                    if base_field not in required_fields:
                        required_fields.append(base_field)
                else:
                    if field not in required_fields and not field.startswith('WORKFLOW.'):
                        required_fields.append(field)
        
        # Remove duplicates while preserving order
        required_fields = list(dict.fromkeys(required_fields))
            
        # Validate required fields
        missing_inputs = []
        empty_inputs = []
        
        for req_input in required_fields:
            if req_input not in input_data:
                missing_inputs.append(req_input)
            elif input_data[req_input] is None or str(input_data[req_input]).strip() == '':
                empty_inputs.append(req_input)
        
        # Build error message if validation fails
        error_msgs = []
        if missing_inputs:
            error_msgs.append(f"Missing required inputs: {', '.join(missing_inputs)}")
        if empty_inputs:
            error_msgs.append(f"Required inputs cannot be None or empty: {', '.join(empty_inputs)}")
        
        if error_msgs:
            raise ValueError(' '.join(error_msgs))

    def _prepare_step_input(self, input_data, step_config, previous_results):
        """Prepare input for a step based on its configuration and previous results"""
        step_input = input_data.copy()
        
        # Add dependencies from previous steps
        for dep_key in step_config.get('input', []):
            if dep_key.startswith('WORKFLOW.'):
                # Extract step number from dependency
                dep_step_num = dep_key.split('.')[1]
                dep_step_id = f"step_{dep_step_num}"
                
                # Get the result from previous steps
                step_result = None
                if dep_step_id in previous_results:
                    step_result = previous_results[dep_step_id]
                    # Extract the actual output from the result
                    if isinstance(step_result, dict):
                        if 'output' in step_result:
                            step_result = step_result['output']
                            if isinstance(step_result, dict) and 'result' in step_result:
                                step_result = step_result['result']
                
                if step_result is not None:
                    step_input[dep_key] = step_result
        
        return step_input

class ResearchStep:
    """A class representing a research workflow step that can be executed remotely."""

    def __init__(self, step_id: str, config: Dict[str, Any]):
        """Initialize the research step.
        
        Args:
            step_id: Unique identifier for this step
            config: Configuration dictionary for this step
        """
        self.step_id = step_id
        self.config = config
        self.type = config.get('type', 'research')
        self.name = config.get('name', f'Step {step_id}')
        self.description = config.get('description', '')
        self.input_fields = config.get('input', [])
        self.output_config = config.get('output', {})
        self.agent_config = config.get('agent_config', {})

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research step.
        
        Args:
            input_data: Input data for this step
            
        Returns:
            Dict containing the step results
        """
        try:
            # Validate input data
            self._validate_input(input_data)
            
            # Process the step based on its type
            if self.type == 'research':
                result = await self._execute_research(input_data)
            elif self.type == 'document':
                result = await self._execute_document(input_data)
            else:
                result = await self._execute_default(input_data)

            # Format the output according to the output config
            formatted_result = self._format_output(result)
            
            return {
                'output': formatted_result,
                'step_id': self.step_id,
                'type': self.type,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error in step {self.step_id}: {str(e)}")
            raise

    def _validate_input(self, input_data: Dict[str, Any]):
        """Validate input data against required input fields."""
        for field in self.input_fields:
            if not field.startswith('WORKFLOW.'):  # Skip workflow dependencies
                if field not in input_data:
                    raise ValueError(f"Required input field '{field}' not found in input data")

    async def _execute_research(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research type step."""
        # For now, just return a mock research result
        return {
            'research_output': {
                'topic': input_data.get('research_topic', 'Unknown'),
                'analysis': 'Research analysis would go here',
                'timestamp': time.time()
            }
        }

    async def _execute_document(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document type step."""
        # For now, just return a mock document result
        return {
            'document_output': {
                'content': 'Document content would go here',
                'format': self.output_config.get('format', 'plain text'),
                'timestamp': time.time()
            }
        }

    async def _execute_default(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute default step type."""
        return {
            'result': input_data,
            'timestamp': time.time()
        }

    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format the output according to the output configuration."""
        output_type = self.output_config.get('type', 'default')
        output_format = self.output_config.get('format', 'plain text')
        
        formatted_result = {
            'type': output_type,
            'format': output_format,
            'result': result
        }
        
        # Add any additional output configuration
        for key, value in self.output_config.items():
            if key not in ['type', 'format']:
                formatted_result[key] = value
                
        return formatted_result

class ResearchDistributedWorkflow(DistributedWorkflow):
    """Distributed research workflow implementation"""
    
    def __init__(
            self, 
            workflow_config: Dict[str, Any] = None, 
            config: Optional[Dict[str, Any]] = None, 
            state_manager: Optional[WorkflowStateManager] = None,
            workflow_def: Dict[str, Any] = None  
        ):
        """Initialize research distributed workflow.
        
        Args:
            workflow_config: Workflow configuration dictionary
            config: Optional configuration dictionary
            state_manager: Optional state manager instance
            workflow_def: Alternative workflow definition (for API compatibility)
        """
        # Prefer workflow_def if provided, otherwise use workflow_config
        if workflow_def is not None:
            # Convert workflow_def to workflow_config format
            workflow_config = {
                'WORKFLOW': workflow_def.get('execution_policies', {}).get('steps', [])
            }
        
        # Validate that we have a workflow configuration
        if workflow_config is None:
            raise ValueError("Either workflow_config or workflow_def must be provided")

        # Convert workflow_config to standard format if needed
        if isinstance(workflow_config, dict) and 'WORKFLOW' in workflow_config:
            steps = workflow_config['WORKFLOW']
            if isinstance(steps, list):
                # Normalize each step to ensure consistent format
                normalized_steps = []
                for step in steps:
                    normalized_step = {
                        'step': step.get('step', len(normalized_steps) + 1),
                        'type': step.get('type', 'research'),
                        'name': step.get('title', f'Step {step.get("step", len(normalized_steps) + 1)}'),
                        'description': step.get('description', ''),
                        'input': step.get('input', []),
                        'output': step.get('output', {}),
                        'agent_config': step.get('agent_config', {})
                    }
                    normalized_steps.append(normalized_step)
                workflow_config['WORKFLOW'] = normalized_steps

        super().__init__(workflow_config=workflow_config, config=config, state_manager=state_manager)
        
    def _initialize_step_configs(self):
        """Initialize step configurations for research workflow.
        Overrides the base implementation to add research-specific configurations.
        """
        # Initialize base configurations
        super()._initialize_step_configs()
        
        # Add research-specific configurations
        for step in self.workflow_config.get('WORKFLOW', []):
            step_id = f"step_{step['step']}"
            step_config = self.config.get(f'{step_id}_config', {})
            
            # Set default timeout for research steps
            if 'timeout' not in step_config:
                step_config['timeout'] = 60  # Default 60 second timeout
                
            # Set default retry configuration
            if 'max_retries' not in step_config:
                step_config['max_retries'] = self.config.get('max_retries', 3)
                
            self.config[f'{step_id}_config'] = step_config
            
            # Initialize step in state manager
            self.state_manager.initialize_step(step_id)
            
            # Create distributed step if not exists
            if step_id not in self.distributed_steps:
                step_class = ResearchStep
                if not hasattr(step_class, '_remote'):
                    step_class = ray.remote(step_class)
                
                # Pass step_id and config separately
                self.distributed_steps[step_id] = step_class.remote(
                    step_id=step_id, 
                    config={
                        'step': step['step'],
                        'input': step.get('input', []),
                        'output': step.get('output', {}),
                        **step_config
                    }
                )

    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously.
        
        Args:
            input_data: Input data for workflow execution
            
        Returns:
            Dict containing execution results
        """
        try:
            # Validate input data
            self._validate_input(input_data)
            
            # Initialize workflow state
            self.state_manager.initialize_workflow()
            
            # Execute steps in order
            results = {}
            final_output = None
            
            for step in self.workflow_config.get('WORKFLOW', []):
                step_id = f"step_{step['step']}"
                step_config = self.config.get(f'{step_id}_config', {})
                
                # Prepare step input
                step_input = self._prepare_step_input(step, input_data, results)
                
                # Adjust max_retries to ensure total attempts is 3
                max_retries = step_config.get('max_retries', self.config.get('max_retries', 3))
                adjusted_max_retries = max(2, max_retries)  # Ensure at least 2 retries

                # Execute step with retry
                try:
                    step_result = await self._execute_step_with_retry(
                        step_id,
                        step_input,
                        adjusted_max_retries,
                        step_config.get('retry_delay', self.config.get('retry_delay', 1.0)),
                        step_config.get('retry_backoff', self.config.get('retry_backoff', 2.0))
                    )
                    
                    # Store the raw step result
                    results[step_id] = step_result
                    # Store the result in the state manager
                    self.state_manager.set_step_result(step_id, step_result)
                    
                    # Update workflow state
                    self.state_manager.set_step_status(step_id, StepStatus.SUCCESS)
                    
                    # Transform result to match expected output format
                    if isinstance(step_result, dict):
                        # Create output structure with step output configuration
                        step_output = step.get('output', {})
                        if step_output:
                            # Extract the actual result content
                            result_content = step_result.get('result', {}) if isinstance(step_result.get('result'), dict) else {'result': step_result.get('result', step_result)}
                            
                            # Ensure result_content is a dictionary
                            if not isinstance(result_content, dict):
                                result_content = {'result': result_content}
                            
                            # Create the final output structure
                            final_output = {
                                'output': {
                                    **step_output,
                                    **result_content
                                }
                            }
                        else:
                            final_output = step_result
                    else:
                        # If step_result is not a dict, wrap it in the proper structure
                        final_output = {
                            'output': {
                                **(step.get('output', {})),
                                'result': step_result
                            }
                        }
                
                except Exception as e:
                    self.state_manager.set_step_status(step_id, StepStatus.FAILED)
                    raise WorkflowExecutionError(f"Persistent failure in step {step_id}: {str(e)}")

            # Return final output or results
            return final_output if final_output is not None else {'output': results}

        except ValueError as ve:
            # Directly re-raise ValueError for input validation
            raise

        except Exception as e:
            self.state_manager.set_workflow_status(WorkflowStatus.FAILED)
            persistent_failure_match = re.search(r"Persistent failure in step (\w+)", str(e))
            if persistent_failure_match:
                raise WorkflowExecutionError(f"Workflow execution failed: {str(e)}")
            else:
                raise WorkflowExecutionError(f"Workflow execution failed with error: {str(e)}")

    async def _execute_step_with_retry(
            self, 
            step_id: str, 
            step_input: Dict[str, Any],
            max_retries: int = 3,
            retry_delay: float = 1.0,
            retry_backoff: float = 2.0
        ) -> Dict[str, Any]:
            """Execute a workflow step with retry mechanism.
            
            Args:
                step_id: Identifier for the current step
                step_input: Input data for the step
                max_retries: Maximum number of retries
                retry_delay: Initial delay between retries
                retry_backoff: Exponential backoff factor
                
            Returns:
                Dict containing step execution results
                
            Raises:
                WorkflowExecutionError: If step execution fails after all retries
            """
            # Retrieve the step from distributed steps
            step = self.distributed_steps.get(step_id)
            if not step:
                raise ValueError(f"No step found for step_id: {step_id}")

            # Preprocess input
            preprocessed_input = await self._preprocess_input(step_input, step_id)

            # Execute step with retry mechanism
            current_retry = 0
            current_delay = retry_delay
            last_exception = None

            # Reset retry count at the start of execution
            self.state_manager.reset_step_retry_count(step_id)

            # Ensure 3 total attempts (initial + retries)
            while current_retry < max_retries + 1:
                try:
                    # Check if the step is a Ray actor or a regular object
                    if hasattr(step, 'execute') and hasattr(step.execute, 'remote'):
                        # Ray actor
                        raw_result = await step.execute.remote(preprocessed_input)
                    else:
                        # Regular async method
                        raw_result = await step.execute(preprocessed_input)
                    
                    # Structure the result properly
                    result = {
                        'result': raw_result,
                        'step_id': step_id,
                        'status': 'success'
                    }
                    
                    # If successful, return the result
                    return result

                except Exception as e:
                    current_retry += 1
                    last_exception = e
                    
                    # Update retry status
                    self.state_manager.retry_step(step_id)
                    
                    # Update step metadata with last error
                    error_metadata = {
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'timestamp': time.time()
                    }
                    self.state_manager.update_step_metadata(step_id, {'last_error': error_metadata})

                    # Increment retry count
                    self.state_manager.increment_retry_count(step_id)

                    # Log retry attempt
                    logger.warning(f"Step {step_id} failed (attempt {current_retry}/{max_retries + 1}): {str(e)}")

                    # If all retries exhausted, raise the final error
                    if current_retry == max_retries + 1:
                        logger.error(f"Step {step_id} failed after {max_retries} retries: {str(last_exception)}")
                        self.state_manager.update_step_status(step_id, StepStatus.FAILED)
                        raise WorkflowExecutionError(f"Persistent failure in step {step_id}: {str(last_exception)}")
                    
                    # Wait before retrying with exponential backoff
                    await asyncio.sleep(current_delay)
                    current_delay *= retry_backoff

def initialize_ray():
    """Initialize Ray with basic configuration."""
    try:
        if not ray.is_initialized():
            ray.init(
                logging_level=logging.INFO,
                dashboard_host='0.0.0.0',
                ignore_reinit_error=True
            )
            logger.info("Ray initialized with basic configuration")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        raise

def execute_step(step_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone function to execute a step with the given type and input data.

    Args:
        step_type (str): The type of step to execute.
        input_data (Dict[str, Any]): Input data for the step.

    Returns:
        Dict[str, Any]: Output data from the step execution.
    """
    # This is a placeholder implementation. You may need to modify it based on your specific requirements.
    # For now, it simply returns the input data as a demonstration.
    return input_data

def _execute_step_with_retry(step_type: str, step_def: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone function to execute a step with retry mechanism.

    Args:
        step_type (str): The type of step to execute.
        step_def (Dict[str, Any]): Step definition configuration.
        input_data (Dict[str, Any]): Input data for the step.

    Returns:
        Dict[str, Any]: Output data from the step execution.
    """
    # This is a placeholder implementation. You may need to modify it based on your specific requirements.
    # For now, it simply returns the input data as a demonstration.
    return input_data
