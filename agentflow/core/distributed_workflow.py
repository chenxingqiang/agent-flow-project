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
from .workflow_state import WorkflowStateManager
from .retry import RetryConfig, with_retry
import copy
import json
from enum import Enum
from agentflow.core.exceptions import StepExecutionError, WorkflowExecutionError

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepStatus(Enum):
    """Step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowExecutionError(Exception):
    """Exception raised for errors during workflow execution"""
    pass

class DistributedWorkflowStep:
    """Base class for workflow steps"""
    def __init__(self, step_id: str, config: Dict[str, Any] = None):
        self.step_id = step_id
        self.config = config or {}

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step"""
        raise NotImplementedError("Subclasses must implement execute method")

@ray.remote
class ResearchStep(DistributedWorkflowStep):
    """A research workflow step that can be executed remotely"""
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research step"""
        try:
            # Ensure input_data is a dictionary
            if not isinstance(input_data, dict):
                logger.warning(f"Received non-dictionary input: {input_data}. Converting to dictionary.")
                input_data = {'STUDENT_NEEDS': input_data} if input_data else {}

            # Process input data
            topic = input_data.get('STUDENT_NEEDS', {})
            if isinstance(topic, dict):
                topic = topic.get('RESEARCH_TOPIC', 'Unknown')
            elif not isinstance(topic, str):
                topic = str(topic)

            language = input_data.get('LANGUAGE', {})
            if isinstance(language, dict):
                language = language.get('TYPE', 'English')
            elif not isinstance(language, str):
                language = str(language)

            template = input_data.get('TEMPLATE', 'Default')
            if not isinstance(template, str):
                template = str(template)

            # Return research results
            return {
                'output': {
                    'research_findings': f"Research findings for {topic}",
                    'language': language,
                    'template': template,
                    'timestamp': time.time()
                }
            }
        except Exception as e:
            logger.error(f"Error in research step: {str(e)}")
            raise

@ray.remote
class DocumentStep(DistributedWorkflowStep):
    """A document generation step that can be executed remotely"""
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the document step"""
        try:
            # Get research findings from previous step
            research_data = input_data.get('WORKFLOW.1.output', {})
            findings = research_data.get('research_findings', 'No findings available')

            # Return document results
            return {
                'output': {
                    'document': f"Document based on: {findings}",
                    'format': 'markdown',
                    'timestamp': time.time()
                }
            }
        except Exception as e:
            logger.error(f"Error in document step: {str(e)}")
            raise

@ray.remote
class ImplementationStep(DistributedWorkflowStep):
    """An implementation step that can be executed remotely"""
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the implementation step"""
        try:
            # Get document from previous step
            doc_data = input_data.get('WORKFLOW.2.output', {})
            document = doc_data.get('document', 'No document available')

            # Return implementation results
            return {
                'output': {
                    'implementation': f"Implementation based on: {document}",
                    'format': 'code',
                    'language': 'python',
                    'timestamp': time.time()
                }
            }
        except Exception as e:
            logger.error(f"Error in implementation step: {str(e)}")
            raise

class DistributedWorkflow(BaseWorkflow, ABC):
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
                    step_result_ref = _execute_step_with_retry.remote(
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
            self.state_manager.set_workflow_status(WorkflowStatus.SUCCESS)
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
                    step_result_ref = _execute_step_with_retry.remote(
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
            self.state_manager.set_workflow_status(WorkflowStatus.SUCCESS)
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

@ray.remote
def _execute_step_with_retry(
    distributed_steps: Dict[str, Any], 
    step_id: str, 
    step_input: Dict[str, Any], 
    max_retries: int = 2, 
    retry_delay: float = 1.0, 
    retry_backoff: float = 2.0
):
    """
    Execute a workflow step with robust retry mechanism

    Args:
        distributed_steps (Dict[str, Any]): Dictionary of distributed steps
        step_id (str): Unique identifier for the step
        step_input (Dict[str, Any]): Input data for the step
        max_retries (int): Maximum number of retries
        retry_delay (float): Initial delay between retries
        retry_backoff (float): Exponential backoff factor

    Returns:
        Ray object reference with step result
    """
    current_retry = 0
    current_delay = retry_delay

    while current_retry <= max_retries:
        try:
            # Retrieve the distributed step for execution
            distributed_step = distributed_steps.get(step_id)
            
            if distributed_step is None:
                raise ValueError(f"No step found for step_id: {step_id}")

            # Execute the step
            result = distributed_step.execute.remote(step_input)
            
            # If execution is successful, return the result
            return ray.get(result)

        except Exception as e:
            logger.warning(f"Attempt {current_retry + 1} failed for step {step_id}: {str(e)}")
            
            # If max retries reached, raise the last exception
            if current_retry == max_retries:
                logger.error(f"Step {step_id} failed after {max_retries + 1} attempts")
                raise

            # Exponential backoff
            time.sleep(current_delay)
            current_retry += 1
            current_delay *= retry_backoff

class ResearchDistributedWorkflow(DistributedWorkflow):
    """A specialized distributed workflow for research-related tasks.
    
    This workflow is designed to handle research-specific workflow configurations
    and provides additional functionality for research-oriented tasks.
    """
    
    def __init__(
        self,
        workflow_config: Dict[str, Any] = None,
        config: Optional[Dict[str, Any]] = None,
        state_manager: Optional[WorkflowStateManager] = None
    ):
        super().__init__(workflow_config, config, state_manager)
        
        # Standardize workflow configuration
        if workflow_config and 'steps' in workflow_config:
            workflow_config['WORKFLOW'] = workflow_config.pop('steps')
        
        # Ensure WORKFLOW is a dictionary with structured steps
        workflow_config.setdefault('WORKFLOW', {})
        
        # Standardize step configurations
        for step_id, step_config in workflow_config['WORKFLOW'].items():
            # Ensure each step has standard metadata
            if isinstance(step_config, dict):
                # Add missing standard fields if not present
                step_config.setdefault('step', int(step_id.replace('step_', '')))
                step_config.setdefault('title', f'Research Step {step_id}')
                step_config.setdefault('description', 'A generic research workflow step')
                step_config.setdefault('input', [])
                step_config.setdefault('output', {
                    'type': 'generic',
                    'details': 'Step output',
                    'format': 'plain text',
                    'word_count': 0
                })
                
                # Set step type with a default
                step_type = step_config.get('type', 'research')
                step_config_override = config.get(f'{step_id}_config', {}) if config else {}
                
                if step_type == 'research':
                    self.distributed_steps[step_id] = ResearchStep.remote(step_id, step_config_override)
                elif step_type == 'document':
                    self.distributed_steps[step_id] = DocumentStep.remote(step_id, step_config_override)
                elif step_type == 'implementation':
                    self.distributed_steps[step_id] = ImplementationStep.remote(step_id, step_config_override)
        
        # Add steps attribute for compatibility
        self.steps = self.distributed_steps
        
        # Restore default attributes with more flexibility
        self.required_fields = workflow_config.get('ENVIRONMENT', {}).get('INPUT', 
            {"STUDENT_NEEDS", "LANGUAGE", "TEMPLATE"}
        )
        self.default_status = WorkflowStatus.PENDING
        self.error_handling = {
            'missing_input_error': 'Missing or empty inputs',
            'missing_field_error': 'Missing required fields: {}',
            'handler': self._default_error_handler
        }
        
        # Store the original workflow configuration for reference
        self.original_workflow_config = workflow_config

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow synchronously"""
        return super().execute(input_data)

    def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously"""
        return super().execute_async(input_data)

    @classmethod
    def create_remote_workflow(
        cls,
        workflow_config: Dict[str, Any] = None,
        config: Optional[Dict[str, Any]] = None,
        state_manager: Optional[WorkflowStateManager] = None
    ):
        """Create a Ray remote workflow instance"""
        workflow = cls(workflow_config, config, state_manager)
        return ray.put(workflow)

    @ray.remote
    def execute_remote(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remote execution method for Ray"""
        return self.execute(input_data)

    @ray.remote
    def execute_async_remote(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remote async execution method for Ray"""
        return self.execute_async(input_data)

    @ray.remote
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Callable method for Ray remote execution"""
        return self.execute(input_data)

    async def execute_step(self, step_id: str, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research workflow step.
        
        Args:
            step_id: The ID of the step to execute
            step_input: Input data for the step
            
        Returns:
            Dict[str, Any]: The output from the step execution
        """
        # Get step configuration
        step_config = self.workflow_config.get("WORKFLOW", {}).get(step_id, {})
        step_type = step_config.get("type", "")
        
        # Execute the appropriate step based on type
        if step_type == "research":
            return await self._execute_research_step(step_id, step_input)
        elif step_type == "document":
            return await self._execute_document_step(step_id, step_input)
        elif step_type == "implementation":
            return await self._execute_implementation_step(step_id, step_input)
        else:
            # Default to base class behavior for unknown step types
            return await super().execute_step(step_id, step_input)
            
    async def _execute_research_step(self, step_id: str, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research-specific step."""
        step = ResearchStep.remote(step_id, self.config.get(f"{step_id}_config", {}))
        return await step.execute.remote(step_input)
        
    async def _execute_document_step(self, step_id: str, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a documentation step."""
        step = DocumentStep.remote(step_id, self.config.get(f"{step_id}_config", {}))
        return await step.execute.remote(step_input)
        
    async def _execute_implementation_step(self, step_id: str, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an implementation step."""
        step = ImplementationStep.remote(step_id, self.config.get(f"{step_id}_config", {}))
        return await step.execute.remote(step_input)

    def _default_error_handler(self, error: Exception, input_data: Optional[Dict[str, Any]] = None) -> None:
        """Default error handler for workflow errors."""
        if isinstance(error, ValueError):
            raise error
        raise WorkflowExecutionError(str(error))
        
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
        
    def get_required_input_fields(self) -> List[str]:
        """Get the required input fields for the workflow."""
        return list(self.required_fields)
        
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
