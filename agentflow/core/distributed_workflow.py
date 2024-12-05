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
                def execute(self, input_data):
                    # Default implementation, can be overridden
                    return {'result': input_data, 'step_num': step_num}
            
            self.distributed_steps[step_num] = DistributedStep.remote()
    
    @abstractmethod
    def _get_step_function(self, step: Dict[str, Any]) -> Callable:
        """
        Get the function to execute for a specific step
        
        Args:
            step: Step definition
        
        Returns:
            Callable step function
        """
        pass
    
    @abstractmethod
    def process_step(self, step_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a specific step type
        
        Args:
            step_type: Type of workflow step
            input_data: Input data for the step
        
        Returns:
            Processed step results
        """
        pass
    
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
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps"""
        if not input_data:
            raise ValueError("Empty input data")
            
        try:
            results = {}
            current_data = input_data.copy()
            
            # Execute each step in sequence
            for step_def in self.workflow_def.get('WORKFLOW', []):
                step_num = step_def['step']
                
                # Get step configuration
                step_config = self.config.get(f'step_{step_num}_config', {})
                
                # Apply preprocessors
                for preprocessor in step_config.get('preprocessors', []):
                    current_data = preprocessor(current_data)
                
                # Validate input
                self.validate_step_input(step_num, current_data)
                
                # Prepare step input
                step_input = self._prepare_step_input(step_def, current_data, results)
                
                # Get step actor
                step_actor = self.distributed_steps.get(step_num)
                if step_actor is None:
                    raise ValueError(f"Step {step_num} not found in workflow")
                    
                # Execute step
                try:
                    # Use step function from config if available
                    step_function = step_config.get('step_function', None)
                    if step_function:
                        result = step_function(step_input)
                    else:
                        result = ray.get(step_actor.execute.remote(step_input))
                    
                    # Apply postprocessors
                    for postprocessor in step_config.get('postprocessors', []):
                        result = postprocessor(result)
                    
                    # Normalize result structure
                    if not isinstance(result, dict):
                        result = {'result': result, 'step_num': step_num}
                    
                    # Ensure step_num is in result
                    if 'step_num' not in result:
                        result['step_num'] = step_num
                    
                    # Preserve original input context
                    result['input'] = step_input
                    
                    # Preserve original input data
                    result['original_input'] = input_data
                    
                    # Merge preprocessing modifications
                    result_data = result.get('result', result)
                    for key, value in current_data.items():
                        if key not in result_data:
                            result_data[key] = value
                    result['result'] = result_data
                    
                    # Merge result with current_data to preserve preprocessing modifications
                    if isinstance(result_data, dict):
                        for key, value in result_data.items():
                            current_data[key] = value
                    
                    # Store result with both step number and string key
                    results[step_num] = result
                    results[f'step_{step_num}'] = result
                    results[str(step_num)] = result
                    
                    # Ensure the result contains the original input data and preprocessing modifications
                    if 'research_topic' in input_data:
                        result['research_topic'] = input_data['research_topic']
                        result['result']['research_topic'] = input_data['research_topic']
                    
                    # Ensure the result contains preprocessing modifications
                    for preprocessor in step_config.get('preprocessors', []):
                        # Attempt to apply preprocessors to the result to capture modifications
                        try:
                            modified_result = preprocessor(result)
                            if isinstance(modified_result, dict):
                                for key, value in modified_result.items():
                                    result[key] = value
                                    result['result'][key] = value
                        except Exception:
                            pass
                    
                except Exception as e:
                    raise ValueError(f"Step {step_num} execution failed: {str(e)}")
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Distributed workflow execution failed: {str(e)}")
            raise
    
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
                def execute(self, input_data):
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
            
        self.state_manager.start_step(step_num)
        
        try:
            result = await step_actor.execute.remote(step_input)
            self.state_manager.complete_step(step_num, result)
            return result
        except Exception as e:
            self.state_manager.fail_step(step_num, str(e))
            raise
            
    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow asynchronously."""
        if not input_data:
            raise ValueError("Empty input data")
            
        results = {}
        for step_num, step_actor in self.distributed_steps.items():
            step_def = next((s for s in self.workflow_def.get('WORKFLOW', []) if s.get('step') == step_num), None)
            if not step_def:
                raise ValueError(f"Step definition for step {step_num} not found")
                
            step_input = self._prepare_step_input(step_def, input_data, results)
            
            # Execute with retry logic
            for attempt in range(self.retry_config.max_retries):
                try:
                    result = await step_actor.execute.remote(step_input)
                    results[step_num] = result
                    break
                except Exception as e:
                    if attempt == self.retry_config.max_retries - 1:
                        raise
                    await asyncio.sleep(
                        self.retry_config.delay * (self.retry_config.backoff_factor ** attempt)
                    )
                    
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
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps"""
        if not input_data:
            raise ValueError("Empty input data")
            
        try:
            results = {}
            current_data = input_data.copy()
            
            # Execute each step in sequence
            for step_def in self.workflow_def.get('WORKFLOW', []):
                step_num = step_def['step']
                
                # Get step configuration
                step_config = self.config.get(f'step_{step_num}_config', {})
                
                # Apply preprocessors
                for preprocessor in step_config.get('preprocessors', []):
                    current_data = preprocessor(current_data)
                
                # Validate input
                self.validate_step_input(step_num, current_data)
                
                # Prepare step input
                step_input = self._prepare_step_input(step_def, current_data, results)
                
                # Get step actor
                step_actor = self.distributed_steps.get(step_num)
                if step_actor is None:
                    raise ValueError(f"Step {step_num} not found in workflow")
                    
                # Execute step
                try:
                    # Use step function from config if available
                    step_function = step_config.get('step_function', None)
                    if step_function:
                        result = step_function(step_input)
                    else:
                        result = ray.get(step_actor.execute.remote(step_input))
                    
                    # Apply postprocessors
                    for postprocessor in step_config.get('postprocessors', []):
                        result = postprocessor(result)
                    
                    # Normalize result structure
                    if not isinstance(result, dict):
                        result = {'result': result, 'step_num': step_num}
                    
                    # Ensure step_num is in result
                    if 'step_num' not in result:
                        result['step_num'] = step_num
                    
                    # Preserve original input context
                    result['input'] = step_input
                    
                    # Preserve original input data
                    result['original_input'] = input_data
                    
                    # Merge preprocessing modifications
                    result_data = result.get('result', result)
                    for key, value in current_data.items():
                        if key not in result_data:
                            result_data[key] = value
                    result['result'] = result_data
                    
                    # Merge result with current_data to preserve preprocessing modifications
                    if isinstance(result_data, dict):
                        for key, value in result_data.items():
                            current_data[key] = value
                    
                    # Store result with both step number and string key
                    results[step_num] = result
                    results[f'step_{step_num}'] = result
                    
                    # Ensure the result contains the original input data and preprocessing modifications
                    if 'research_topic' in input_data:
                        result['research_topic'] = input_data['research_topic']
                        result['result']['research_topic'] = input_data['research_topic']
                    
                    # Ensure the result contains preprocessing modifications
                    for preprocessor in step_config.get('preprocessors', []):
                        # Attempt to apply preprocessors to the result to capture modifications
                        try:
                            modified_result = preprocessor(result)
                            if isinstance(modified_result, dict):
                                for key, value in modified_result.items():
                                    result[key] = value
                                    result['result'][key] = value
                        except Exception:
                            pass
                    
                except Exception as e:
                    raise ValueError(f"Step {step_num} execution failed: {str(e)}")
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Distributed workflow execution failed: {str(e)}")
            raise
    
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
