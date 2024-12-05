from typing import Dict, Any, List, Optional
import logging
from abc import ABC, abstractmethod

class BaseWorkflow(ABC):
    """Base class for all workflows"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, workflow_def: Optional[Dict[str, Any]] = None):
        """Initialize workflow
        
        Args:
            config: Configuration settings
            workflow_def: Workflow definition
        """
        self.config = config or {}
        self.workflow_def = workflow_def or {}
        self.logger = logging.getLogger(__name__)
        self.state = {}
        
        # Set default workflow definition if not provided
        if not self.workflow_def or 'WORKFLOW' not in self.workflow_def:
            self.workflow_def = {
                'WORKFLOW': [
                    {
                        'step': 1,
                        'input': [],
                        'output': {'type': 'research'}
                    }
                ]
            }
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps"""
        results = {}
        
        # Get all steps from workflow definition
        steps = self.workflow_def['WORKFLOW']
        
        # Execute each step in sequence
        for step in steps:
            step_num = step['step']
            
            # Only execute first step for now
            if step_num > 1:
                break
                
            # Validate step input
            self.validate_step_input(step_num, input_data)
            
            # Prepare input for current step
            step_input = self._prepare_step_input(step, input_data, results)
            
            # Process step
            step_result = self.process_step(step_num, step_input)
            
            # Store result
            results[f"step_{step_num}"] = step_result
            
            # Update input data with step result for next steps
            if isinstance(step_result, dict):
                input_data.update(step_result)
        
        return results

    def _prepare_step_input(self, 
                           step: Dict[str, Any], 
                           input_data: Dict[str, Any],
                           previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for a workflow step"""
        prepared_input = {}
        
        # Check for required input fields
        required_inputs = step.get('input', [])
        missing_inputs = []
        
        for input_var in required_inputs:
            # Handle WORKFLOW.X references
            if isinstance(input_var, str) and input_var.startswith('WORKFLOW.'):
                try:
                    # Extract step number from WORKFLOW.X
                    step_num = int(input_var.split('.')[1])
                    step_key = f'step_{step_num}'
                    
                    # If the step result exists, use it
                    if step_key in previous_results:
                        prepared_input.update(previous_results[step_key])
                    else:
                        self.logger.error(f"Required previous step result {step_key} not found")
                        missing_inputs.append(input_var)
                except (ValueError, IndexError) as e:
                    self.logger.error(f"Invalid workflow reference: {input_var}")
                    missing_inputs.append(input_var)
            else:
                # Handle direct input variables
                if input_var not in input_data or not input_data[input_var]:
                    missing_inputs.append(input_var)
                else:
                    prepared_input[input_var] = input_data[input_var]
        
        # Raise error if any required inputs are missing
        if missing_inputs:
            raise ValueError(f"Missing or empty inputs: {', '.join(missing_inputs)}")
            
        return prepared_input
        
    def validate_step_input(self, step_num: int, step_input: Dict[str, Any]) -> None:
        """Validate step input
        
        Args:
            step_num: Step number
            step_input: Input data for the step
            
        Raises:
            ValueError: If input is empty or invalid
        """
        # Skip validation if no input is required
        if not step_input:
            raise ValueError("Empty input data")

        step_def = next(s for s in self.workflow_def['WORKFLOW'] if s['step'] == step_num)
        required_inputs = step_def.get('input', [])

        # Skip validation for workflow references
        direct_inputs = [i for i in required_inputs if not i.startswith('WORKFLOW.')]

        # Special case: Allow empty research_topic for research workflow
        if step_def.get('output', {}).get('type') == 'research':
            return

        # Check for empty required inputs
        missing_inputs = []
        for input_name in direct_inputs:
            # Skip WORKFLOW references
            if input_name.startswith('WORKFLOW.'):
                continue
            
            # Check if input is missing or empty
            if input_name not in step_input or not str(step_input.get(input_name, '')).strip():
                missing_inputs.append(input_name)

        if missing_inputs and step_num == 1:  # Only enforce for first step
            # If any required input is missing or empty, raise a specific error
            raise ValueError(f"Missing or empty inputs: {', '.join(missing_inputs)}")
    
        # Additional validation can be added here if needed

    @abstractmethod
    def process_step(self, step_number: int, inputs: Dict[str, Any]) -> Any:
        """Process a single workflow step
        
        Args:
            step_number: Step number to process
            inputs: Input data for the step
            
        Returns:
            Step processing results
        """
        pass