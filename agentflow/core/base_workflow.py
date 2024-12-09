from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging
from .config import WorkflowConfig

class BaseWorkflow(ABC):
    """Base workflow class that defines the core workflow execution logic"""

    def __init__(self, workflow_def: Dict[str, Any]):
        """Initialize workflow with definition
        
        Args:
            workflow_def: Dictionary containing workflow definition and configuration
        """
        self.workflow_def = workflow_def
        if isinstance(workflow_def, dict):
            self.config = WorkflowConfig(**workflow_def)
        else:
            self.config = workflow_def
            
        self.state = {}
        self.logger = logging.getLogger(__name__)
        
        # Get workflow settings from config
        self.required_fields = self.config.execution_policies.required_fields if hasattr(self.config, 'execution_policies') else []
        self.error_handling = self.config.execution_policies.error_handling if hasattr(self.config, 'execution_policies') else {}
        self.default_status = self.config.execution_policies.default_status if hasattr(self.config, 'execution_policies') else None
        self.steps = self.config.execution_policies.steps if hasattr(self.config, 'execution_policies') else []

    def initialize_state(self):
        """Initialize workflow state when execution starts"""
        self.state = {
            "status": self.default_status,
            "current_step": 0,
            "steps": [],
            "errors": []
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate workflow input data"""
        if not input_data:
            error_msg = self.error_handling.get('missing_input_error', 'Missing or empty inputs')
            raise ValueError(error_msg)
        
        required_fields = self.get_required_input_fields()
        missing_fields = [field for field in required_fields if not input_data.get(field)]
        
        if missing_fields:
            error_msg = self.error_handling.get(
                'missing_field_error',
                f'Missing required fields: {", ".join(missing_fields)}'
            )
            raise ValueError(error_msg)
            
        return True

    def get_required_input_fields(self) -> List[str]:
        """Get list of required input fields"""
        return self.required_fields

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow"""
        self.initialize_state()
        try:
            self.validate_input(input_data)
            step_input = input_data.copy()
            
            for step in self.steps:
                step_num = step.get('step', 0)
                self.state['current_step'] = step_num
                step_input = self.execute_step(step, step_input)
                self.state['steps'].append({
                    'step': step_num,
                    'status': 'completed'
                })
                
            return step_input
            
        except Exception as e:
            error_handler = self.error_handling.get('handler', self._default_error_handler)
            return error_handler(e, input_data)
            
    def _default_error_handler(self, error: Exception, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """默认错误处理"""
        self.logger.error(f"Workflow execution failed: {str(error)}")
        return {
            "status": "error",
            "error": str(error),
            "input": input_data
        }

    @abstractmethod
    def execute_step(self, step: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in the workflow
        
        Args:
            step: Current step definition
            inputs: Input data for the step
            
        Returns:
            Dict containing step results
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get current workflow state
        
        Returns:
            Dict containing current workflow state
        """
        return self.state

    def validate_step_output(self, step_number: int, output: Dict[str, Any]) -> bool:
        """Validate step output
        
        Args:
            step_number: Step number to validate
            output: Step output to validate
            
        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        if not isinstance(output, dict):
            raise ValueError(f"Step {step_number} output must be a dictionary")
            
        workflow_steps = self.config.execution_policies.steps if hasattr(self.config, 'execution_policies') else []
        expected_outputs = workflow_steps[step_number - 1].get("outputs", [])
        for field in expected_outputs:
            if field not in output:
                raise ValueError(f"Step {step_number} missing required output field: {field}")
                
        return True

    def update_state(self, new_state: Dict[str, Any]):
        """Update the workflow state with new values"""
        self.state.update(new_state)

    def get_step_dependencies(self, step_name: str) -> List[str]:
        """Get dependencies for a workflow step.

        Args:
            step_name: Name of the step

        Returns:
            List[str]: List of dependent step names
        """
        # Find step in workflow steps
        for step in self.steps:
            if step.get('name') == step_name:
                return step.get('dependencies', [])
        return []

    def check_dependencies(self, step_name: str, context: Dict[str, Any]) -> bool:
        """Check if dependencies for a step are satisfied.

        Args:
            step_name: Name of the step
            context: Current workflow context

        Returns:
            bool: True if dependencies are satisfied, False otherwise
        """
        dependencies = self.get_step_dependencies(step_name)
        return all(dep in context for dep in dependencies)

    def get_step_status(self, step_name: str, context: Dict[str, Any]) -> str:
        """Get status of a workflow step.

        Args:
            step_name: Name of the step
            context: Current workflow context

        Returns:
            str: Status of the step
        """
        return context.get(f"{step_name}_status", self.default_status)
