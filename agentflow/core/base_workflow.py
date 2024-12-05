from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging

class BaseWorkflow(ABC):
    """Base workflow class that defines the core workflow execution logic"""

    def __init__(self, workflow_def: Dict[str, Any]):
        """Initialize workflow with definition
        
        Args:
            workflow_def: Dictionary containing workflow definition and configuration
        """
        self.workflow_def = workflow_def
        self.state = {}
        self.logger = logging.getLogger(__name__)

    def initialize_state(self):
        """Initialize workflow state when execution starts"""
        self.state = {
            "status": "initialized",
            "current_step": 0,
            "steps": [],
            "errors": []
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate workflow input data"""
        if not input_data:
            raise ValueError("Missing or empty research inputs")
        
        required_fields = self.get_required_input_fields()
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                raise ValueError("Missing or empty research inputs")
            
        return True

    def get_required_input_fields(self) -> List[str]:
        """Get list of required input fields"""
        return ["research_topic", "deadline", "academic_level"]

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow"""
        self.initialize_state()
        try:
            self.validate_input(input_data)
            step_input = input_data.copy()
            
            for step in self.workflow_def["steps"]:
                step_num = step["step"]
                self.state["current_step"] = step_num
                
                result = self.process_step(step_num, step_input)
                self.state["steps"].append({
                    "step": step_num,
                    "status": "completed",
                    "result": result
                })
                step_input.update(result)
                
            self.state["status"] = "completed"
            return step_input
            
        except Exception as e:
            self.state["status"] = "failed"
            self.state["errors"].append(str(e))
            raise

    @abstractmethod
    def process_step(self, step_number: int, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single step in the workflow
        
        Args:
            step_number: Current step number
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
            
        expected_outputs = self.workflow_def.get("steps", [])[step_number - 1].get("outputs", [])
        for field in expected_outputs:
            if field not in output:
                raise ValueError(f"Step {step_number} missing required output field: {field}")
                
        return True

    def update_state(self, new_state: Dict[str, Any]):
        """Update the workflow state with new values"""
        self.state.update(new_state)
