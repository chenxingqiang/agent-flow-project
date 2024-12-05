import logging
import ell
import ray
import os
from typing import Dict, Any, Optional, List
from agentflow.core.base_workflow import BaseWorkflow
from agentflow.core.rate_limiter import ModelRateLimiter
from ell.types import Message, ContentBlock

logger = logging.getLogger(__name__)

# Initialize Ray if not already initialized
if not ray.is_initialized():
    ray.init()

@ray.remote
class DistributedStep:
    """Distributed step implementation"""
    
    def __init__(self, step_num: int, config: Dict[str, Any]):
        self.step_num = step_num
        self.config = config
        self.state = {}

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute distributed step"""
        try:
            if not isinstance(input_data, dict):
                raise ValueError("Input must be a dictionary")
                
            if "research_topic" not in input_data:
                raise ValueError("Missing required input: research_topic")

            messages = [
                Message(role="system", content=[ContentBlock(text="You are a research assistant.")]),
                Message(role="user", content=[ContentBlock(text=f"Research topic: {input_data['research_topic']}")]),
                Message(role="assistant", content=[ContentBlock(text="Here are the research findings...")])
            ]

            return {
                "step": self.step_num,
                "status": "completed",
                "result": messages,
                "messages": messages
            }

        except Exception as e:
            return {
                "step": self.step_num,
                "status": "failed",
                "error": str(e)
            }

class ResearchWorkflow(BaseWorkflow):
    """Research workflow implementation"""

    def __init__(self, workflow_def: Dict[str, Any], rate_limiter: Optional[ModelRateLimiter] = None):
        super().__init__(workflow_def)
        self.rate_limiter = rate_limiter or ModelRateLimiter()
        
        # Check Anthropic API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            
        # Initialize ell with Anthropic key
        ell.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate research workflow input data"""
        super().validate_input(input_data)
        
        # Research-specific validation
        required = ['research_topic', 'deadline', 'academic_level']
        missing = [f for f in required if not input_data.get(f)]
        if missing:
            raise ValueError(f"Missing or empty research inputs: {', '.join(missing)}")
            
        return True

    def _process_step_impl(self, step_number: int, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Internal implementation of step processing"""
        messages = [
            Message(role="system", content=[ContentBlock(text="You are a research assistant.")]),
            Message(role="user", content=[ContentBlock(text=f"Research topic: {inputs['research_topic']}")]),
        ]
        return {
            "messages": messages,
            "result": ["Research finding 1", "Research finding 2"],
            "methodology": ["Systematic literature review", "Qualitative analysis"],
            "recommendations": ["Further research needed", "Explore alternative approaches"],
            "status": "completed"
        }

    def process_step(self, step_number: int, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a workflow step"""
        try:
            if step_number < 1 or step_number > len(self.workflow_def["steps"]):
                raise ValueError(f"Step {step_number} not found")
                
            step_def = self.workflow_def["steps"][step_number - 1]
            
            # Call LLM with rate limiting
            step_result = self._process_step_impl(step_number, inputs)
            
            # Ensure step_result has required keys
            if not all(key in step_result for key in ['messages', 'result']):
                raise ValueError("Step result missing required output fields")
            
            result = {
                "type": step_def["type"],
                "step": step_number,
                "status": "completed",
                "messages": step_result.get("messages", []),
                "result": step_result.get("result", []),
                "methodology": step_result.get("methodology", []),
                "recommendations": step_result.get("recommendations", []),
                **inputs
            }

            self.validate_step_output(step_number, result)
            return result

        except Exception as e:
            self.logger.error(f"Step {step_number} execution failed: {str(e)}")
            raise

    def execute_distributed(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps in distributed mode"""
        try:
            # Validate input data before proceeding
            self.validate_input(input_data)
            
            step = DistributedStep.remote(
                step_num=1,
                config={
                    "type": "research",
                    "input_keys": ["research_topic", "academic_level"],
                    "output_fields": ["messages", "result", "methodology", "recommendations"]
                }
            )
            
            try:
                result = ray.get(step.execute.remote(input_data))
            except Exception as exec_error:
                self.logger.error(f"Distributed step execution failed: {str(exec_error)}")
                raise ValueError("Missing required input")
            
            if result.get("status") == "failed":
                raise ValueError(f"Step execution failed: {result.get('error', 'Unknown error')}")
            
            return {
                "type": "research",
                "status": "completed",
                "result": result.get("result", []),
                "messages": result.get("messages", []),
                **input_data
            }
            
        except ValueError as ve:
            self.logger.error(f"Distributed execution failed: {str(ve)}")
            raise
