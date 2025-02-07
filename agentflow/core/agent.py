"""Agent class for workflow execution."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from .state import AgentState


class Agent(BaseModel):
    """Agent class for workflow execution."""

    id: str
    name: str
    type: str = "default"
    mode: str = "sequential"
    state: AgentState = Field(default_factory=AgentState)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent.
        
        Args:
            input_data: Input data for execution
            
        Returns:
            Dict[str, Any]: Execution results
        """
        # In test mode, return test response
        if input_data.get("test_mode"):
            return {
                "content": "Test response",
                "metadata": {}
            }
            
        # Execute actual agent logic
        try:
            # Update agent state
            self.state.status = "running"
            
            # Execute agent logic here
            result = {
                "content": "Agent response",
                "metadata": {}
            }
            
            # Update agent state
            self.state.status = "success"
            return result
            
        except Exception as e:
            # Update agent state
            self.state.status = "failed"
            self.state.error = str(e)
            raise 