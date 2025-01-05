"""Base classes for AgentFlow components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel as PydanticBaseModel
import uuid

class BaseComponent(PydanticBaseModel, ABC):
    """Base component class for all AgentFlow components."""
    
    id: str
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the component logic."""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata
        }

class BaseAgent(BaseComponent):
    """Base agent class."""
    
    type: str = "agent"
    capabilities: List[str] = []
    
    @abstractmethod
    async def plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan agent actions."""
        pass
    
    @abstractmethod
    async def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent actions."""
        pass
    
    @abstractmethod
    async def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Observe environment."""
        pass

class BaseModel(BaseComponent):
    """Base model class."""
    
    type: str = "model"
    provider: str
    version: str
    
    @abstractmethod
    async def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions."""
        pass
    
    @abstractmethod
    async def train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model state."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model state."""
        pass

class BaseWorkflow(BaseComponent):
    """Base workflow class."""
    
    type: str = "workflow"
    steps: List[Dict[str, Any]] = []
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize workflow."""
        super().__init__(
            id=config.get("id", str(uuid.uuid4())),
            name=config.get("name", "default"),
            description=config.get("description"),
            metadata=config.get("metadata", {})
        )
        self.steps = config.get("steps", [])
    
    async def initialize(self) -> None:
        """Initialize workflow."""
        # Validate steps
        if not self.steps:
            raise ValueError("Workflow must have at least one step")
        
        # Initialize each step
        for step in self.steps:
            if "id" not in step:
                step["id"] = str(uuid.uuid4())
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow."""
        results = {}
        
        for step in self.steps:
            step_result = await self.run_step(step, context)
            context.update(step_result)
            results[step["id"]] = step_result
        
        return results
    
    async def run_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a workflow step."""
        step_type = step.get("type")
        if not step_type:
            raise ValueError(f"Step {step['id']} must have a type")
        
        # Execute step based on type
        if step_type == "agent":
            return await self._run_agent_step(step, context)
        elif step_type == "function":
            return await self._run_function_step(step, context)
        else:
            raise ValueError(f"Unknown step type: {step_type}")
    
    async def _run_agent_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run an agent step."""
        agent = step.get("agent")
        if not agent:
            raise ValueError(f"Agent step {step['id']} must have an agent")
        
        return await agent.execute(context)
    
    async def _run_function_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a function step."""
        func = step.get("function")
        if not func:
            raise ValueError(f"Function step {step['id']} must have a function")
        
        return await func(context)
