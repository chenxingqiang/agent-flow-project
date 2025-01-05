"""Agent manager module."""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import uuid

from .agent_types import AgentConfig, AgentType, AgentMode, AgentStatus
from .agent import Agent
from .agent_factory import AgentFactory

class AgentManager(BaseModel):
    """Agent manager class."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    description: Optional[str] = None
    factory: AgentFactory = Field(default_factory=AgentFactory)
    agents: Dict[str, Agent] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def create_agent(self, config: Union[Dict[str, Any], AgentConfig]) -> Agent:
        """Create an agent."""
        agent = self.factory.create_agent(config)
        self.agents[agent.id] = agent
        return agent
        
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent."""
        return self.agents.get(agent_id)
        
    def list_agents(self) -> List[str]:
        """List all agents."""
        return list(self.agents.keys())
        
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            
    async def execute_agent(self, agent_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent."""
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        return await agent.execute(context)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "factory": self.factory.to_dict(),
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentManager":
        """Create from dictionary."""
        if "factory" in data:
            data["factory"] = AgentFactory.from_dict(data["factory"])
        if "agents" in data:
            data["agents"] = {k: Agent.from_dict(v) for k, v in data["agents"].items()}
        return cls(**data) 