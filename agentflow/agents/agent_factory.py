"""Agent factory module."""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import uuid

from .agent_types import AgentConfig, AgentType, AgentMode
from .agent import Agent

class AgentFactory:
    """Agent factory class."""
    
    _agent_types: Dict[str, type] = {}
    
    def __init__(self, name: str = "", description: Optional[str] = None):
        """Initialize agent factory."""
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.agents: Dict[str, Agent] = {}
        self.metadata: Dict[str, Any] = {}
    
    @classmethod
    def register_agent(cls, agent_type: str, agent_class: type) -> None:
        """Register an agent type."""
        cls._agent_types[agent_type] = agent_class
    
    @classmethod
    def create_agent(cls, config: Union[Dict[str, Any], AgentConfig]) -> Agent:
        """Create an agent."""
        if isinstance(config, dict):
            agent_type = config.get("AGENT", {}).get("type")
            if not agent_type:
                raise ValueError("Agent type not specified in configuration")
            
            # Convert string to AgentType enum if needed
            if isinstance(agent_type, str):
                try:
                    agent_type = AgentType(agent_type)
                except ValueError:
                    raise ValueError(f"Invalid agent type: {agent_type}")
            
            # Use the enum value for lookup
            agent_type_str = agent_type.value if isinstance(agent_type, AgentType) else agent_type
            if agent_type_str in cls._agent_types:
                agent_class = cls._agent_types[agent_type_str]
                return agent_class(config=config)
            else:
                raise ValueError(f"Agent type not registered: {agent_type_str}")
        elif isinstance(config, AgentConfig):
            agent_type = config.type
            agent_type_str = agent_type.value if isinstance(agent_type, AgentType) else agent_type
            if agent_type_str in cls._agent_types:
                agent_class = cls._agent_types[agent_type_str]
                return agent_class(config=config)
            else:
                raise ValueError(f"Agent type not registered: {agent_type_str}")
        return Agent(config=config)
        
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
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentFactory":
        """Create from dictionary."""
        factory = cls(name=data.get("name", ""), description=data.get("description"))
        factory.id = data.get("id", str(uuid.uuid4()))
        factory.metadata = data.get("metadata", {})
        
        # Load agents
        for agent_id, agent_data in data.get("agents", {}).items():
            agent = Agent.from_dict(agent_data)
            factory.agents[agent_id] = agent
            
        return factory
