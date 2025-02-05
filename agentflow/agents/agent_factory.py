"""Agent factory module."""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import uuid
import logging

from .agent_types import AgentConfig, AgentType, AgentMode
from .agent import Agent
from .base import Agent
from .default_agent import DefaultAgent
from .test_agent import TestAgent
from ..core.config import AgentConfig

logger = logging.getLogger(__name__)

def create_agent(config: Optional[AgentConfig] = None, agent_type: Optional[AgentType] = None) -> 'Agent':
    """Create an agent instance.
    
    Args:
        config: Agent configuration
        agent_type: Type of agent to create (overrides config.type if provided)
        
    Returns:
        Agent: Created agent instance
        
    Raises:
        ValueError: If agent type is invalid
    """
    # Import here to avoid circular imports
    from .agent import Agent
    
    if config is None:
        config = AgentConfig()
    
    if agent_type is not None:
        config.type = agent_type
        
    return Agent(config)

class AgentFactory:
    """Factory class for creating agents."""
    
    _instance = None
    _agents: Dict[str, Agent] = {}
    
    def __init__(self):
        """Initialize agent factory."""
        self.id = str(uuid.uuid4())
        self.name = "AgentFactory"
        self.description = "Factory for creating agents"
        self.metadata: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentFactory, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def create_agent(cls, config: Union[Dict[str, Any], AgentConfig]) -> Agent:
        """Create an agent."""
        if isinstance(config, dict):
            # First try the new format
            agent_type = config.get("agent_type")
            if not agent_type:
                # Try the old format
                agent_type = config.get("AGENT", {}).get("type")
            if not agent_type:
                raise ValueError("Agent type not specified in configuration")
            
            # Convert to AgentConfig if needed
            if not isinstance(config, AgentConfig):
                config = AgentConfig(
                    type=agent_type,
                    name=config.get("name", str(uuid.uuid4()))
                )
        
        return cls._create_agent_instance(config)
    
    @classmethod
    def _create_agent_instance(cls, config: AgentConfig) -> Agent:
        """Create an agent instance based on configuration."""
        agent_type = config.type.lower()
        
        if agent_type == "default":
            return DefaultAgent(config)
        elif agent_type == "test":
            return TestAgent(config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent."""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """List all agents."""
        return list(self._agents.keys())
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent."""
        if agent_id in self._agents:
            del self._agents[agent_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert factory to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agents": {k: v.to_dict() for k, v in self._agents.items()},
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentFactory':
        """Create factory from dictionary."""
        factory = cls()
        factory.id = data.get("id")
        factory.name = data.get("name")
        factory.description = data.get("description")
        factory.metadata = data.get("metadata", {})
        
        for agent_id, agent_data in data.get("agents", {}).items():
            agent = Agent.from_dict(agent_data)
            factory._agents[agent_id] = agent
        
        return factory
