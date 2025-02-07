"""Agent factory module."""

from typing import Dict, Any, Optional, List, Union, Type
from pydantic import BaseModel, Field
import uuid
import logging

from .agent_types import AgentConfig, AgentType, AgentMode
from .agent import Agent
from .base import Agent
from .default_agent import DefaultAgent
from .test_agent import TestAgent
from ..core.config import AgentConfig
from ..core.types import AgentType

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
    _agent_types: Dict[str, Type[Agent]] = {}
    
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
    def register_agent(cls, agent_type: str, agent_class: Type[Agent]) -> None:
        """Register an agent type.
        
        Args:
            agent_type: Agent type identifier
            agent_class: Agent class to register
        """
        cls._agent_types[agent_type] = agent_class
    
    @classmethod
    def create_agent(cls, config: Union[Dict[str, Any], AgentConfig]) -> Agent:
        """Create an agent instance.
        
        Args:
            config: Agent configuration
            
        Returns:
            Agent: Created agent instance
            
        Raises:
            ValueError: If agent type is not registered
        """
        # Extract agent type from config
        if isinstance(config, dict):
            agent_type = config.get("AGENT", {}).get("type")
            if not agent_type:
                raise ValueError("Agent type not specified in configuration")
        else:
            agent_type = config.type
            
        if agent_type not in cls._agent_types:
            raise ValueError(f"Agent type {agent_type} not registered")
            
        agent_class = cls._agent_types[agent_type]
        return agent_class(config=config)
    
    @classmethod
    def get_registered_types(cls) -> Dict[str, Type[Agent]]:
        """Get registered agent types.
        
        Returns:
            Dict[str, Type[Agent]]: Dictionary of registered agent types
        """
        return cls._agent_types.copy()
    
    @classmethod
    def clear_registrations(cls) -> None:
        """Clear all registered agent types."""
        cls._agent_types.clear()
    
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
