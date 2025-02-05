from typing import Dict, Any, Optional
from ..core.config import AgentConfig
import uuid

class Agent:
    """Base agent class."""
    
    def __init__(self, config: Optional[AgentConfig] = None, name: Optional[str] = None):
        """Initialize agent.
        
        Args:
            config: Agent configuration
            name: Agent name (optional, will use config name if not provided)
        """
        self.id = str(uuid.uuid4())
        if config is None:
            config = AgentConfig(name=name or str(uuid.uuid4()))
        self.name = name or config.name
        self.type = config.type
        self.config = config
        self.metadata: Dict[str, Any] = {}
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize agent resources."""
        if not self._initialized:
            # Perform any necessary initialization
            self._initialized = True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "config": self.config.dict(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Create agent from dictionary."""
        config = AgentConfig(**data.get("config", {}))
        agent = cls(config)
        agent.id = data.get("id")
        agent.name = data.get("name")
        agent.type = data.get("type")
        agent.metadata = data.get("metadata", {})
        return agent

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent workflow with input data.
        
        Args:
            input_data: Input data for execution
            
        Returns:
            Dict[str, Any]: Execution results
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Execute method must be implemented by subclass") 