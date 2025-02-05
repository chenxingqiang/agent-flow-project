from .base import Agent
from ..core.config import AgentConfig

class DefaultAgent(Agent):
    """Default agent implementation."""
    
    def __init__(self, config: AgentConfig):
        """Initialize default agent."""
        super().__init__(config) 