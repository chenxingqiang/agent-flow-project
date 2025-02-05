from .base import Agent
from ..core.config import AgentConfig

class TestAgent(Agent):
    """Test agent implementation."""
    
    def __init__(self, config: AgentConfig):
        """Initialize test agent."""
        super().__init__(config)
        self.executed = False
        
    def execute(self):
        """Execute test agent."""
        self.executed = True 