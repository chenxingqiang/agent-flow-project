"""Base test agent for testing purposes"""

from typing import Dict, Any, Union
import logging
from ..core.agent import AgentBase
from ..core.config import AgentConfig

logger = logging.getLogger(__name__)

class BaseTestAgent(AgentBase):
    """Test agent for unit testing."""

    def __init__(self, config: Union[Dict[str, Any], AgentConfig]):
        # Ensure config is an AgentConfig instance
        self.agent_config = config if isinstance(config, AgentConfig) else AgentConfig(**config)
        
        # Initialize base agent with config
        super().__init__(self.agent_config)
        
        # Extract knowledge and data from config
        if isinstance(config, dict):
            self.knowledge = config.get('knowledge', {})
            self.data = config.get('data', {})
        else:
            self.knowledge = getattr(config, 'knowledge', {})
            self.data = getattr(config, 'data', {})

    async def execute(self, context: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Execute test agent functionality.
        
        Args:
            context (Dict[str, Any]): Execution context
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict[str, Any]: Execution results with knowledge and data
        """
        try:
            # Merge knowledge and data into result
            result = {
                'knowledge': self.knowledge,
                'data': self.data
            }
            
            # Add any additional context
            result.update(context)
            
            return result
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            raise
