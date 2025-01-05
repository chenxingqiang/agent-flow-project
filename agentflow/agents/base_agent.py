"""Base test agent for testing purposes"""

from typing import Dict, Any, Union, Optional, TYPE_CHECKING
import logging
import json
import ray
import asyncio

if TYPE_CHECKING:
    from .agent import Agent as AgentBase
    from ..core.config import AgentConfig

logger = logging.getLogger(__name__)

class BaseTestAgent:
    """Base test agent class."""
    def __init__(self, config: Union[Dict[str, Any], 'AgentConfig']):
        # Ensure config is an AgentConfig instance
        # Map 'test' agent type to 'default'
        if isinstance(config, dict):
            if config.get('type') == 'test':
                config['type'] = 'default'
            self.config = config
        else:
            if config.type == 'test':
                config.type = 'default'
            self.config = config.dict()

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    async def execute(self, context: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Execute the test agent."""
        self.logger.info("Executing test agent with context: %s", context)
        
        # Simulate some processing
        await asyncio.sleep(0.1)
        
        # Return a simple response
        response = {
            "status": "success",
            "message": "Test agent executed successfully",
            "input_context": context,
            "agent_config": self.config,
            "args": args,
            "kwargs": kwargs
        }
        
        self.logger.info("Test agent execution completed with response: %s", response)
        return response

@ray.remote
class CustomAgent:
    """Custom agent class for distributed execution."""
    def __init__(self, config: Union[str, Dict[str, Any], 'AgentConfig'], workflow_path: Optional[str] = None):
        self.config = self._validate_custom_config(config)
        self.workflow_path = workflow_path

    def _validate_custom_config(self, config: Union[str, Dict[str, Any], 'AgentConfig']) -> Dict[str, Any]:
        """Validate and process custom agent configuration."""
        if isinstance(config, str):
            try:
                with open(config, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load config from file: {e}")
        
        if isinstance(config, dict):
            return config
        
        return config.dict()
