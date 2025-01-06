"""Agent module."""

import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from dataclasses import dataclass

from ..core.config import AgentConfig, ModelConfig, WorkflowConfig
from ..core.workflow_state import AgentStatus

@dataclass
class AgentState:
    """Agent state class."""
    status: AgentStatus = AgentStatus.IDLE
    error: Optional[str] = None
    context: Dict[str, Any] = None

class Agent:
    """Agent class."""
    
    def __init__(self, config: Union[str, Dict[str, Any]]):
        """Initialize agent.
        
        Args:
            config: Agent configuration file path or dictionary
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If configuration file not found
        """
        # Load configuration
        if isinstance(config, str):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config}")
            with open(config_path) as f:
                config_data = json.load(f)
        else:
            config_data = config
            
        # Validate configuration
        if not config_data or not isinstance(config_data, dict):
            raise ValueError("Invalid configuration")
            
        # Extract configuration sections
        agent_config = config_data.get("AGENT", {})
        model_config = config_data.get("MODEL", {})
        workflow_config = config_data.get("WORKFLOW", {})
        
        # Create model configuration
        if model_config:
            provider = model_config.get("provider")
            if provider not in ["openai", "anthropic"]:
                raise ValueError(f"Invalid model provider: {provider}")
            self.model = ModelConfig(**model_config)
        else:
            self.model = ModelConfig()
            
        # Create workflow configuration
        if workflow_config:
            if workflow_config.get("max_iterations", 1) <= 0:
                raise ValueError("max_iterations must be greater than 0")
            self.workflow = WorkflowConfig(**workflow_config)
            self.is_distributed = workflow_config.get("distributed", False)
        else:
            self.workflow = WorkflowConfig()
            self.is_distributed = False
            
        # Set agent properties
        self.name = agent_config.get("name")
        self.type = agent_config.get("type", "research")
        self.version = agent_config.get("version", "1.0.0")
        
        # Initialize state
        self.state = AgentState()
