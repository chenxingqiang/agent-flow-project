"""Configuration manager module."""

import os
import json
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

from .config import AgentConfig

class ProcessorConfig(BaseModel):
    """Configuration for a processor."""
    model_config = ConfigDict(frozen=False, validate_assignment=True)
    
    id: str
    type: str
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConfigManager:
    """Configuration manager for agent and workflow configurations."""

    def __init__(self, config_dir: str):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configurations
        """
        self.config_dir = Path(config_dir)
        self.agent_dir = self.config_dir / "agents"
        
        # Create directories if they don't exist
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_agent_path(self, agent_id: str) -> Path:
        """Get path for agent configuration file."""
        return self.agent_dir / f"{agent_id}.json"
        
    def save_agent_config(self, agent_config: AgentConfig) -> None:
        """Save agent configuration.
        
        Args:
            agent_config: Agent configuration to save
        """
        # Convert config to dictionary
        config_dict = agent_config.model_dump()
        
        # Save to file
        with open(self._get_agent_path(agent_config.id), "w") as f:
            json.dump(config_dict, f, indent=4)
            
    def load_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Load agent configuration.
        
        Args:
            agent_id: ID of agent configuration to load
            
        Returns:
            Loaded agent configuration or None if not found
        """
        config_path = self._get_agent_path(agent_id)
        if not config_path.exists():
            return None
            
        # Load from file
        with open(config_path) as f:
            config_dict = json.load(f)
            
        # Create agent config
        return AgentConfig(**config_dict)
