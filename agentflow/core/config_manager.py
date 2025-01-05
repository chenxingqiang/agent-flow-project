"""Configuration manager module."""

import os
import json
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field

from .config import AgentConfig, WorkflowConfig, ModelConfig, ConfigurationType

@dataclass
class ProcessorConfig:
    """Configuration for a processor."""
    name: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    timeout: float = 60.0
    retry_count: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "params": self.params,
            "enabled": self.enabled,
            "timeout": self.timeout,
            "retry_count": self.retry_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessorConfig":
        """Create from dictionary."""
        return cls(**data)

class ConfigManager:
    """Configuration manager for agent and workflow configurations."""

    def __init__(self, config_dir: str):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configurations
        """
        self.config_dir = Path(config_dir)
        self.agent_dir = self.config_dir / "agents"
        self.workflow_dir = self.config_dir / "workflows"
        
        # Create directories if they don't exist
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_dir.mkdir(parents=True, exist_ok=True)

    def _get_agent_path(self, agent_id: str) -> Path:
        """Get path for agent configuration file."""
        return self.agent_dir / f"{agent_id}.json"

    def _get_workflow_path(self, workflow_id: str) -> Path:
        """Get path for workflow configuration file."""
        return self.workflow_dir / f"{workflow_id}.json"

    def _prepare_agent_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare agent data for loading.
        
        Args:
            data: Raw agent data
            
        Returns:
            Prepared agent data
        """
        # Convert model data to ModelConfig if needed
        if isinstance(data.get('model'), dict):
            data['model'] = ModelConfig(**data['model'])
            
        # Handle agent type
        if 'agent_type' in data:
            data['type'] = data.pop('agent_type')
            
        return data

    def save_agent_config(self, config: AgentConfig) -> None:
        """Save agent configuration.
        
        Args:
            config: Agent configuration to save
        """
        path = self._get_agent_path(config.id)
        with open(path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    def save_workflow_config(self, config: WorkflowConfig) -> None:
        """Save workflow configuration.
        
        Args:
            config: Workflow configuration to save
        """
        path = self._get_workflow_path(config.id)
        with open(path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    def load_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Load agent configuration.
        
        Args:
            agent_id: ID of agent configuration to load
            
        Returns:
            Loaded agent configuration or None if not found
        """
        path = self._get_agent_path(agent_id)
        if not path.exists():
            return None
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        data = self._prepare_agent_data(data)
        return AgentConfig(**data)

    def load_workflow_config(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """Load workflow configuration.
        
        Args:
            workflow_id: ID of workflow configuration to load
            
        Returns:
            Loaded workflow configuration or None if not found
        """
        path = self._get_workflow_path(workflow_id)
        if not path.exists():
            return None
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Convert agent configs
        if 'agents' in data:
            agents = []
            for agent_data in data['agents']:
                agent_data = self._prepare_agent_data(agent_data)
                agents.append(AgentConfig(**agent_data))
            data['agents'] = agents
            
        return WorkflowConfig(**data)

    def list_agent_configs(self) -> List[AgentConfig]:
        """List all agent configurations.
        
        Returns:
            List of agent configurations
        """
        configs = []
        for path in self.agent_dir.glob('*.json'):
            with open(path, 'r') as f:
                data = json.load(f)
                data = self._prepare_agent_data(data)
                configs.append(AgentConfig(**data))
        return configs

    def list_workflow_configs(self) -> List[WorkflowConfig]:
        """List all workflow configurations.
        
        Returns:
            List of workflow configurations
        """
        configs = []
        for path in self.workflow_dir.glob('*.json'):
            with open(path, 'r') as f:
                data = json.load(f)
                if 'agents' in data:
                    agents = []
                    for agent_data in data['agents']:
                        agent_data = self._prepare_agent_data(agent_data)
                        agents.append(AgentConfig(**agent_data))
                    data['agents'] = agents
                configs.append(WorkflowConfig(**data))
        return configs

    def delete_agent_config(self, agent_id: str) -> bool:
        """Delete agent configuration.
        
        Args:
            agent_id: ID of agent configuration to delete
            
        Returns:
            True if configuration was deleted, False otherwise
        """
        path = self._get_agent_path(agent_id)
        if not path.exists():
            return False
            
        path.unlink()
        return True

    def delete_workflow_config(self, workflow_id: str) -> bool:
        """Delete workflow configuration.
        
        Args:
            workflow_id: ID of workflow configuration to delete
            
        Returns:
            True if configuration was deleted, False otherwise
        """
        path = self._get_workflow_path(workflow_id)
        if not path.exists():
            return False
            
        path.unlink()
        return True

    def export_config(self, config_id: str, export_path: str) -> bool:
        """Export configuration to file.
        
        Args:
            config_id: ID of configuration to export
            export_path: Path to export configuration to
            
        Returns:
            True if configuration was exported, False otherwise
        """
        # Try loading as agent config first
        config = self.load_agent_config(config_id)
        if config:
            with open(export_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            return True
            
        # Try loading as workflow config
        config = self.load_workflow_config(config_id)
        if config:
            with open(export_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            return True
            
        return False

    def import_config(self, import_path: str) -> bool:
        """Import configuration from file.
        
        Args:
            import_path: Path to import configuration from
            
        Returns:
            True if configuration was imported, False otherwise
        """
        if not os.path.exists(import_path):
            return False
            
        with open(import_path, 'r') as f:
            data = json.load(f)
            
        # Create configuration object based on type
        if data.get('type') == ConfigurationType.AGENT.value:
            data = self._prepare_agent_data(data)
            config = AgentConfig(**data)
            self.save_agent_config(config)
            return True
        elif data.get('type') == ConfigurationType.WORKFLOW.value:
            if 'agents' in data:
                agents = []
                for agent_data in data['agents']:
                    agent_data = self._prepare_agent_data(agent_data)
                    agents.append(AgentConfig(**agent_data))
                data['agents'] = agents
            config = WorkflowConfig(**data)
            self.save_workflow_config(config)
            return True
            
        return False
