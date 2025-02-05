"""Configuration manager module."""

import os
import json
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

from .config import AgentConfig, WorkflowConfig, ModelConfig

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

    def __init__(self, config_dir: str = None):
        """
        Initialize ConfigManager with a configuration directory.
        
        Args:
            config_dir: Directory to store and load configurations
        """
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configs')
        
        os.makedirs(config_dir, exist_ok=True)
        self.config_dir = config_dir
        
        # Create subdirectories for different config types
        self.agent_config_dir = os.path.join(config_dir, 'agents')
        self.workflow_config_dir = os.path.join(config_dir, 'workflows')
        
        os.makedirs(self.agent_config_dir, exist_ok=True)
        os.makedirs(self.workflow_config_dir, exist_ok=True)

    def save_agent_config(self, agent_config: AgentConfig) -> None:
        """
        Save an agent configuration to a JSON file.
        
        Args:
            agent_config: AgentConfig object to save
        """
        filename = os.path.join(self.agent_config_dir, f"{agent_config.id}.json")
        with open(filename, 'w') as f:
            json.dump(agent_config.model_dump(mode='json', by_alias=True, exclude_unset=False), f, indent=2)

    def load_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """
        Load an agent configuration from a JSON file.
        
        Args:
            agent_id: ID of the agent configuration to load
        
        Returns:
            Loaded AgentConfig or None
        """
        filename = os.path.join(self.agent_config_dir, f"{agent_id}.json")
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        
        # Recursively convert nested dictionaries to Pydantic models
        def convert_nested_models(config_dict):
            from .config import ModelConfig, AgentConfig  # Import here to avoid circular import
            
            # Handle nested model conversions
            if isinstance(config_dict, dict):
                # Special handling for specific keys
                if 'model' in config_dict:
                    if not isinstance(config_dict['model'], ModelConfig):
                        config_dict['model'] = ModelConfig(**config_dict['model'])
                
                # Recursively convert nested dictionaries
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        config_dict[key] = convert_nested_models(value)
                    elif isinstance(value, list):
                        config_dict[key] = [convert_nested_models(item) if isinstance(item, dict) else item for item in value]
            
            return config_dict
        
        # Convert nested models and create AgentConfig
        converted_dict = convert_nested_models(config_dict)
        return AgentConfig(**converted_dict)

    def save_workflow_config(self, workflow_config: WorkflowConfig) -> None:
        """
        Save a workflow configuration to a JSON file.
        
        Args:
            workflow_config: WorkflowConfig object to save
        """
        filename = os.path.join(self.workflow_config_dir, f"{workflow_config.id}.json")
        with open(filename, 'w') as f:
            json.dump(workflow_config.model_dump(mode='json', by_alias=True, exclude_unset=False), f, indent=2)

    def load_workflow_config(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """
        Load a workflow configuration from a JSON file.
        
        Args:
            workflow_id: ID of the workflow configuration to load
        
        Returns:
            Loaded WorkflowConfig or None
        """
        filename = os.path.join(self.workflow_config_dir, f"{workflow_id}.json")
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        
        # Recursively convert nested dictionaries to Pydantic models
        def convert_nested_models(config_dict):
            from .config import ModelConfig, AgentConfig, WorkflowConfig  # Import here to avoid circular import
            
            # Handle nested model conversions
            if isinstance(config_dict, dict):
                # Special handling for specific keys
                if 'model' in config_dict:
                    config_dict['model'] = ModelConfig(**config_dict['model'])
                
                # Recursively convert nested dictionaries
                for key, value in config_dict.items():
                    if key == 'agents' and isinstance(value, list):
                        # Convert each agent config
                        config_dict[key] = [AgentConfig(**convert_nested_models(agent)) for agent in value]
                    elif isinstance(value, dict):
                        config_dict[key] = convert_nested_models(value)
                    elif isinstance(value, list):
                        config_dict[key] = [convert_nested_models(item) if isinstance(item, dict) else item for item in value]
            
            return config_dict
        
        # Convert nested models and create WorkflowConfig
        converted_dict = convert_nested_models(config_dict)
        return WorkflowConfig(**converted_dict)

    def list_agent_configs(self) -> List[AgentConfig]:
        """
        List all saved agent configurations.
        
        Returns:
            List of AgentConfig objects
        """
        return [self.load_agent_config(f.replace('.json', '')) 
                for f in os.listdir(self.agent_config_dir) 
                if f.endswith('.json')]

    def list_workflow_configs(self) -> List[WorkflowConfig]:
        """
        List all saved workflow configurations.
        
        Returns:
            List of WorkflowConfig objects
        """
        return [self.load_workflow_config(f.replace('.json', '')) 
                for f in os.listdir(self.workflow_config_dir) 
                if f.endswith('.json')]

    def delete_agent_config(self, agent_id: str) -> bool:
        """
        Delete an agent configuration.
        
        Args:
            agent_id: ID of the agent configuration to delete
        
        Returns:
            True if deleted successfully, False otherwise
        """
        filename = os.path.join(self.agent_config_dir, f"{agent_id}.json")
        try:
            os.remove(filename)
            return True
        except FileNotFoundError:
            print(f"Agent configuration {agent_id} not found")
            return False

    def delete_workflow_config(self, workflow_id: str) -> bool:
        """
        Delete a workflow configuration.
        
        Args:
            workflow_id: ID of the workflow configuration to delete
        
        Returns:
            True if deleted successfully, False otherwise
        """
        filename = os.path.join(self.workflow_config_dir, f"{workflow_id}.json")
        try:
            os.remove(filename)
            return True
        except FileNotFoundError:
            print(f"Workflow configuration {workflow_id} not found")
            return False

    def export_config(self, config_id: str, export_path: str, config_type: str = 'agent'):
        """
        Export a configuration to a specified path.
        
        Args:
            config_id: ID of the configuration to export
            export_path: Path to export the configuration
            config_type: Type of configuration ('agent' or 'workflow')
        """
        if config_type == 'agent':
            config = self.load_agent_config(config_id)
            source_file = os.path.join(self.agent_config_dir, f"{config_id}.json")
        elif config_type == 'workflow':
            config = self.load_workflow_config(config_id)
            source_file = os.path.join(self.workflow_config_dir, f"{config_id}.json")
        else:
            raise ValueError(f"Invalid config type: {config_type}")

        if config is None:
            raise ValueError(f"Configuration {config_id} not found")

        import shutil
        shutil.copy(source_file, export_path)

    def import_config(self, import_path: str, config_type: str = 'agent'):
        """
        Import a configuration from a specified path.
        
        Args:
            import_path: Path to import the configuration from
            config_type: Type of configuration ('agent' or 'workflow')
        
        Returns:
            Imported configuration
        """
        with open(import_path, 'r') as f:
            config_dict = json.load(f)
        
        if config_type == 'agent':
            config = AgentConfig(**config_dict)
            self.save_agent_config(config)
        elif config_type == 'workflow':
            config = WorkflowConfig(**config_dict)
            self.save_workflow_config(config)
        else:
            raise ValueError(f"Invalid config type: {config_type}")

        return config
