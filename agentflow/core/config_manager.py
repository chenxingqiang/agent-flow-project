"""
Configuration management for AgentFlow
"""

import json
import os
from typing import Dict, Any, Optional, Union, List, Type
from typing_extensions import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, ValidationError, validator
from pathlib import Path
import logging

# Forward references
AgentConfig = None
ProcessorConfig = None
ConnectionConfig = None

class ModelConfig(BaseModel):
    """Model configuration for AI agents"""
    provider: str
    name: Optional[str] = None
    parameters: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ToolConfig(BaseModel):
    """Tool configuration"""
    name: str
    description: str
    parameters: Dict[str, dict]
    required: List[str]

class ConfigurationSchema(BaseModel):
    """
    Flexible configuration schema for agents and workflows.
    
    Supports dynamic configuration with optional fields.
    """
    # Allow additional fields dynamically
    model_config = ConfigDict(
        extra='allow',  # Allow extra fields
        validate_assignment=True  # Enable validation on attribute assignment
    )
    
    # Optional base fields
    id: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    agent_type: Optional[str] = 'generic'
    description: Optional[str] = None
    
    # Flexible configuration fields
    model: Optional[ModelConfig] = None
    input_specification: Optional[Dict[str, Any]] = None
    
    @field_validator('agent_type', mode='before')
    @classmethod
    def validate_agent_type(cls, v):
        """
        Validate and normalize agent type.
        
        Args:
            v: Agent type input
        
        Returns:
            Normalized agent type
        """
        if not v:
            return 'generic'
        
        valid_types = ['generic', 'research', 'analysis', 'creative', 'technical']
        normalized = str(v).lower()
        
        if normalized not in valid_types:
            logging.warning(f"Unsupported agent type: {v}. Defaulting to 'generic'")
            return 'generic'
        
        return normalized

class AgentConfig(ConfigurationSchema):
    """Agent configuration with additional fields"""
    type: Optional[str] = None
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    system_prompt: Optional[str] = None
    tools: List[ToolConfig] = Field(default_factory=list)
    memory_config: Dict[str, Union[str, int]] = Field(default_factory=dict)
    execution_policies: Dict[str, Any] = Field(default_factory=dict)
    processor: Optional[Union[str, Type]] = None
    
    @model_validator(mode='before')
    @classmethod
    def transform_config(cls, data):
        """
        Transform legacy configuration formats.
        
        Handles different input configuration styles.
        """
        if isinstance(data, dict):
            # Normalize agent type
            if 'agent_type' in data:
                data['agent_type'] = data['agent_type']
            elif 'type' in data:
                data['agent_type'] = data['type']
            
            # Merge config if present
            if 'config' in data:
                data.update(data['config'])
        
        return data

# Alias for backward compatibility
AgentConfiguration = AgentConfig

class ProcessorConfig(BaseModel):
    """Processor node configuration"""
    id: str
    name: Optional[str] = None
    type: Optional[str] = None
    processor: Optional[Union[str, Type]] = None
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

class ConnectionConfig(BaseModel):
    """Node connection configuration"""
    source_id: str
    target_id: str
    source_port: str = 'output'
    target_port: str = 'input'

class WorkflowConfig(BaseModel):
    """Workflow configuration for agents"""
    id: Optional[str] = 'default_workflow'
    name: Optional[str] = 'Default Workflow'
    description: str = 'A default workflow configuration'
    agents: List[AgentConfig] = Field(default_factory=list)
    processors: List[ProcessorConfig] = Field(default_factory=list)
    connections: List[ConnectionConfig] = Field(default_factory=list)
    max_iterations: int = 5
    logging_level: str = 'INFO'
    distributed: bool = False
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator('name', mode='before')
    @classmethod
    def set_default_name(cls, v, values):
        """Set default name if not provided"""
        return v or values.get('id', 'default_workflow')

    @field_validator('description', mode='before')
    @classmethod
    def set_default_description(cls, v):
        """Set default description if not provided"""
        return v or 'A default workflow configuration'

class ConfigManager:
    """Manager for agent and workflow configurations"""
    
    def __init__(self, config_dir: str = None):
        """Initialize config manager
        
        Args:
            config_dir: Directory to store configurations
        """
        self.config_dir = config_dir or os.path.expanduser("~/.agentflow/configs")
        self._ensure_config_dir()
        
    def _ensure_config_dir(self):
        """Ensure configuration directory exists"""
        os.makedirs(self.config_dir, exist_ok=True)
        
    def save_agent_config(self, config: AgentConfig):
        """Save agent configuration
        
        Args:
            config: Agent configuration
        """
        path = os.path.join(self.config_dir, "agents", f"{config.id}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(config.dict(), f, indent=2)
            
    def load_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Load agent configuration
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent configuration if found, None otherwise
        """
        path = os.path.join(self.config_dir, "agents", f"{agent_id}.json")
        
        if not os.path.exists(path):
            return None
            
        with open(path) as f:
            return AgentConfig(**json.load(f))
            
    def list_agent_configs(self) -> List[AgentConfig]:
        """List all agent configurations
        
        Returns:
            List of agent configurations
        """
        agent_dir = os.path.join(self.config_dir, "agents")
        
        if not os.path.exists(agent_dir):
            return []
            
        configs = []
        for file in os.listdir(agent_dir):
            if file.endswith(".json"):
                with open(os.path.join(agent_dir, file)) as f:
                    configs.append(AgentConfig(**json.load(f)))
        return configs
        
    def save_workflow_config(self, config: WorkflowConfig):
        """Save workflow configuration
        
        Args:
            config: Workflow configuration
        """
        path = os.path.join(self.config_dir, "workflows", f"{config.id}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(config.dict(), f, indent=2)
            
    def load_workflow_config(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """Load workflow configuration
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow configuration if found, None otherwise
        """
        path = os.path.join(self.config_dir, "workflows", f"{workflow_id}.json")
        
        if not os.path.exists(path):
            return None
            
        with open(path) as f:
            return WorkflowConfig(**json.load(f))
            
    def list_workflow_configs(self) -> List[WorkflowConfig]:
        """List all workflow configurations
        
        Returns:
            List of workflow configurations
        """
        workflow_dir = os.path.join(self.config_dir, "workflows")
        
        if not os.path.exists(workflow_dir):
            return []
            
        configs = []
        for file in os.listdir(workflow_dir):
            if file.endswith(".json"):
                with open(os.path.join(workflow_dir, file)) as f:
                    configs.append(WorkflowConfig(**json.load(f)))
        return configs
        
    def delete_agent_config(self, agent_id: str) -> bool:
        """Delete agent configuration
        
        Args:
            agent_id: Agent ID
            
        Returns:
            True if configuration was deleted, False otherwise
        """
        path = os.path.join(self.config_dir, "agents", f"{agent_id}.json")
        
        if not os.path.exists(path):
            return False
            
        os.remove(path)
        return True
        
    def delete_workflow_config(self, workflow_id: str) -> bool:
        """Delete workflow configuration
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if configuration was deleted, False otherwise
        """
        path = os.path.join(self.config_dir, "workflows", f"{workflow_id}.json")
        
        if not os.path.exists(path):
            return False
            
        os.remove(path)
        return True
        
    def export_config(self, config_id: str, export_path: str):
        """Export configuration to file
        
        Args:
            config_id: Configuration ID
            export_path: Path to export to
        """
        # Try loading as agent first
        config = self.load_agent_config(config_id)
        if config:
            with open(export_path, "w") as f:
                json.dump(config.dict(), f, indent=2)
            return
            
        # Try loading as workflow
        config = self.load_workflow_config(config_id)
        if config:
            with open(export_path, "w") as f:
                json.dump(config.dict(), f, indent=2)
            return
            
        raise ValueError(f"No configuration found with ID: {config_id}")
        
    def import_config(self, import_path: str):
        """Import configuration from file
        
        Args:
            import_path: Path to import from
        """
        with open(import_path) as f:
            data = json.load(f)
            
        # Try loading as agent
        try:
            config = AgentConfig(**data)
            self.save_agent_config(config)
            return
        except ValidationError:
            pass
            
        # Try loading as workflow
        try:
            config = WorkflowConfig(**data)
            self.save_workflow_config(config)
            return
        except ValidationError:
            pass
            
        raise ValueError("Invalid configuration format")

class Workflow(BaseModel):
    """Workflow configuration"""
    id: str
    name: Optional[str] = None
    description: str = ''
    agents: List[AgentConfig]
    processors: List[ProcessorConfig] = Field(default_factory=list)
    connections: List[ConnectionConfig]
    metadata: Dict[str, str] = Field(default_factory=dict)
    
    @validator('name', pre=True)
    def set_default_name(cls, v, values):
        """Set default name if not provided"""
        return v or values.get('id', 'default_workflow')
