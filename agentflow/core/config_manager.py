"""
Configuration management for AgentFlow
"""

import json
import os
from typing import Dict, List, Optional, Union, Any, Type
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationError, validator
from typing_extensions import Literal

class ToolConfig(BaseModel):
    """Tool configuration"""
    name: str
    description: str
    parameters: Dict[str, dict]
    required: List[str]
    
class ModelConfig(BaseModel):
    """Language model configuration"""
    name: str
    provider: str
    parameters: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)
    
class AgentConfig(BaseModel):
    """Agent configuration"""
    id: str
    name: str
    description: str
    type: str
    model: ModelConfig
    system_prompt: str
    tools: List[ToolConfig] = Field(default_factory=list)
    memory_config: Dict[str, Union[str, int]] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)
    execution_policies: Dict[str, Any] = Field(default_factory=dict)
    
    # Optional processor configuration
    processor: Optional[Union[str, Type]] = None
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
    
class ProcessorConfig(BaseModel):
    """Processor node configuration"""
    id: str
    name: Optional[str] = None
    type: Literal['processor', 'agent'] = 'processor'
    processor: Union[str, Type]
    input_format: Dict[str, str] = Field(default_factory=dict)
    output_format: Dict[str, str] = Field(default_factory=dict)
    processing_rules: List[Dict[str, str]] = Field(default_factory=list)
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
    
    @validator('name', pre=True)
    def set_default_name(cls, v, values):
        """Set default name if not provided"""
        return v or values.get('id', 'default_processor')
    
class ConnectionConfig(BaseModel):
    """Node connection configuration"""
    source_id: str
    target_id: str
    source_port: str = 'output'
    target_port: str = 'input'
    
class WorkflowConfig(BaseModel):
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
