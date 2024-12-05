import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

from pydantic import BaseModel, Field, field_validator, ConfigDict

class ModelConfig(BaseModel):
    """Configuration for AI model"""
    provider: str = Field(..., description="AI model provider")
    name: str = Field(..., description="Model name")
    temperature: float = Field(default=0.5, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum token limit")
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        """Validate model provider"""
        supported_providers = ['openai', 'anthropic', 'google', 'azure', 'ollama']
        if v.lower() not in supported_providers:
            raise ValueError(f"Unsupported model provider: {v}. Supported providers: {supported_providers}")
        return v.lower()

class WorkflowStepConfig(BaseModel):
    """Configuration for a workflow step"""
    type: str
    config: Dict[str, Any] = {}

class WorkflowConfig(BaseModel):
    """Configuration for workflow execution"""
    max_iterations: int = Field(default=10, ge=1, le=20, description="Maximum workflow iterations")
    logging_level: str = Field(default="INFO", description="Logging level")
    distributed: bool = Field(default=False, description="Enable distributed execution")
    memory_limit: Optional[int] = Field(default=None, description="Memory limit for workflow")
    timeout: Optional[int] = Field(default=None, description="Workflow timeout in seconds")
    steps: Optional[List[WorkflowStepConfig]] = None
    error_handling: str = "strict"
    
    @field_validator('logging_level')
    @classmethod
    def validate_logging_level(cls, v):
        """Validate logging level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid logging level: {v}. Valid levels: {valid_levels}")
        return v.upper()

class AgentConfig(BaseModel):
    """Configuration for AI agent"""
    model: ModelConfig
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    agent_type: str = Field(..., description="Type of agent")
    
    model_config = ConfigDict(
        extra='forbid',  # Prevent additional fields
        frozen=True  # Make configuration immutable
    )
    
    @field_validator('agent_type')
    @classmethod
    def validate_agent_type(cls, v):
        """Validate agent type"""
        supported_types = ['research', 'coding', 'analysis', 'creative']
        if v.lower() not in supported_types:
            raise ValueError(f"Unsupported agent type: {v}. Supported types: {supported_types}")
        return v.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.model_dump()

class ConfigManager:
    """Configuration manager for handling config files and validation"""
    
    def __init__(self, config_path: str):
        """Initialize config manager
        
        Args:
            config_path: Path to config file
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path) as f:
                config = json.load(f)
                
            # Add variables section if not present
            if 'variables' not in config:
                config['variables'] = {}
                
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise
            
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If config is invalid
        """
        required_sections = {'variables', 'agent_type', 'model', 'workflow'}
        missing = required_sections - set(config.keys())
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
            
        # Validate model config
        model_config = config.get('model', {})
        if not all(k in model_config for k in ['provider', 'name']):
            raise ValueError("Model config missing required fields")
            
        # Validate workflow config
        workflow_config = config.get('workflow', {})
        if not isinstance(workflow_config, dict):
            raise ValueError("Workflow config must be a dictionary")
            
    def extract_variables(self) -> Dict[str, Any]:
        """Extract variables from config
        
        Returns:
            Dictionary of variables
        """
        return self.config.get('variables', {})
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration
        
        Args:
            updates: Configuration updates to apply
            
        Raises:
            ValueError: If updates are invalid
        """
        # Validate updates
        for key, value in updates.items():
            if key not in self.config:
                raise ValueError(f"Invalid config key: {key}")
                
            if key == 'variables':
                if not isinstance(value, dict):
                    raise ValueError("Variables must be a dictionary")
                    
        # Apply updates
        self.config.update(updates)
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

class ConfigValidator:
    @staticmethod
    def validate_workflow_config(config: Dict[str, Any], workflow_def: Dict[str, Any]):
        """Validate workflow configuration"""
        if not config:
            raise ValueError("Empty configuration")
            
        if not workflow_def or 'WORKFLOW' not in workflow_def:
            raise ValueError("Invalid workflow definition")
            
        # Validate step configurations
        for step in workflow_def['WORKFLOW']:
            step_num = step['step']
            step_config_key = f'step_{step_num}_config'
            
            if step_config_key not in config:
                raise ValueError(f"Missing configuration for step {step_num}")
                
            step_config = config[step_config_key]
            if not isinstance(step_config, dict):
                raise ValueError(f"Invalid configuration for step {step_num}")
