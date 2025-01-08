"""Configuration module."""

from typing import Dict, Any, Optional, List, Union, Type, Callable
from enum import Enum
import asyncio
import numpy as np
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
import os
import yaml
import copy
import time
import uuid
from datetime import datetime, date
from typing_extensions import Annotated
import logging
import re
import hashlib
import secrets
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class ConfigSecurityManager:
    """Enhanced security management for configurations."""
    
    @classmethod
    def mask_sensitive_fields(cls, config: Union[Dict[str, Any], dict, DictConfig, ListConfig], sensitive_keywords=None) -> Union[Dict[str, Any], dict, DictConfig, ListConfig]:
        """
        Mask sensitive fields in a configuration.
        
        Args:
            config: Configuration to mask
            sensitive_keywords: Optional list of additional sensitive keywords
        
        Returns:
            Configuration with sensitive fields masked
        """
        # Default sensitive keywords
        if sensitive_keywords is None:
            sensitive_keywords = [
                'password', 'secret', 'key', 'token', 
                'credentials', 'api_key', 'access_token', 
                'private_key', 'client_secret', 'email', 'phone', 'username',
                'auth', 'authorization', 'bearer', 'jwt', 'session', 'cookie',
                'database', 'host', 'port', 'url', 'endpoint', 'bucket', 'storage'
            ]
        
        # Create a mutable copy of the config
        config_copy = OmegaConf.create(config)
        
        def is_sensitive_key(key):
            """Check if a key is sensitive."""
            return any(sensitive in str(key).lower() for sensitive in sensitive_keywords)
        
        def mask_recursive(item):
            """Recursively mask sensitive fields."""
            # If item is a dictionary-like config
            if OmegaConf.is_config(item):
                # Check if any key is sensitive
                def has_sensitive_key(d):
                    for key in d.keys():
                        if is_sensitive_key(key):
                            return True
                        # Recursively check nested configs
                        if OmegaConf.is_config(d[key]) and has_sensitive_key(d[key]):
                            return True
                    return False
                
                # If any key is sensitive, mask the entire config
                if has_sensitive_key(item):
                    masked_dict = OmegaConf.create({})
                    for key in item.keys():
                        masked_dict[key] = '***MASKED***'
                    return masked_dict
                
                # Recursively process nested configs
                for key in item.keys():
                    item[key] = mask_recursive(item[key])
                
                return item
            
            # For non-config items, mask if sensitive
            if isinstance(item, str) and any(sensitive in item.lower() for sensitive in sensitive_keywords):
                return '***MASKED***'
            
            return item
        
        return mask_recursive(config_copy)

    @classmethod
    def hash_sensitive_data(cls, data, salt=None):
        """
        Hash sensitive data with optional salt.
        
        Args:
            data (str): Data to hash
            salt (str, optional): Salt for additional security
        
        Returns:
            str: Hashed data
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        salted_data = f"{salt}{data}"
        return hashlib.sha256(salted_data.encode()).hexdigest()

class ConfigTypeConverter:
    """Enhanced type conversion with robust error handling."""
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against a type schema.

        Args:
            config (dict): Configuration to validate
            schema (dict): Type schema

        Returns:
            dict: Validated configuration
        """
        validated_config = {}
        for key, expected_type in schema.items():
            if key not in config:
                continue
            
            try:
                validated_config[key] = cls.convert_value(config[key], expected_type, strict=False)
            except (ValueError, TypeError, ConfigurationError):
                # Log warning or handle invalid conversion
                logging.warning(f"Invalid type for {key}: expected {expected_type}, got {type(config[key])}")
                validated_config[key] = config[key]
        
        return validated_config

    @classmethod
    def convert_value(cls, value: Any, target_type: type, strict: bool = False, default: Any = None, schema: Dict[str, Any] = None):
        """
        Enhanced type conversion with schema support and optional strict mode.

        Args:
            value: Configuration to convert
            target_type: Target type for conversion
            strict: If True, raise error on invalid conversion
            default: Default value if conversion fails
            schema: Optional type schema for nested conversions
        
        Returns:
            Converted configuration
        """
        # If schema is provided and target_type is dict, validate and convert
        if schema and target_type == dict:
            return cls.validate_config(value, schema)

        # Basic type conversion
        try:
            # Handle boolean conversion
            if target_type == bool and isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'y']

            if target_type == list:
                return [value] if not isinstance(value, list) else value
            elif target_type == dict:
                if isinstance(value, list):
                    return {str(i): v for i, v in enumerate(value)}
                elif value is not None:
                    return {'value': value}
                return {}

            # Handle date conversion
            if target_type == date and isinstance(value, str):
                return date.fromisoformat(value)

            # Standard type conversion
            if not isinstance(value, target_type):
                # Special handling for numeric conversions
                if target_type in (int, float) and isinstance(value, str):
                    return target_type(value)
                
                if strict:
                    raise ConfigurationError(f"Cannot convert {value} to {target_type}")
                return default if default is not None else value
            return value
        except (ValueError, TypeError) as e:
            if strict:
                raise ConfigurationError(f"Cannot convert {value} to {target_type}: {str(e)}")
            return default if default is not None else value

class ConfigurationInheritanceResolver:
    """
    Resolver for configuration inheritance with advanced merging capabilities.
    """
    @classmethod
    def _load_yaml_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            config_path (str): Path to the YAML configuration file
        
        Returns:
            dict: Loaded configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    @classmethod
    def load_config_with_inheritance(cls, config_path: str, config_name: str, environment: str) -> Dict[str, Any]:
        """
        Load configuration with inheritance.
        
        Args:
            config_path (str): Base path for configurations
            config_name (str): Base configuration name
            environment (str): Environment name
        
        Returns:
            dict: Merged configuration
        """
        base_config = cls._load_yaml_config(os.path.join(config_path, f"{config_name}.yaml"))
        env_config = cls._load_yaml_config(os.path.join(config_path, f"{config_name}.{environment}.yaml"))
        
        # Convert to OmegaConf for merging
        base_config_obj = OmegaConf.create(base_config)
        env_config_obj = OmegaConf.create(env_config)
        
        # Resolve inheritance
        merged_config = cls.resolve_inheritance(base_config_obj, env_config_obj)
        
        # Convert back to dictionary
        return OmegaConf.to_container(merged_config)

    @classmethod
    def resolve_inheritance(cls, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> DictConfig:
        """
        Resolve configuration inheritance with Hydra's OmegaConf deep merging.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base
        
        Returns:
            Merged configuration as OmegaConf DictConfig
        """
        # Ensure both configs are OmegaConf instances
        if not isinstance(base_config, DictConfig):
            base_config = OmegaConf.create(base_config)
        if not isinstance(override_config, DictConfig):
            override_config = OmegaConf.create(override_config)
        
        # Perform deep merge
        merged_config = OmegaConf.merge(base_config, override_config)
        
        return merged_config

class BaseAgentConfig(BaseModel):
    """
    Advanced base configuration with comprehensive validation.
    """
    model_config = ConfigDict(
        extra='allow',
        use_enum_values=True
    )
    
    name: str = Field(default="default_research_agent")
    type: str = Field(default="research")
    version: str = Field(default="1.0.0")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseAgentConfig':
        """
        Create configuration from dictionary with strict validation.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Configuration instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Use Pydantic's model_validate to apply field validators
            return cls.model_validate(config_dict)
        except ValidationError as e:
            # Convert Pydantic validation error to our custom ConfigurationError
            error_messages = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(error_messages)}")
        except Exception as e:
            raise ConfigurationError(f"Unexpected error during configuration creation: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return self.model_dump()

    @classmethod
    def from_yaml(cls,
                  config_path: Optional[str] = None,
                  config_name: str = 'base_agent',
                  environment: str = 'default') -> 'BaseAgentConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Configuration directory
            config_name: Base configuration name
            environment: Environment-specific configuration
        
        Returns:
            Configuration instance
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'agents')
        
        # Try multiple file naming conventions
        config_file_variants = [
            os.path.join(config_path, f"{config_name}_{environment}.yaml"),
            os.path.join(config_path, f"{config_name}.yaml"),
            os.path.join(config_path, f"{config_name}.yml")
        ]
        
        config_file = None
        for variant in config_file_variants:
            if os.path.exists(variant):
                config_file = variant
                break
        
        if not config_file:
            raise FileNotFoundError(f"No configuration file found for {config_name}")
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)

class ResearchAgentConfig(BaseAgentConfig):
    """
    Specialized configuration for research-oriented agents with enhanced validation.
    """
    research_context: Optional[Dict[str, Any]] = None
    publication_goals: Optional[Dict[str, Any]] = Field(default_factory=dict)
    challenges: Optional[Dict[str, Any]] = None
    
    @field_validator('name')
    @classmethod
    def validate_agent_name(cls, name: str) -> str:
        """
        Validate agent name with additional rules.
        
        Args:
            name: Agent name to validate
        
        Returns:
            Validated name
        """
        if not name or len(name) < 3:
            raise ValueError("Agent name must be at least 3 characters long")
        
        # Optional: Add more complex name validation
        if not re.match(r'^[A-Za-z0-9_\s-]+$', name):
            raise ValueError("Agent name can only contain letters, numbers, spaces, underscores, and hyphens")
        
        return name
    
    @field_validator('publication_goals')
    @classmethod
    def validate_publication_goals(cls, goals: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Validate publication goals with additional checks.
        
        Args:
            goals: Publication goals dictionary
        
        Returns:
            Validated publication goals
        """
        if not goals:
            return goals
        
        # Validate deadline is a date
        if 'acceptance_deadline' in goals:
            deadline = goals['acceptance_deadline']
            if isinstance(deadline, str):
                try:
                    goals['acceptance_deadline'] = date.fromisoformat(deadline)
                except ValueError:
                    raise ValueError(f"Invalid date format for acceptance_deadline: {deadline}")
        
        return goals

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ResearchAgentConfig':
        """
        Create configuration from dictionary with specialized validation.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ResearchAgentConfig instance
        """
        # Validate publication goals before creating the config
        if 'publication_goals' in config_dict:
            config_dict['publication_goals'] = cls.validate_publication_goals(config_dict.get('publication_goals'))

        return super().from_dict(config_dict)

    @classmethod
    def from_yaml(cls, 
                  config_path: Optional[str] = None, 
                  config_name: str = 'academic_support_agent', 
                  environment: str = 'default') -> 'ResearchAgentConfig':
        """
        Override from_yaml to set default name for academic support agent.
        
        Args:
            config_path: Configuration directory
            config_name: Base configuration name
            environment: Environment-specific configuration
        
        Returns:
            Configuration instance
        """
        config = super().from_yaml(config_path, config_name, environment)
        
        # Set specific name and type for academic support agent
        if config_name == 'academic_support_agent':
            config.name = 'Academic_Research_Support_Agent'
            config.type = 'research_optimization'
        
        return config

class ConfigurationType(str, Enum):
    """Configuration type enumeration."""
    AGENT = "agent"
    MODEL = "model"
    WORKFLOW = "workflow"
    GENERIC = "generic"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DATA_SCIENCE = "data_science"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"

class AgentMode(str, Enum):
    """Agent mode enumeration."""
    SIMPLE = "simple"
    ADVANCED = "advanced"
    EXPERT = "expert"
    AUTO = "auto"
    MANUAL = "manual"

class WorkflowStepType(str, Enum):
    """Workflow step type enumeration."""
    TRANSFORM = "transform"
    ANALYZE = "analyze"
    DECIDE = "decide"
    EXECUTE = "execute"
    VALIDATE = "validate"
    MONITOR = "monitor"

class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = Field(default="gpt-4")
    provider: str = Field(default="openai")
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    
    @classmethod
    def from_yaml(cls, config_path: str = None, config_name: str = 'base') -> 'ModelConfig':
        """
        Load model configuration from YAML using Hydra.

        Args:
            config_path: Path to configuration directory
            config_name: Name of the configuration file

        Returns:
            ModelConfig instance
        """
        if config_path is None:
            config_path = os.path.join('..', '..', 'config')

        try:
            with hydra.initialize(config_path=config_path, version_base=None):
                cfg = hydra.compose(config_name=config_name)
                return cls(**cfg.model)
        except Exception as e:
            logging.error(f"Error loading model configuration: {e}")
            raise ConfigurationError(f"Failed to load model configuration: {e}")

class StepConfig(BaseModel):
    """Step configuration."""
    strategy: str
    params: Dict[str, Any] = Field(default_factory=dict)
    
class WorkflowStep(BaseModel):
    """Workflow step configuration."""
    id: str
    name: str
    type: WorkflowStepType
    config: StepConfig
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step.
        
        Args:
            context: Execution context
            
        Returns:
            Step execution result
        """
        if self.type == WorkflowStepType.TRANSFORM:
            if self.config.strategy == "feature_engineering":
                data = context["data"]
                params = self.config.params
                method = params.get("method", "standard")
                if method == "standard":
                    with_mean = params.get("with_mean", True)
                    with_std = params.get("with_std", True)
                    mean = np.mean(data, axis=0) if with_mean else 0
                    std = np.std(data, axis=0) if with_std else 1
                    transformed_data = (data - mean) / std
                    return {"output": {"data": transformed_data}}
                elif method == "isolation_forest":
                    threshold = params.get("threshold", 0.1)
                    # Simple outlier detection using mean and std
                    mean = np.mean(data, axis=0)
                    std = np.std(data, axis=0)
                    z_scores = np.abs((data - mean) / std)
                    outliers = np.any(z_scores > threshold, axis=1)
                    transformed_data = data[~outliers]
                    return {"output": {"data": transformed_data}}
                    
        return {"output": context}

class WorkflowConfig(BaseModel):
    """Workflow configuration."""
    model_config = ConfigDict(
        extra='allow',
        use_enum_values=True
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="default_workflow")
    max_iterations: int = Field(default=10)
    timeout: int = Field(default=3600)
    logging_level: str = Field(default="INFO")
    required_fields: List[str] = Field(default_factory=list)
    error_handling: Dict[str, str] = Field(default_factory=dict)
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {"max_retries": 3, "retry_delay": 1.0})
    error_policy: Dict[str, bool] = Field(default_factory=lambda: {"ignore_warnings": False, "fail_fast": True})
    steps: List[WorkflowStep] = Field(default_factory=list)
    ell2a_config: Optional[Dict[str, Any]] = Field(default=None)
    
    @classmethod
    def from_yaml(cls, config_path: str = None, config_name: str = 'base_workflow') -> 'WorkflowConfig':
        """
        Load workflow configuration from YAML using Hydra.

        Args:
            config_path: Path to configuration directory
            config_name: Name of the configuration file

        Returns:
            WorkflowConfig instance
        """
        if config_path is None:
            config_path = os.path.join('..', '..', 'config')

        try:
            with hydra.initialize(config_path=config_path, version_base=None):
                cfg = hydra.compose(config_name=config_name)
                return cls(**cfg.workflow)
        except Exception as e:
            logging.error(f"Error loading workflow configuration: {e}")
            # Return a default configuration if loading fails
            return cls(
                name="default_workflow",
                max_iterations=10,
                timeout=3600,
                logging_level='INFO'
            )

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow.
        
        Args:
            context: Execution context
            
        Returns:
            Execution result
        """
        result = {}
        current_context = context.copy()
        
        # Execute steps sequentially
        for step in self.steps:
            step_result = await step.execute(current_context)
            result[step.id] = step_result
            current_context.update(step_result)
            
        return result

class AgentConfig(BaseAgentConfig):
    """
    Comprehensive configuration for an agent using Hydra and Pydantic.
    """
    name: str = Field(default="default_agent")
    type: str = Field(default="generic")
    version: str = Field(default="1.0.0")
    system_prompt: Optional[str] = Field(default=None)
    
    @classmethod
    def from_yaml(cls, 
                  config_path: Optional[str] = None, 
                  config_name: str = 'academic_support_agent') -> 'AgentConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration directory
            config_name: Name of the configuration file
        
        Returns:
            AgentConfig instance
        """
        # Determine the relative path from the current file
        if config_path is None:
            config_path = os.path.join('..', '..', 'config', 'agents')
        
        try:
            # Ensure the path is relative to the current file
            base_dir = os.path.dirname(__file__)
            relative_config_path = os.path.relpath(
                os.path.normpath(os.path.join(base_dir, config_path)), 
                base_dir
            )
            
            with hydra.initialize(config_path=relative_config_path, version_base=None):
                config_dict = hydra.compose(config_name=config_name)
                
                # Convert to dictionary to ensure mutability
                config_dict = OmegaConf.to_container(config_dict, resolve=True)
                
                # Set default values for missing attributes
                config_dict['name'] = config_dict.get('name', 'default_agent')
                config_dict['type'] = config_dict.get('type', 'generic')
                config_dict['version'] = config_dict.get('version', '1.0.0')
                
                # Add default capabilities if not present
                config_dict['capabilities'] = config_dict.get('capabilities', ['generic_agent_capability'])
                
                # Add default optimization and other attributes
                config_dict['optimization'] = config_dict.get('optimization', {
                    'strategy': 'default_optimization',
                    'goals': []
                })
                
                # Add default workflow if not present
                config_dict['workflow'] = config_dict.get('workflow', {
                    'steps': [],
                    'default_mode': 'sequential'
                })
                
                return cls.from_dict(config_dict)
        except Exception as e:
            logging.error(f"Error loading agent configuration: {e}")
            raise ConfigurationError(f"Failed to load agent configuration: {e}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """
        Create AgentConfig from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            AgentConfig instance
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert AgentConfig to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            k: v for k, v in self.model_dump().items() 
            if v is not None
        }

def load_global_config(config_path: Optional[str] = None, config_name: str = 'base') -> DictConfig:
    """
    Load the global configuration using Hydra with enhanced error handling.

    Args:
        config_path: Path to configuration directory
        config_name: Name of the configuration file

    Returns:
        Global configuration as a DictConfig
    """
    if config_path is None:
        config_path = os.path.join('..', '..', 'config')
    
    try:
        # Ensure the path is relative to the current file
        base_dir = os.path.dirname(__file__)
        relative_config_path = os.path.relpath(
            os.path.normpath(os.path.join(base_dir, config_path)), 
            base_dir
        )
        
        with hydra.initialize(config_path=relative_config_path, version_base=None):
            config = hydra.compose(config_name=config_name)
            
            # Convert to dictionary to ensure mutability
            config_dict = OmegaConf.to_container(config, resolve=True)
            
            # Create a new dictionary with the 'global_' key
            config_dict['global_'] = {
                'logging_level': config_dict.get('logging_level', 'INFO')
            }
            
            # Convert back to OmegaConf
            return OmegaConf.create(config_dict)
    except Exception as e:
        logging.error(f"Error loading global configuration: {e}")
        raise ConfigurationError(f"Failed to load global configuration: {e}")

def _parse_publication_goals(goals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse publication goals, converting date strings to date objects.
    
    Args:
        goals: Dictionary of publication goals
    
    Returns:
        Parsed publication goals
    """
    if 'acceptance_deadline' in goals and isinstance(goals['acceptance_deadline'], str):
        try:
            goals['acceptance_deadline'] = datetime.strptime(goals['acceptance_deadline'], '%Y-%m-%d').date()
        except ValueError:
            # Keep original string if parsing fails
            pass
    return goals

class ConfigSecurityManagerHydra:
    @classmethod
    def mask_sensitive_fields(cls, config: Union[Dict, DictConfig, Any]) -> Dict:
        """
        Mask sensitive fields in a configuration dictionary.
        
        Args:
            config: Configuration dictionary or OmegaConf configuration
        
        Returns:
            Configuration with sensitive fields masked
        """
        sensitive_keys = ['password', 'key', 'secret', 'token', 'credentials', 'api_key', 'access_token', 'private_key', 'client_secret', 'email', 'phone', 'username', 'auth', 'authorization', 'bearer', 'jwt', 'session', 'cookie']
        
        def mask_recursive(cfg):
            if isinstance(cfg, dict):
                masked = {}
                for k, v in cfg.items():
                    if any(sens_key in k.lower() for sens_key in sensitive_keys):
                        masked[k] = '***MASKED***'
                    elif isinstance(v, (dict, DictConfig)):
                        masked[k] = mask_recursive(v)
                    else:
                        masked[k] = v
                return masked
            elif isinstance(cfg, DictConfig):
                masked = {}
                for k, v in cfg.items():
                    if any(sens_key in k.lower() for sens_key in sensitive_keys):
                        masked[k] = '***MASKED***'
                    elif isinstance(v, (dict, DictConfig)):
                        masked[k] = mask_recursive(v)
                    else:
                        masked[k] = v
                return masked
            return cfg
        
        return mask_recursive(config)

class ConfigTypeConverterHydra:
    @classmethod
    def convert_value(cls, value: Any, target_type: type, schema: Dict[str, Any] = None, strict: bool = False) -> Any:
        """
        Convert a value to the specified target type with Hydra-enhanced type conversion.
        
        Args:
            value: The value to convert
            target_type: The target type for conversion
            schema: Schema for nested type validation
            strict: If True, raises exceptions on conversion failures
        
        Returns:
            Converted value or original value
        """
        # If using Hydra's OmegaConf, leverage its type conversion
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)
        
        def _convert_single_value(val, expected_type):
            if val is None:
                return None
            
            try:
                # Handle boolean conversion
                if expected_type == bool:
                    if isinstance(val, bool):
                        return val
                    val_str = str(val).lower()
                    if val_str in ['true', '1', 'yes', 'y']:
                        return True
                    if val_str in ['false', '0', 'no', 'n']:
                        return False
                    if strict:
                        raise ValueError(f"Cannot convert {val} to bool")
                    return None

                # Handle numeric conversions
                if expected_type in (int, float):
                    try:
                        return expected_type(val)
                    except ValueError:
                        if strict:
                            raise
                        return None

                # Handle date conversion
                if expected_type == date:
                    try:
                        return date.fromisoformat(str(val))
                    except ValueError:
                        if strict:
                            raise
                        return None

                # Handle list conversion
                if expected_type == list:
                    if isinstance(val, list):
                        return val
                    return [val]

                # Handle dict conversion
                if expected_type == dict:
                    if isinstance(val, dict):
                        return val
                    return {'value': val}

                # Default fallback
                return val
            
            except Exception as e:
                if strict:
                    raise
                return None

        # Nested schema validation
        if schema and isinstance(value, dict):
            converted_config = {}
            for key, expected_type in schema.items():
                # Skip if the key is not in the config
                if key not in value:
                    continue
                
                if isinstance(expected_type, dict):
                    converted_config[key] = {}
                    for nested_key, nested_type in expected_type.items():
                        converted_config[key][nested_key] = _convert_single_value(
                            value[key].get(nested_key), 
                            nested_type
                        )
                else:
                    # Convert single values
                    converted_config[key] = _convert_single_value(
                        value[key], 
                        expected_type
                    )
            
            return converted_config
        
        return _convert_single_value(value, target_type)

class ConfigurationInheritanceResolverHydra:
    @classmethod
    def resolve_inheritance(cls, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> DictConfig:
        """
        Resolve configuration inheritance with Hydra's OmegaConf deep merging.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base
        
        Returns:
            Merged configuration as OmegaConf DictConfig
        """
        # Ensure both configs are OmegaConf instances
        if not isinstance(base_config, DictConfig):
            base_config = OmegaConf.create(base_config)
        if not isinstance(override_config, DictConfig):
            override_config = OmegaConf.create(override_config)
        
        # Perform deep merge
        merged_config = OmegaConf.merge(base_config, override_config)
        
        return merged_config

class BaseAgentConfigHydra(BaseModel):
    """Base configuration for agents with Hydra integration."""
    
    @classmethod
    def from_yaml(cls, config_path: str, config_name: str, environment: str = 'default') -> 'BaseAgentConfigHydra':
        """
        Load configuration from YAML using Hydra's configuration management.

        Args:
            config_path: Base configuration directory
            config_name: Base configuration name
            environment: Environment-specific configuration

        Returns:
            Instantiated configuration object

        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        try:
            # Validate config path
            if not config_path or not os.path.exists(config_path):
                raise ConfigurationError(f"Configuration path {config_path} does not exist.")

            # Validate config name
            if not config_name:
                raise ConfigurationError("Configuration name cannot be empty.")

            # Use Hydra's composition to merge configurations
            with hydra.initialize(version_base=None, config_path=os.path.relpath(config_path)):
                try:
                    # Compose configuration across environments
                    cfg = hydra.compose(
                        config_name=config_name,
                        overrides=[f'+group={environment}']
                    )
                except Exception as compose_error:
                    raise ConfigurationError(f"Failed to compose configuration: {compose_error}")

                # Convert to dictionary and validate
                try:
                    config_dict = OmegaConf.to_container(cfg, resolve=True)
                except Exception as convert_error:
                    raise ConfigurationError(f"Failed to convert configuration: {convert_error}")

                # Validate configuration structure
                try:
                    # Attempt to create an instance to validate structure
                    cls(**config_dict)
                except Exception as validation_error:
                    raise ConfigurationError(f"Configuration validation failed: {validation_error}")

                # Mask sensitive data for logging
                masked_config = ConfigSecurityManagerHydra.mask_sensitive_fields(config_dict)
                logging.info(f"Loaded configuration: {masked_config}")

                return cls(**config_dict)

        except (ValidationError, FileNotFoundError, ValueError, hydra.errors.HydraException) as e:
            # Log the error and raise a ConfigurationError
            logging.error(f"Configuration loading error: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}") from e

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseAgentConfigHydra':
        """
        Create configuration from a dictionary with Hydra-style validation.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Instantiated configuration object

        Raises:
            ConfigurationError: If dictionary validation fails
        """
        try:
            # Convert dictionary to OmegaConf for type resolution
            cfg = DictConfig(config_dict)

            # Validate and convert configuration
            validated_config = ConfigTypeConverterHydra.convert_value(
                cfg, 
                dict
            )

            return cls(**validated_config)

        except (ValidationError, ValueError) as e:
            logging.error(f"Configuration validation error: {e}")
            raise ConfigurationError(f"Invalid configuration: {e}")

class ResearchAgentConfigHydra(BaseAgentConfigHydra):
    """Specialized configuration for research-oriented agents with Hydra validation."""
    pass

class DynamicConfigGenerator:
    """
    Utility class for generating dynamic configurations with advanced features.
    """
    
    @classmethod
    def generate_config_from_function(cls, config_generator: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a configuration by calling a configuration generator function.
        
        Args:
            config_generator (callable): A function that returns a configuration dictionary
        
        Returns:
            dict: Generated configuration
        """
        return config_generator()
    
    @classmethod
    def generate_timestamped_config(cls, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a timestamped configuration by adding metadata to a base configuration.
        
        Args:
            base_config (dict): Base configuration dictionary
        
        Returns:
            dict: Configuration with added metadata
        """
        timestamped_config = copy.deepcopy(base_config)
        timestamped_config['_metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'version': str(uuid.uuid4())[:8]
        }
        return timestamped_config

from omegaconf import OmegaConf, DictConfig, ListConfig
import logging

def convert_type(value, target_type, schema=None, strict: bool = False) -> Any:
    """
    Convert a value to the specified target type with Hydra-enhanced type conversion.

    Args:
        value: The value to convert
        target_type: The target type for conversion
        schema: Schema for nested type validation
        strict: If True, raises exceptions on conversion failures

    Returns:
        Converted value or original value
    """
    # If using Hydra's OmegaConf, leverage its type conversion
    if isinstance(value, (DictConfig, ListConfig)):
        try:
            # Convert OmegaConf to a standard dictionary or list
            value = OmegaConf.to_object(value)
        except Exception as e:
            logging.warning(f"Could not convert OmegaConf to object: {e}")
    
    # If the value is already of the target type, return it
    if isinstance(value, target_type):
        return value
    
    try:
        # Handle nested type conversion for dictionaries
        if target_type is dict and isinstance(value, dict):
            if schema:
                # Validate and convert nested dictionary according to schema
                return {k: convert_type(v, schema.get(k, type(v)), strict=strict) for k, v in value.items()}
            return value
        
        # Handle list type conversion
        if target_type is list and isinstance(value, list):
            return value
        
        # Attempt direct type conversion
        return target_type(value)
    
    except (TypeError, ValueError) as e:
        if strict:
            raise TypeError(f"Could not convert {value} to {target_type}: {e}")
        
        # If conversion fails and strict is False, return the original value
        logging.warning(f"Type conversion failed for {value} to {target_type}: {e}")
        return value
