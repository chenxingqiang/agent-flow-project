"""Configuration module."""

from typing import Dict, Any, Optional, List, Union, Type, Callable, cast, TYPE_CHECKING
from enum import Enum
import asyncio
import numpy as np
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict, model_validator, PrivateAttr
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

from .retry_policy import RetryPolicy
from .exceptions import ConfigurationError as BaseConfigurationError, WorkflowExecutionError
from .base_types import AgentType, AgentMode, AgentStatus, DictKeyType, MessageType
from .workflow_types import WorkflowConfig, WorkflowStepType
from .agent_config import AgentConfig, ModelConfig

# Re-export the ConfigurationError from exceptions
ConfigurationError = BaseConfigurationError

if TYPE_CHECKING:
    from ..core.isa.selector import InstructionSelector

class ConfigSecurityManager:
    """Enhanced security management for configurations."""
    
    @classmethod
    def mask_sensitive_fields(cls, config: Union[Dict[str, Any], dict, DictConfig, ListConfig]) -> Union[Dict[str, Any], dict, DictConfig, ListConfig]:
        """
        Mask sensitive fields in a configuration.
        
        Args:
            config: Configuration to mask
            sensitive_keywords: Optional list of additional sensitive keywords
        
        Returns:
            Configuration with sensitive fields masked
        """
        # Default sensitive keywords
        sensitive_keywords = [
            'password', 'secret', 'key', 'token', 
            'credentials', 'api_key', 'access_token', 
            'private_key', 'client_secret', 'email', 'phone', 'username',
            'auth', 'authorization', 'bearer', 'jwt', 'session', 'cookie',
            'database', 'host', 'port', 'url', 'endpoint', 'bucket', 'storage'
        ]
        
        # Create a mutable copy of the config
        config_copy = OmegaConf.create(config)
        
        def is_sensitive_key(key: str) -> bool:
            """Check if a key is sensitive."""
            key_str = str(key)
            return any(sensitive in key_str.lower() for sensitive in sensitive_keywords)
        
        def mask_recursive(item: Any) -> Any:
            """Recursively mask sensitive fields."""
            if isinstance(item, (dict, DictConfig)):
                # Convert all keys to strings
                item = {str(k): v for k, v in item.items()}
                
                # Check if any key is sensitive
                if any(is_sensitive_key(k) for k in item.keys()):
                    return {str(k): '***MASKED***' for k in item.keys()}
                
                # Recursively process nested dicts
                return {str(k): mask_recursive(v) for k, v in item.items()}
            
            # For non-dict items, mask if sensitive
            if isinstance(item, str) and any(sensitive in item.lower() for sensitive in sensitive_keywords):
                return '***MASKED***'
            
            return item
        
        return cast(Union[Dict[str, Any], dict, DictConfig, ListConfig], mask_recursive(config_copy))

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
    def convert_value(cls, value: Any, target_type: type, schema: Dict[str, Any] = Field(default_factory=dict), strict: bool = False) -> Any:
        """Convert a value to the specified target type with Hydra-enhanced type conversion."""
        # Handle None values
        if value is None:
            if target_type == dict:
                return {}
            return value

        # If using Hydra's OmegaConf, leverage its type conversion
        if isinstance(value, (DictConfig, ListConfig)):
            try:
                value = OmegaConf.to_container(value, resolve=True)
            except Exception as e:
                logging.warning(f"Could not convert OmegaConf to object: {e}")
        
        # If the value is already of the target type, return it
        if isinstance(value, target_type):
            return value
        
        try:
            # Handle nested type conversion for dictionaries
            if target_type is dict and isinstance(value, dict):
                # Validate and convert nested dictionary according to schema
                return {str(k): convert_type(v, schema.get(str(k), type(v)), strict=strict) 
                       for k, v in value.items()}
            
            # Handle list type conversion
            if target_type is list and isinstance(value, list):
                return value
            
            # Attempt direct type conversion
            return target_type(value)
        
        except (TypeError, ValueError) as e:
            if strict:
                raise TypeError(f"Could not convert {value} to {target_type}: {e}")
            
            # If conversion fails and strict is False, return empty dict for dict type or original value
            return {} if target_type is dict else value

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
        return cast(Dict[str, Any], OmegaConf.to_container(merged_config))

    @classmethod
    def resolve_inheritance(cls, base_config: Union[Dict[str, Any], DictConfig], override_config: Union[Dict[str, Any], DictConfig]) -> DictConfig:
        """
        Resolve configuration inheritance with Hydra's OmegaConf deep merging.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base
            
        Returns:
            DictConfig: Merged configuration
        """
        # Convert inputs to DictConfig if they aren't already
        if not isinstance(base_config, DictConfig):
            base_config = OmegaConf.create(base_config)
        if not isinstance(override_config, DictConfig):
            override_config = OmegaConf.create(override_config)
            
        # Merge configurations
        merged = OmegaConf.merge(base_config, override_config)
        
        # Ensure we return a DictConfig
        return cast(DictConfig, merged)

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

def load_global_config(config_path: Optional[str] = None, config_name: str = 'base') -> Dict[str, Any]:
    """
    Load the global configuration using Hydra with enhanced error handling.

    Args:
        config_path: Path to configuration directory
        config_name: Name of the configuration file

    Returns:
        Global configuration as a dictionary
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
            if not isinstance(config_dict, dict):
                config_dict = {}
            
            # Create a new dictionary with the 'global_' key
            result: Dict[str, Any] = {
                'global_': {
                    'logging_level': config_dict.get('logging_level', 'INFO') if isinstance(config_dict, dict) else 'INFO'
                }
            }
            
            return result
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

def convert_type(value: Any, target_type: type, schema: Optional[Dict[str, Any]] = None, strict: bool = False) -> Any:
    """Convert a value to the specified target type with Hydra-enhanced type conversion."""
    # If using Hydra's OmegaConf, leverage its type conversion
    if isinstance(value, (DictConfig, ListConfig)):
        try:
            value = OmegaConf.to_container(value, resolve=True)
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
                return {str(k): convert_type(v, schema.get(str(k), type(v)), strict=strict) 
                       for k, v in value.items()}
            return value
        
        # Handle list type conversion
        if target_type is list and isinstance(value, list):
            return value
        
        # Handle None values
        if value is None:
            if strict:
                raise TypeError(f"Cannot convert None to {target_type}")
            return {} if target_type is dict else value
        
        # Attempt direct type conversion
        return target_type(value)
    
    except (TypeError, ValueError) as e:
        if strict:
            raise TypeError(f"Could not convert {value} to {target_type}: {e}")
        
        # If conversion fails and strict is False, return empty dict for dict type or original value
        return {} if target_type is dict else value

class ConfigSecurityManagerHydra:
    @classmethod
    def mask_sensitive_fields(cls, config: Union[Dict[str, Any], DictConfig, Any]) -> Dict[str, Any]:
        """Mask sensitive fields in a configuration dictionary."""
        sensitive_keys = ['password', 'key', 'secret', 'token', 'credentials', 'api_key', 'access_token']
        
        def mask_recursive(cfg: Any) -> Any:
            if isinstance(cfg, (dict, DictConfig)):
                masked = {}
                for k, v in cfg.items():
                    k_str = str(k)
                    if any(sens_key in k_str.lower() if isinstance(k_str, str) else False 
                          for sens_key in sensitive_keys):
                        masked[k_str] = '***MASKED***'
                    elif isinstance(v, (dict, DictConfig)):
                        masked[k_str] = mask_recursive(v)
                    else:
                        masked[k_str] = v
                return masked
            return cfg
        
        return mask_recursive(config)

class ConfigurationInheritanceResolverHydra:
    @classmethod
    def resolve_inheritance(cls, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve configuration inheritance with Hydra's OmegaConf deep merging."""
        # Convert inputs to DictConfig if they aren't already
        base_conf = OmegaConf.create(base_config)
        override_conf = OmegaConf.create(override_config)
            
        # Merge configurations and convert back to dict
        merged = OmegaConf.merge(base_conf, override_conf)
        result = OmegaConf.to_container(merged, resolve=True)
        
        # Ensure we return a dictionary with string keys
        if isinstance(result, dict):
            return {str(k): v for k, v in result.items()}
        return {}

class BaseAgentConfigHydra(BaseModel):
    """Base configuration for agents with Hydra integration."""
    
    @classmethod
    def from_yaml(cls, config_path: str, config_name: str, environment: str = 'default') -> 'BaseAgentConfigHydra':
        """Load configuration from YAML using Hydra's configuration management."""
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
                    if not isinstance(config_dict, dict):
                        config_dict = {}
                except Exception as convert_error:
                    raise ConfigurationError(f"Failed to convert configuration: {convert_error}")

                # Convert all keys to strings to ensure proper type handling
                str_config = {str(k): v for k, v in config_dict.items()}

                # Validate configuration structure
                try:
                    # Create instance with validated dictionary
                    instance = cls(**str_config)
                except Exception as validation_error:
                    raise ConfigurationError(f"Configuration validation failed: {validation_error}")

                # Mask sensitive data for logging
                masked_config = ConfigSecurityManagerHydra.mask_sensitive_fields(str_config)
                logging.info(f"Loaded configuration: {masked_config}")

                return instance

        except Exception as e:
            # Log the error and raise a ConfigurationError
            logging.error(f"Configuration loading error: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseAgentConfigHydra':
        """Create configuration from a dictionary with Hydra-style validation."""
        try:
            # Convert dictionary to OmegaConf for type resolution
            cfg = DictConfig(config_dict)

            # Validate and convert configuration
            validated_config = convert_type(cfg, dict)
            if not isinstance(validated_config, dict):
                validated_config = {}

            # Convert all keys to strings
            str_config = {str(k): v for k, v in validated_config.items()}

            return cls(**str_config)

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

class ConfigTypeConverterHydra:
    """Enhanced type conversion with Hydra integration."""
    
    @classmethod
    def convert_value(cls, value: Any, target_type: type, schema: Dict[str, Any] = Field(default_factory=dict), strict: bool = False) -> Any:
        """Convert a value to the specified target type with Hydra-enhanced type conversion."""
        # Handle None values
        if value is None:
            if target_type == dict:
                return {}
            return value

        # If using Hydra's OmegaConf, leverage its type conversion
        if isinstance(value, (DictConfig, ListConfig)):
            try:
                value = OmegaConf.to_container(value, resolve=True)
            except Exception as e:
                logging.warning(f"Could not convert OmegaConf to object: {e}")
        
        # If the value is already of the target type, return it
        if isinstance(value, target_type):
            return value
        
        try:
            # Handle nested type conversion for dictionaries
            if target_type is dict and isinstance(value, dict):
                # Validate and convert nested dictionary according to schema
                return {str(k): convert_type(v, schema.get(str(k), type(v)), strict=strict) 
                       for k, v in value.items()}
            
            # Handle list type conversion
            if target_type is list and isinstance(value, list):
                return value
            
            # Attempt direct type conversion
            return target_type(value)
        
        except (TypeError, ValueError) as e:
            if strict:
                raise TypeError(f"Could not convert {value} to {target_type}: {e}")
            
            # If conversion fails and strict is False, return empty dict for dict type or original value
            return {} if target_type is dict else value
