import os
import json
import configparser
import logging
from typing import Optional, Dict, Any, List

class ConfigManager:
    """Unified configuration management for the project"""
    
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.load_config()
        
    def load_config(self):
        """Load configuration from files and environment variables"""
        # Base config path
        config_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load default config first
        default_config = os.path.join(config_dir, 'default.ini')
        if not self.config.read(default_config):
            raise ValueError(f"Failed to read config file: {default_config}")
        
        # Load environment-specific config
        env = os.getenv('AGENTFLOW_ENV', 'development')
        env_config = os.path.join(config_dir, f'{env}.ini')
        if os.path.exists(env_config):
            self.config.read(env_config)
            
        # Override with environment variables
        self._load_env_variables()
        
    def _load_env_variables(self):
        """Load API keys from environment variables"""
        env_mapping = {
            'ANTHROPIC_API_KEY': ('api_keys', 'anthropic'),
            'OPENAI_API_KEY': ('api_keys', 'openai'),
            'MISTRAL_API_KEY': ('api_keys', 'mistral'),
            'AI21_API_KEY': ('api_keys', 'ai21'),
            'COHERE_API_KEY': ('api_keys', 'cohere'),
            'AWS_ACCESS_KEY_ID': ('api_keys', 'aws_access_key_id'),
            'AWS_SECRET_ACCESS_KEY': ('api_keys', 'aws_secret_access_key'),
            'AWS_REGION': ('api_keys', 'aws_region'),
        }
        
        for env_var, (section, option) in env_mapping.items():
            value = os.getenv(env_var)
            if value:
                if not self.config.has_section(section):
                    self.config.add_section(section)
                self.config.set(section, option, value)
                
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider"""
        return self.config.get('api_keys', provider, fallback=None)
        
    def get_model_settings(self) -> Dict[str, Any]:
        """Get model configuration settings"""
        return dict(self.config['model_settings'])
        
    def get_available_models(self, provider: Optional[str] = None) -> List[str]:
        """Get available models for specified provider or all providers"""
        if provider:
            models_str = self.config.get('available_models', f'{provider}_models', fallback='')
            return [m.strip() for m in models_str.split(',') if m.strip()]
        
        all_models = []
        for option in self.config.options('available_models'):
            if option.endswith('_models'):
                models_str = self.config.get('available_models', option)
                all_models.extend([m.strip() for m in models_str.split(',') if m.strip()])
        return all_models
        
    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get parameters for specific model"""
        params = dict(self.config['model_parameters'])
        model_section = f'model_parameters.{model_name}'
        if self.config.has_section(model_section):
            params.update(dict(self.config[model_section]))
        return params
        
    def get_rate_limits(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limiting settings"""
        if provider:
            section = f'rate_limits.{provider}'
            if self.config.has_section(section):
                return dict(self.config[section])
        return dict(self.config['rate_limits'])
        
    def get_fallback_models(self, provider: str) -> List[str]:
        """Get fallback models for provider"""
        fallbacks = self.config.get('model_fallbacks', provider, fallback='')
        return [m.strip() for m in fallbacks.split(',') if m.strip()]
        
    def get_provider_priority(self) -> List[str]:
        """Get provider priority order"""
        priority = self.config.get('provider_priorities', 'order', fallback='')
        return [p.strip() for p in priority.split(',') if p.strip()]
        
    def setup_logging(self):
        """Configure logging based on settings"""
        logging.basicConfig(
            level=self.config.get('logging', 'level', fallback='INFO'),
            format=self.config.get('logging', 'format', fallback='%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            filename=self.config.get('logging', 'file_path', fallback=None),
            maxBytes=int(self.config.get('logging', 'max_bytes', fallback=10485760)),
            backupCount=int(self.config.get('logging', 'backup_count', fallback=5))
        )

# Global config instance
config = ConfigManager() 