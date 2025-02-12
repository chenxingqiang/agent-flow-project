"""Model configuration module."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict

class ModelConfig(BaseModel):
    """Model configuration."""
    
    model_config = ConfigDict(
        frozen=False,
        arbitrary_types_allowed=True,
        extra='allow'
    )
    
    provider: str = Field(default="openai", description="Model provider (e.g., OpenAI, Anthropic)")
    name: str = Field(default="gpt-4", description="Model name")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_tokens: Optional[int] = Field(default=4096, description="Maximum number of tokens")
    top_p: float = Field(default=1.0, description="Top-p sampling parameter")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(default=None, description="List of stop sequences")
    api_key: Optional[str] = Field(default=None, description="API key for the model provider")
    api_base: Optional[str] = Field(default=None, description="Base URL for API requests")
    api_version: Optional[str] = Field(default=None, description="API version")
    api_type: Optional[str] = Field(default=None, description="API type")
    organization_id: Optional[str] = Field(default=None, description="Organization ID for the model provider")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model-specific parameters")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Model metadata")

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().model_dump(**kwargs)
        return data

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> 'ModelConfig':
        """Validate and create instance."""
        if isinstance(obj, ModelConfig):
            return obj
        if isinstance(obj, dict):
            # Validate provider
            provider = obj.get('provider', 'openai')
            valid_providers = {'openai', 'anthropic', 'mistral', 'cohere', 'ai21', 'default'}
            if provider not in valid_providers:
                raise ValueError(f"Invalid provider: {provider}. Must be one of {valid_providers}")
            # Convert stop sequence to proper format
            if 'stop' in obj and isinstance(obj['stop'], (str, list)):
                obj['stop'] = obj['stop']
        return super().model_validate(obj, **kwargs)

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration.
        
        Returns:
            Dict[str, Any]: API configuration
        """
        config = {
            "provider": self.provider,
            "name": self.name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop
        }
        
        # Add optional API configuration
        if self.api_key:
            config["api_key"] = self.api_key
        if self.api_base:
            config["api_base"] = self.api_base
        if self.api_version:
            config["api_version"] = self.api_version
        if self.api_type:
            config["api_type"] = self.api_type
        if self.organization_id:
            config["organization_id"] = self.organization_id
            
        # Add additional parameters
        config.update(self.additional_params)
        
        return config 