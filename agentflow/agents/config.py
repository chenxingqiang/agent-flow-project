"""Agent configuration module."""

from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from uuid import uuid4

class AgentType(str, Enum):
    """Agent type enum."""
    
    GENERIC = "generic"
    RESEARCH = "research"
    DATA_SCIENCE = "data_science"
    WORKFLOW = "workflow"

class AgentConfig(BaseModel):
    """Agent configuration."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: AgentType = AgentType.GENERIC
    version: str = "1.0.0"
    description: Optional[str] = None
    is_distributed: bool = False
    distributed: bool = False
    model_provider: str = "openai"
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    system_prompt: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create AgentConfig from a dictionary, extracting system prompt from AGENT config."""
        # Create a copy to avoid modifying the original dictionary
        config_copy = config_dict.copy()
        
        # Extract system prompt from AGENT config if not directly specified
        if 'AGENT' in config_copy and 'system_prompt' in config_copy['AGENT']:
            config_copy['system_prompt'] = config_copy['AGENT']['system_prompt']
            config_copy['name'] = config_copy['AGENT'].get('name', '')
        
        # Remove nested AGENT configuration to prevent conflicts
        config_copy.pop('AGENT', None)
        
        return cls(**config_copy)