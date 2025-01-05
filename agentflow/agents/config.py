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