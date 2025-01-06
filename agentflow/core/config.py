"""Configuration module for AgentFlow."""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict

class ConfigurationType(Enum):
    """Configuration type enumeration."""
    AGENT = "agent"
    MODEL = "model"
    WORKFLOW = "workflow"
    GENERIC = "generic"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    DATA_SCIENCE = "data_science"
    TEST = "test"

class AgentMode(Enum):
    """Agent mode enumeration."""
    SIMPLE = "simple"
    COMPLEX = "complex"
    HYBRID = "hybrid"

class ModelConfig(BaseModel):
    """Model configuration."""
    
    provider: str = Field(default="openai")
    name: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    
    model_config = ConfigDict(frozen=True)
    
    @property
    def dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()

class WorkflowStep(BaseModel):
    """Workflow step configuration."""
    type: str
    config: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(frozen=True)

class WorkflowConfig(BaseModel):
    """Workflow configuration."""
    max_iterations: int = Field(default=10)
    timeout: float = Field(default=3600)
    logging_level: str = Field(default="INFO")
    required_fields: List[str] = Field(default_factory=list)
    error_handling: Dict[str, str] = Field(default_factory=dict)
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {"max_retries": 3, "retry_delay": 1.0})
    error_policy: Dict[str, bool] = Field(default_factory=lambda: {"ignore_warnings": False, "fail_fast": True})
    steps: List[WorkflowStep] = Field(default_factory=list)
    
    model_config = ConfigDict(frozen=True)
    
    @property
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

class AgentConfig(BaseModel):
    """Agent configuration."""
    id: Optional[str] = None
    name: Optional[str] = None
    type: str = Field(default="research")
    description: Optional[str] = None
    version: str = Field(default="1.0.0")
    model: Optional[ModelConfig] = None
    workflow: Optional[WorkflowConfig] = Field(default_factory=WorkflowConfig)
    workflow_path: Optional[str] = None
    domain_config: Dict[str, Any] = Field(default_factory=dict)
    max_retries: int = Field(default=3)
    
    model_config = ConfigDict(frozen=True)
    
    @property
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    def __init__(self, **data):
        """Initialize agent configuration."""
        # Set default model if not provided
        if "model" not in data:
            data["model"] = ModelConfig()
            
        # Validate provider
        if "model" in data and isinstance(data["model"], dict) and "provider" in data["model"]:
            provider = data["model"]["provider"]
            if provider not in ["openai", "anthropic"]:
                raise ValueError(f"Unsupported provider: {provider}")
        
        # Validate type
        if "type" in data and data["type"] not in ["research", "analysis", "creative", "technical", "data_science"]:
            raise ValueError(f"Invalid agent type: {data['type']}")
            
        # Convert workflow dict to WorkflowConfig
        if "workflow" in data and isinstance(data["workflow"], dict):
            data["workflow"] = WorkflowConfig(**data["workflow"])
            
        # Convert model dict to ModelConfig
        if "model" in data and isinstance(data["model"], dict):
            data["model"] = ModelConfig(**data["model"])
            
        super().__init__(**data)
