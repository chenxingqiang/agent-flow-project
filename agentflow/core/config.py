"""Configuration module for AgentFlow."""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

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

class AgentMode(Enum):
    """Agent mode enumeration."""
    SIMPLE = "simple"
    COMPLEX = "complex"
    HYBRID = "hybrid"

class ModelConfig(BaseModel):
    """Model configuration."""
    
    provider: str = Field(default="default")
    name: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "name": self.name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }

class WorkflowConfig(BaseModel):
    """Workflow configuration."""
    id: str
    name: str
    description: Optional[str] = None
    agents: List[Any] = Field(default_factory=list)
    processors: List[Any] = Field(default_factory=list)
    connections: List[Any] = Field(default_factory=list)
    max_steps: int = 100
    timeout: float = 300.0
    parallel: bool = False
    error_handling: str = "retry"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agents": [agent.to_dict() if hasattr(agent, 'to_dict') else agent for agent in self.agents],
            "processors": self.processors,
            "connections": self.connections,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
            "parallel": self.parallel,
            "error_handling": self.error_handling,
            "type": ConfigurationType.WORKFLOW.value
        }

@dataclass
class AgentConfig:
    """Agent configuration."""
    id: str
    name: str
    description: Optional[str] = None
    type: Union[ConfigurationType, str] = ConfigurationType.GENERIC
    version: str = "1.0.0"
    system_prompt: str = ""
    model: Union[ModelConfig, Dict[str, Any]] = None
    workflow: Optional[Union[WorkflowConfig, Dict[str, Any]]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post initialization."""
        # Convert type to enum if needed
        if isinstance(self.type, str):
            try:
                self.type = ConfigurationType(self.type)
            except ValueError:
                self.type = ConfigurationType.GENERIC
        
        # Convert model dict to ModelConfig if needed
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        elif self.model is None:
            self.model = ModelConfig()
            
        # Convert workflow dict to WorkflowConfig if needed
        if isinstance(self.workflow, dict):
            self.workflow = WorkflowConfig(**self.workflow)
            
        # Initialize config if needed
        if self.config is None:
            self.config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": ConfigurationType.AGENT.value,
            "agent_type": self.type.value,
            "version": self.version,
            "system_prompt": self.system_prompt,
            "model": self.model.to_dict() if self.model else None,
            "workflow": self.workflow.to_dict() if self.workflow else None,
            "config": self.config
        }
