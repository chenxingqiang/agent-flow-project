"""Agent types module."""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import uuid

class AgentType(str, Enum):
    """Agent type enum."""
    GENERIC = "generic"
    RESEARCH = "research"
    DATA_SCIENCE = "data_science"
    CUSTOM = "custom"

class AgentMode(str, Enum):
    """Agent mode enum."""
    SYNC = "sync"
    ASYNC = "async"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"

class AgentStatus(str, Enum):
    """Agent status enum."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class ModelConfig(BaseModel):
    """Model configuration."""
    provider: str = Field(default="openai")
    name: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = None
    top_p: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)
    stop: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentConfig(BaseModel):
    """Agent configuration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="Default Agent")
    type: AgentType
    version: str = Field(default="1.0.0")
    mode: AgentMode = Field(default=AgentMode.SYNC)
    max_retries: int = Field(default=3)
    timeout: float = Field(default=60.0)
    workflow_path: Optional[str] = None
    workflow: Dict[str, Any] = Field(default_factory=dict)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    transformations: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(default_factory=list)
    domain_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    system_prompt: str = Field(default="You are a helpful assistant.")
    isa_config_path: Optional[str] = None
    max_errors: Optional[int] = Field(default=10)
    model: ModelConfig = Field(default_factory=ModelConfig)
    is_distributed: bool = Field(default=False)
    provider: str = Field(default="openai")
    workflow_policies: Dict[str, Any] = Field(default_factory=dict)
    error_handling: Dict[str, Any] = Field(default_factory=dict)
    retry_policy: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        # Lazily import to avoid circular imports
        super().__init__(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create an AgentConfig from a dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, AgentType) else self.type,
            "name": self.name,
            "version": self.version,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "mode": self.mode.value,
            "metadata": self.metadata,
            "domain_config": self.domain_config,
            "isa_config_path": self.isa_config_path,
            "workflow_path": self.workflow_path,
            "system_prompt": self.system_prompt,
            "max_errors": self.max_errors,
            "model": self.model.model_dump() if self.model else None,
            "workflow": self.workflow,
            "is_distributed": self.is_distributed,
            "provider": self.provider,
            "workflow_policies": self.workflow_policies,
            "error_handling": self.error_handling,
            "retry_policy": self.retry_policy
        }
