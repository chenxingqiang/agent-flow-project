"""Agent configuration module."""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, ValidationError
import uuid
from .base_types import AgentType, AgentMode, AgentStatus
from .workflow_types import WorkflowConfig

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

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate model provider."""
        valid_providers = {'openai', 'anthropic', 'mistral', 'cohere', 'ai21', 'default'}
        if v not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of {valid_providers}")
        return v

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().model_dump(**kwargs)
        return data

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> 'ModelConfig':
        """Validate and create instance."""
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

class AgentConfig(BaseModel):
    """Agent configuration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="Default Agent")
    type: AgentType = Field(default=AgentType.GENERIC)
    version: str = Field(default="1.0.0")
    mode: AgentMode = Field(default=AgentMode.SYNC)
    max_retries: int = Field(default=3)
    timeout: float = Field(default=60.0)
    workflow_path: Optional[str] = None
    workflow: Union[Dict[str, Any], WorkflowConfig] = Field(default_factory=dict)
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
    status: AgentStatus = Field(default=AgentStatus.IDLE)

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate model provider."""
        valid_providers = {'openai', 'anthropic', 'mistral', 'cohere', 'ai21', 'default'}
        if v not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of {valid_providers}")
        return v

    @field_validator('workflow')
    @classmethod
    def validate_workflow(cls, v: Optional[Union[Dict[str, Any], WorkflowConfig]]) -> Union[Dict[str, Any], WorkflowConfig]:
        """Validate the workflow configuration."""
        if v is None:
            return {}
        if isinstance(v, WorkflowConfig):
            return v
        if isinstance(v, dict):
            try:
                if "name" not in v and "id" not in v:
                    v["name"] = "default"
                    v["id"] = str(uuid.uuid4())
                # Pass context to allow empty steps during initialization
                return WorkflowConfig.model_validate(v, context={"allow_empty_steps": True})
            except ValidationError as e:
                raise ValueError(f"Invalid workflow configuration: {e}")
        raise ValueError("Workflow must be either a dictionary or WorkflowConfig instance")

    def __init__(self, **data):
        if 'workflow' in data and isinstance(data['workflow'], dict):
            try:
                # Use model_validate with context to allow empty steps
                data['workflow'] = WorkflowConfig.model_validate(
                    data['workflow'],
                    context={"allow_empty_steps": True}
                )
            except ValidationError as e:
                raise ValueError(f"Invalid workflow configuration: {e}")
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
            "workflow": self.workflow.model_dump() if isinstance(self.workflow, WorkflowConfig) else self.workflow,
            "is_distributed": self.is_distributed,
            "provider": self.provider,
            "workflow_policies": self.workflow_policies,
            "error_handling": self.error_handling,
            "retry_policy": self.retry_policy,
            "status": self.status.value
        } 