"""Agent configuration module."""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
import uuid
from .base_types import AgentType, AgentMode, AgentStatus
from .workflow_types import WorkflowConfig
from .model_config import ModelConfig

class AgentConfig(BaseModel):
    """Agent configuration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(
        description="Name of the agent", 
        min_length=3, 
        pattern=r'^[a-zA-Z0-9_\- ]+$',
        json_schema_extra={
            "error_messages": {
                "value_error.min_length": "Name must be at least 3 characters long",
                "value_error.pattern": "Name can only contain letters, numbers, spaces, underscores, and hyphens"
            }
        }
    )
    description: str = Field(default="")
    type: str = Field(
        description="Type of the agent", 
        pattern=r'^(generic|custom|research|data_science|workflow|assistant|interactive)$',
        json_schema_extra={
            "error_messages": {
                "value_error.pattern": "Invalid agent type. Must be one of: generic, custom, research, data_science, workflow, assistant, interactive"
            }
        }
    )
    version: str = Field(default="1.0.0")
    mode: AgentMode = Field(default=AgentMode.SYNC)
    max_retries: int = Field(default=3)
    timeout: float = Field(default=300.0, ge=0, le=3600, description="Timeout for agent operations")
    max_iterations: int = Field(default=10, ge=1, le=100, description="Maximum number of iterations for agent")
    workflow_path: Optional[str] = None
    workflow: Optional[WorkflowConfig] = Field(default=None)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    transformations: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(default_factory=list)
    domain_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    system_prompt: str = Field(default="You are a helpful assistant.")
    isa_config_path: Optional[str] = None
    max_errors: Optional[int] = Field(default=10)
    model_provider: str = Field(
        default="openai", 
        description="Provider for the model", 
        pattern=r'^(openai|anthropic|google|azure|local)$',
        json_schema_extra={
            "error_messages": {
                "value_error.pattern": "Invalid provider. Must be one of: openai, anthropic, google, azure, local"
            }
        }
    )
    model: ModelConfig = Field(default_factory=ModelConfig)
    is_distributed: bool = Field(default=False)
    provider: str = Field(default="openai")
    workflow_policies: Dict[str, Any] = Field(default_factory=dict)
    error_handling: Dict[str, Any] = Field(default_factory=dict)
    retry_policy: Dict[str, Any] = Field(default_factory=dict)
    status: AgentStatus = Field(default=AgentStatus.INITIALIZED)
    error_policy: str = Field(default="continue", pattern=r'^(continue|stop|retry)$')

    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',
        protected_namespaces=()  # Resolve the namespace warning
    )

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

    @field_validator('model', mode='before')
    @classmethod
    def validate_model(cls, v: Union[Dict[str, Any], ModelConfig, None]) -> ModelConfig:
        """Validate the model configuration."""
        if v is None:
            return ModelConfig()
        if isinstance(v, ModelConfig):
            return v
        if isinstance(v, dict):
            try:
                return ModelConfig.model_validate(v)
            except ValidationError as e:
                raise ValueError(f"Invalid model configuration: {e}")
        raise ValueError("Model must be either a dictionary or ModelConfig instance")

    def __init__(self, **data):
        """Initialize agent configuration."""
        # Validate name
        if not data.get('name'):
            raise ValueError("Name cannot be empty")
        
        # Handle workflow configuration
        if 'workflow' in data and isinstance(data['workflow'], dict):
            try:
                data['workflow'] = WorkflowConfig.model_validate(
                    data['workflow'],
                    context={"allow_empty_steps": True}
                )
            except ValidationError as e:
                raise ValueError(f"Invalid workflow configuration: {e}")

        # Handle model configuration
        model_data = data.get('model', {})
        if model_data is None:
            data['model'] = ModelConfig()
        elif isinstance(model_data, dict):
            try:
                data['model'] = ModelConfig.model_validate(model_data)
            except ValidationError as e:
                raise ValueError(f"Invalid model configuration: {e}")
        elif not isinstance(model_data, ModelConfig):
            data['model'] = ModelConfig(
                provider=data.get('model_provider', 'openai'),
                name=data.get('model_name', 'gpt-4'),
                temperature=data.get('temperature', 0.7),
                max_tokens=data.get('max_tokens', 4096)
            )
        
        super().__init__(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "version": self.version,
            "max_retries": self.max_retries,
            "mode": self.mode.value if isinstance(self.mode, AgentMode) else self.mode,
            "metadata": self.metadata,
            "domain_config": self.domain_config,
            "isa_config_path": self.isa_config_path,
            "workflow_path": self.workflow_path,
            "system_prompt": self.system_prompt,
            "max_errors": self.max_errors,
            "model": self.model.model_dump(),
            "workflow": self.workflow.model_dump() if self.workflow else None,
            "is_distributed": self.is_distributed,
            "provider": self.provider,
            "workflow_policies": self.workflow_policies,
            "error_handling": self.error_handling,
            "retry_policy": self.retry_policy,
            "status": self.status.value if isinstance(self.status, AgentStatus) else self.status
        }
        if self.timeout is not None:
            data["timeout"] = self.timeout
        if self.max_iterations is not None:
            data["max_iterations"] = self.max_iterations
        return data

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().model_dump(**kwargs)
        if isinstance(data.get('model'), ModelConfig):
            data['model'] = data['model'].model_dump()
        if isinstance(data.get('workflow'), WorkflowConfig):
            data['workflow'] = data['workflow'].model_dump()
        return data

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> 'AgentConfig':
        """Validate and create instance."""
        if isinstance(obj, dict):
            # Handle model field
            if 'model' in obj and not isinstance(obj['model'], ModelConfig):
                try:
                    if isinstance(obj['model'], dict):
                        obj['model'] = ModelConfig.model_validate(obj['model'])
                except ValidationError as e:
                    raise ValueError(f"Invalid model configuration: {e}")
            # Handle workflow field
            if 'workflow' in obj and not isinstance(obj['workflow'], WorkflowConfig):
                try:
                    obj['workflow'] = WorkflowConfig.model_validate(obj['workflow'])
                except ValidationError as e:
                    raise ValueError(f"Invalid workflow configuration: {e}")
        return super().model_validate(obj, **kwargs) 

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create configuration from a dictionary."""
        return cls(**config_dict)