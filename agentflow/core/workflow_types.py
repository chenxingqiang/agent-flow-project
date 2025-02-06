"""Workflow types and execution logic."""

import asyncio
import networkx as nx
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union, ClassVar, Callable, Set, Coroutine, TypeVar, cast, Type
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationInfo, ValidationError, PrivateAttr
from pydantic.types import StrictBool
from pydantic.error_wrappers import ValidationError as PydanticValidationError
from pathlib import Path
from datetime import datetime
import os
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import time
import pandas as pd

from .exceptions import WorkflowExecutionError, ConfigurationError
from .retry_policy import RetryPolicy as BaseRetryPolicy
from .model_config import ModelConfig

T = TypeVar('T', bound='WorkflowStep')
C = TypeVar('C', bound='WorkflowConfig')

class Message(BaseModel):
    """Message class for workflow communication."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True
    )
    
    content: Any = Field(default=None, description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    
    def copy(self) -> 'Message':
        """Create a deep copy of the message."""
        return Message(
            content=self.content,
            metadata=self.metadata.copy(),
            timestamp=self.timestamp
        )

class WorkflowStepType(str, Enum):
    """Workflow step types."""
    TRANSFORM = "transform"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    ANALYZE = "analyze"
    RESEARCH_EXECUTION = "research_execution"
    DOCUMENT_GENERATION = "document_generation"
    AGENT = "agent"

    def __str__(self) -> str:
        """Return string representation of enum value."""
        return self.value

class WorkflowStepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"  # Waiting for dependencies

class WorkflowStatus(str, Enum):
    """Workflow status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INITIALIZED = "initialized"

class RetryPolicy(BaseModel):
    """Retry policy configuration."""
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0.0)
    backoff: float = Field(default=2.0, ge=1.0)
    max_delay: float = Field(default=60.0, ge=0.0)
    jitter: bool = True

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary."""
        return super().model_dump(**kwargs)

class ErrorPolicy(BaseModel):
    """Error policy configuration."""
    fail_fast: bool = True
    ignore_warnings: bool = False
    max_errors: int = Field(default=10, ge=0)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    continue_on_error: bool = False

    def __getitem__(self, key: str) -> Any:
        """Get item by key."""
        if key == "retry_policy":
            return self.retry_policy.model_dump()
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key."""
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().model_dump(**kwargs)
        if isinstance(data.get('retry_policy'), RetryPolicy):
            data['retry_policy'] = data['retry_policy'].model_dump()
        return data

class StepConfig(BaseModel):
    """Step configuration."""
    strategy: str = "standard"
    params: Dict[str, Any] = Field(default_factory=dict)
    retry_delay: float = Field(default=1.0, ge=0.0)
    retry_backoff: float = Field(default=2.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    execute: Optional[Callable[..., Coroutine[Any, Any, Dict[str, Any]]]] = None

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().model_dump(**kwargs)
        # Skip execute function when dumping
        if 'execute' in data:
            del data['execute']
        return data

class WorkflowStep(BaseModel):
    """Workflow step configuration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    type: WorkflowStepType = Field(default=WorkflowStepType.TRANSFORM)
    required: bool = True
    optional: bool = False
    is_distributed: bool = False
    dependencies: List[str] = Field(default_factory=list)
    config: StepConfig = Field(default_factory=StepConfig)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().model_dump(**kwargs)
        return data

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> 'WorkflowStep':
        """Validate and create instance."""
        if isinstance(obj, dict):
            if 'config' in obj and isinstance(obj['config'], dict):
                obj['config'] = StepConfig(**obj['config'])
            return super().model_validate(obj, **kwargs)
        return obj

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow step."""
        try:
            # Convert Message object to dictionary if needed
            if not isinstance(context, dict) and hasattr(context, 'model_dump'):
                context = context.model_dump()
            elif not isinstance(context, dict):
                context = {"data": context}

            # Execute step function if available
            if self.config.execute is not None and callable(self.config.execute):
                try:
                    result = await self.config.execute(context.get("data"))
                    return {
                        "data": result,
                        "result": result,
                        "metadata": {}
                    }
                except Exception as e:
                    raise WorkflowExecutionError(f"Step {self.id} failed: {str(e)}") from e

            # Handle different step types
            if self.type == WorkflowStepType.TRANSFORM:
                result = await self._execute_transform(context)
                return {
                    "data": result.get("data"),
                    "result": result.get("result"),
                    "metadata": {}
                }
            elif self.type == WorkflowStepType.FILTER:
                result = await self._execute_filter(context)
                return {
                    "data": result.get("data"),
                    "result": result.get("data"),
                    "metadata": {}
                }
            elif self.type == WorkflowStepType.AGGREGATE:
                result = await self._execute_aggregate(context)
                return {
                    "data": result.get("data"),
                    "result": result.get("data"),
                    "metadata": {}
                }
            elif self.type == WorkflowStepType.ANALYZE:
                result = await self._execute_analyze(context)
                return {
                    "data": result.get("data"),
                    "result": result.get("data"),
                    "metadata": {}
                }
            elif self.type == WorkflowStepType.RESEARCH_EXECUTION:
                result = await self._execute_research(context)
                return {
                    "data": result.get("data"),
                    "result": result.get("data"),
                    "metadata": {}
                }
            elif self.type == WorkflowStepType.DOCUMENT_GENERATION:
                result = await self._execute_document_generation(context)
                return {
                    "data": result.get("data"),
                    "result": result.get("data"),
                    "metadata": {}
                }
            elif self.type == WorkflowStepType.AGENT:
                # For AGENT type, just pass through the input data
                return {
                    "data": context.get("data", None),
                    "content": context.get("message", ""),
                    "metadata": {}
                }
            else:
                raise ValueError(f"Unsupported step type: {self.type}")
        except Exception as e:
            if isinstance(e, WorkflowExecutionError):
                raise
            raise WorkflowExecutionError(f"Step {self.id} failed: {str(e)}") from e

    async def _execute_transform(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute transform step."""
        if self.config.strategy == "feature_engineering":
            data = context.get("data")
            if not isinstance(data, (list, np.ndarray)):
                raise WorkflowExecutionError(f"Transform step {self.id} requires numeric data")
            # Convert data to numpy array
            data = np.asarray(data)
            params = self.config.params
            method = params.get("method", "standard")
            if method == "standard":
                with_mean = params.get("with_mean", True)
                with_std = params.get("with_std", True)
                mean = np.mean(data, axis=0) if with_mean else 0
                std = np.std(data, axis=0) if with_std else 1
                transformed_data = (data - mean) / std
                return {
                    "data": transformed_data,
                    "result": {
                        "data": {
                            "data": transformed_data
                        }
                    }
                }
            elif method == "isolation_forest":
                threshold = params.get("threshold", 0.1)
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                z_scores = np.abs((data - mean) / std)
                outliers = np.any(z_scores > threshold, axis=1)
                transformed_data = data[~outliers]
                return {
                    "data": transformed_data,
                    "result": {
                        "data": {
                            "data": transformed_data
                        }
                    }
                }
        return {
            "data": context.get("data"),
            "result": {
                "data": {
                    "data": context.get("data")
                }
            }
        }

    async def _execute_filter(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute filter step."""
        data = context.get("data")
        if not isinstance(data, (list, dict)):
            raise WorkflowExecutionError(f"Filter step {self.id} requires list or dict data")
        params = self.config.params
        if isinstance(data, list):
            condition = params.get("condition", lambda x: True)
            filtered_data = [x for x in data if condition(x)]
        else:
            condition = params.get("condition", lambda k, v: True)
            filtered_data = {k: v for k, v in data.items() if condition(k, v)}
        return {"data": filtered_data}

    async def _execute_aggregate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute aggregate step."""
        data = context.get("data")
        if not isinstance(data, (list, dict)):
            raise WorkflowExecutionError(f"Aggregate step {self.id} requires list or dict data")
        params = self.config.params
        method = params.get("method", "sum")
        if method == "sum":
            result = sum(data) if isinstance(data, list) else sum(data.values())
        elif method == "mean":
            result = np.mean(data) if isinstance(data, list) else np.mean(list(data.values()))
        else:
            result = data
        return {"data": result}

    async def _execute_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analyze step."""
        data = context.get("data")
        if not isinstance(data, (list, dict, np.ndarray)):
            raise WorkflowExecutionError(f"Analyze step {self.id} requires list, dict, or array data")
        params = self.config.params
        method = params.get("method", "basic_stats")
        if method == "basic_stats":
            if isinstance(data, (list, np.ndarray)):
                result = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data))
                }
            else:
                result = {
                    "count": len(data),
                    "keys": list(data.keys())
                }
        else:
            result = {"data": data}
        return {"data": result}

    async def _execute_research(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research step."""
        raise WorkflowExecutionError(f"Research execution step {self.id} requires a research agent to be configured")

    async def _execute_document_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document generation step."""
        raise WorkflowExecutionError(f"Document generation step {self.id} requires a document generation agent to be configured")

class WorkflowConfig(BaseModel):
    """Workflow configuration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="default_workflow")
    type: str = "sequential"
    steps: List[WorkflowStep] = Field(default_factory=list)
    max_iterations: int = Field(default=10, ge=1)
    timeout: float = Field(default=3600.0, ge=0.0)
    error_policy: ErrorPolicy = Field(default_factory=ErrorPolicy)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ell2a_config: Optional[Dict[str, Any]] = None
    agent: Optional[Any] = None  # Store agent reference
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True
    )
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().model_dump(**kwargs)
        # Skip agent when dumping
        if 'agent' in data:
            del data['agent']
        return data

    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v: List[WorkflowStep], info: ValidationInfo) -> List[WorkflowStep]:
        """Validate steps list."""
        # Allow empty steps list during initialization or if explicitly allowed
        if info.context and (
            info.context.get("allow_empty_steps") or
            info.context.get("is_initialization", False) or
            info.context.get("distributed", False)
        ):
            return v
        # Allow empty steps for distributed workflows
        if info.context and info.context.get("distributed", False):
            return v
        if not v:
            # Add default step for empty workflows
            return [
                WorkflowStep(
                    id="default-step",
                    name="default_step",
                    type=WorkflowStepType.AGENT,
                    config=StepConfig(strategy="default")
                )
            ]
        return v

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by its ID.
        
        Args:
            step_id: The ID of the step to get.
            
        Returns:
            The step with the given ID, or None if not found.
        """
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> 'WorkflowConfig':
        """Validate and create instance."""
        if isinstance(obj, dict):
            # Convert steps to proper format if needed
            if 'steps' in obj and isinstance(obj['steps'], list):
                obj['steps'] = [
                    WorkflowStep.model_validate(step) if not isinstance(step, WorkflowStep) else step
                    for step in obj['steps']
                ]
            # Convert error policy to proper format if needed
            if 'error_policy' in obj and not isinstance(obj['error_policy'], ErrorPolicy):
                if isinstance(obj['error_policy'], dict):
                    obj['error_policy'] = ErrorPolicy(**obj['error_policy'])
                elif hasattr(obj['error_policy'], 'model_dump'):
                    obj['error_policy'] = ErrorPolicy(**obj['error_policy'].model_dump())
                elif hasattr(obj['error_policy'], 'dict'):
                    obj['error_policy'] = ErrorPolicy(**obj['error_policy'].dict())
        return super().model_validate(obj, **kwargs)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps.
        
        Args:
            context: Execution context
            
        Returns:
            Dict containing execution results
            
        Raises:
            WorkflowExecutionError: If execution fails
        """
        if not self.steps:
            raise ValueError("Workflow has no steps")
            
        results = {
            "steps": {},
            "status": "completed",
            "error": None
        }
        completed_steps = set()
        
        for step in self.steps:
            try:
                # Validate dependencies
                if step.dependencies:
                    for dep_id in step.dependencies:
                        if dep_id not in completed_steps:
                            raise WorkflowExecutionError(f"Missing dependency '{dep_id}' required by step '{step.id}'")
                
                step_result = await step.execute(context)
                results["steps"][step.id] = {
                    "data": step_result.get("data"),
                    "result": step_result.get("result"),
                    "metadata": step_result.get("metadata", {})
                }
                context.update(step_result)
                completed_steps.add(step.id)
            except Exception as e:
                if self.error_policy.fail_fast:
                    raise WorkflowExecutionError(f"Error executing step {step.id}: {str(e)}") from e
                if not self.error_policy.ignore_warnings:
                    results["warnings"] = results.get("warnings", []) + [str(e)]
                    
        return results

class WorkflowMetrics(BaseModel):
    """Metrics for workflow execution."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra='allow'
    )

    # Public fields
    total_steps: int = Field(default=0)
    completed_steps: int = Field(default=0)
    failed_steps: int = Field(default=0)
    retried_steps: Set[str] = Field(default_factory=set)
    task_durations: List[float] = Field(default_factory=list)
    step_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        'duration': {},
        'retries': {},
        'status': {},
        'errors': {},
        'dependencies': {}
    })

    def initialize_step(self, step_id: str) -> None:
        """Initialize metrics for a new step."""
        self.step_metrics['duration'][step_id] = 0.0
        self.step_metrics['retries'][step_id] = 0
        self.step_metrics['status'][step_id] = 'pending'
        self.step_metrics['errors'][step_id] = []

    def update_step_duration(self, step_id: str, duration: float) -> None:
        """Update duration for a step."""
        self.step_metrics['duration'][step_id] = duration
        self.task_durations.append(duration)

    def increment_step_retries(self, step_id: str) -> None:
        """Increment retry count for a step."""
        if step_id not in self.retried_steps:
            self.retried_steps.add(step_id)
        self.step_metrics['retries'][step_id] += 1

    def update_step_status(self, step_id: str, status: str) -> None:
        """Update status for a step."""
        self.step_metrics['status'][step_id] = status
        if status == 'completed':
            self.completed_steps += 1
        elif status == 'failed':
            self.failed_steps += 1

    def add_step_error(self, step_id: str, error: str) -> None:
        """Add error for a step."""
        self.step_metrics['errors'][step_id].append(error)

    def get_step_metrics(self, step_id: str) -> Dict[str, Any]:
        """Get metrics for a specific step."""
        return {
            'duration': self.step_metrics['duration'].get(step_id, 0.0),
            'retries': self.step_metrics['retries'].get(step_id, 0),
            'status': self.step_metrics['status'].get(step_id, 'unknown'),
            'errors': self.step_metrics['errors'].get(step_id, [])
        }

class Workflow:
    """Workflow execution engine."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize workflow with configuration."""
        self.config = config
        self.error_policy = ErrorPolicy()
        if isinstance(config.error_policy, dict):
            self.error_policy = ErrorPolicy(**config.error_policy)
        else:
            self.error_policy = config.error_policy
        
    async def execute(self, data: Dict[str, Any] = Field(default_factory=dict)) -> Dict[str, Any]:
        """Execute the workflow.
        
        Args:
            data: Input data for workflow execution
            
        Returns:
            Dict containing execution results
            
        Raises:
            WorkflowExecutionError: If execution fails
        """
        if data is None:
            data = {}
            
        results = {}
        context = data.copy()
        completed_steps = set()
        
        for step in self.config.steps:
            try:
                # Validate dependencies
                if step.dependencies:
                    for dep_id in step.dependencies:
                        if dep_id not in completed_steps:
                            raise WorkflowExecutionError(f"Missing dependency '{dep_id}' required by step '{step.id}'")
                
                step_result = await step.execute(context)
                results[step.id] = {
                    "output": {
                        "data": step_result.get("data"),
                        "result": step_result.get("result"),
                        "metadata": step_result.get("metadata", {})
                    }
                }
                context.update(step_result)
                completed_steps.add(step.id)
            except Exception as e:
                if self.error_policy.fail_fast:
                    raise WorkflowExecutionError(f"Error executing step {step.id}: {str(e)}") from e
                if not self.error_policy.ignore_warnings:
                    results["warnings"] = results.get("warnings", []) + [str(e)]
                    
        return results

class AgentConfig(BaseModel):
    """Configuration for an agent."""

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="default_agent")
    type: str = "generic"
    version: str = "1.0.0"
    max_retries: int = Field(default=3, ge=0)
    mode: str = "sequential"
    distributed: bool = False
    system_prompt: Optional[str] = None
    model: Union[Dict[str, Any], ModelConfig] = Field(default_factory=lambda: ModelConfig(name="gpt-3.5-turbo", provider="openai"))
    workflow: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    research_context: Dict[str, Any] = Field(default_factory=dict)
    data_science_context: Dict[str, Any] = Field(default_factory=dict)
    custom_attribute: str = Field(default="custom")
    domain_config: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    @classmethod
    def validate_model_provider(cls, data: Any) -> Any:
        """Validate model provider after model is instantiated."""
        valid_providers = {'openai', 'anthropic', 'mistral', 'cohere', 'ai21', 'default'}
        
        if isinstance(data.model, dict):
            try:
                data.model = ModelConfig(**data.model)
            except Exception as e:
                raise ValueError(f"Invalid model configuration: {e}")
        
        if isinstance(data.model, ModelConfig):
            if data.model.provider not in valid_providers:
                raise ValueError(f"Invalid model provider: {data.model.provider}. Must be one of {valid_providers}")
        
        return data

    @field_validator('workflow')
    @classmethod
    def validate_workflow(cls, v: Optional[Union[Dict[str, Any], WorkflowConfig]]) -> Dict[str, Any]:
        """Validate the workflow configuration."""
        if v is None:
            return {}
        if isinstance(v, WorkflowConfig):
            return v.model_dump()
        if isinstance(v, dict):
            try:
                if "name" not in v and "id" not in v:
                    v["name"] = "default"
                    v["id"] = str(uuid.uuid4())
                workflow_config = WorkflowConfig(**v)
                return workflow_config.model_dump()
            except ValidationError as e:
                raise ValueError(f"Invalid workflow configuration: {e}")
        raise ValueError("Workflow must be either a dictionary or WorkflowConfig instance")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate the agent type."""
        valid_types = {"generic", "research", "data_science", "custom"}
        if v not in valid_types:
            raise ValueError(f"Invalid type: {v}. Must be one of {valid_types}")
        return v

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate the agent mode."""
        valid_modes = {"sequential", "parallel", "simple"}
        if v not in valid_modes:
            raise ValueError(f"Invalid mode: {v}. Must be one of {valid_modes}")
        return v

    @property
    def is_distributed(self) -> bool:
        """Check if agent is distributed."""
        if self.distributed:
            return True
        if isinstance(self.workflow, dict) and self.workflow.get("distributed", False):
            return True
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the configuration."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key: str) -> Any:
        """Get item by key."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        try:
            getattr(self, key)
            return True
        except AttributeError:
            return False

class WorkflowContext(BaseModel):
    """Context for workflow execution."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="default_context")
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    step_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    metrics: WorkflowMetrics = Field(default_factory=WorkflowMetrics)

    def update(self, data: Dict[str, Any]) -> None:
        """Update context with new data."""
        self.data.update(data)

    def get_step_result(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get result for a specific step."""
        return self.step_results.get(step_id)

    def set_step_result(self, step_id: str, result: Dict[str, Any]) -> None:
        """Set result for a specific step."""
        self.step_results[step_id] = result

    def start(self) -> None:
        """Start workflow execution."""
        self.start_time = datetime.now()
        self.status = WorkflowStatus.RUNNING

    def complete(self) -> None:
        """Complete workflow execution."""
        self.end_time = datetime.now()
        self.status = WorkflowStatus.COMPLETED

    def fail(self, error: str) -> None:
        """Mark workflow as failed."""
        self.end_time = datetime.now()
        self.status = WorkflowStatus.FAILED
        self.error = error

    def get_duration(self) -> Optional[float]:
        """Get workflow execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        data = super().model_dump(**kwargs)
        if self.metrics:
            data['metrics'] = self.metrics.model_dump(**kwargs)
        else:
            data['metrics'] = {}
        return data

    def __getitem__(self, key: str) -> Any:
        """Get item by key."""
        if key in self.data:
            return self.data[key]
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item by key."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.data or hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with a default."""
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

def has_circular_dependency(dependencies: Dict[str, List[str]]) -> bool:
    """Check if there are circular dependencies in the graph.
    
    Args:
        dependencies: Dictionary mapping step IDs to their dependencies
        
    Returns:
        bool: True if circular dependencies exist, False otherwise
    """
    visited = set()
    path = set()
    
    def visit(node: str) -> bool:
        if node in path:
            return True
        if node in visited:
            return False
            
        visited.add(node)
        path.add(node)
        
        for neighbor in dependencies.get(node, []):
            if visit(neighbor):
                return True
                
        path.remove(node)
        return False
    
    return any(visit(node) for node in dependencies)