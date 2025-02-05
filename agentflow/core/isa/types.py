"""ISA types module."""
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
from dataclasses import dataclass, field
from datetime import datetime

class AgentType(str, Enum):
    """Agent type."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    REVIEW = "review"
    CUSTOM = "custom"

class InstructionType(str, Enum):
    """Instruction type enumeration."""
    LLM = "llm"
    TOOL = "tool"
    RESOURCE = "resource"
    COMPUTATION = "computation"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    CUSTOM = "custom"

class InstructionStatus(str, Enum):
    """Instruction status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class InstructionPriority(str, Enum):
    """Instruction priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AnalysisType(str, Enum):
    """Analysis type."""
    BEHAVIOR = "behavior"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CUSTOM = "custom"

class PatternType(str, Enum):
    """Pattern type."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BEHAVIORAL = "behavioral"
    CUSTOM = "custom"

class AnalysisResult(BaseModel):
    """Analysis result."""
    type: AnalysisType
    metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    confidence: float

@dataclass
class InstructionMetadata:
    """Instruction metadata class."""
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    source: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    priority: InstructionPriority = InstructionPriority.MEDIUM
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InstructionContext:
    """Instruction context class."""
    
    workflow_id: Optional[str] = None
    step_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class InstructionConfig:
    """Instruction configuration class."""
    
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    additional_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InstructionMetrics:
    """Instruction metrics class."""
    
    tokens_total: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    cost: float = 0.0
    latency: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

@dataclass
class Instruction:
    """Instruction class."""
    
    id: str
    name: str
    type: InstructionType
    params: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    metadata: InstructionMetadata = field(default_factory=InstructionMetadata)
    context: InstructionContext = field(default_factory=InstructionContext)
    config: InstructionConfig = field(default_factory=InstructionConfig)
    metrics: InstructionMetrics = field(default_factory=InstructionMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instruction to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "params": self.params,
            "description": self.description,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat() if self.metadata.updated_at else None,
                "source": self.metadata.source,
                "tags": self.metadata.tags,
                "priority": self.metadata.priority.value,
                "timeout": self.metadata.timeout,
                "retry_count": self.metadata.retry_count,
                "max_retries": self.metadata.max_retries,
                "dependencies": self.metadata.dependencies,
                "additional_info": self.metadata.additional_info
            },
            "context": {
                "workflow_id": self.context.workflow_id,
                "step_id": self.context.step_id,
                "agent_id": self.context.agent_id,
                "session_id": self.context.session_id,
                "user_id": self.context.user_id,
                "environment": self.context.environment,
                "variables": self.context.variables,
                "state": self.context.state,
                "history": self.context.history
            },
            "config": {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "stop_sequences": self.config.stop_sequences,
                "additional_params": self.config.additional_params
            },
            "metrics": {
                "tokens_total": self.metrics.tokens_total,
                "tokens_prompt": self.metrics.tokens_prompt,
                "tokens_completion": self.metrics.tokens_completion,
                "cost": self.metrics.cost,
                "latency": self.metrics.latency,
                "start_time": self.metrics.start_time.isoformat() if self.metrics.start_time else None,
                "end_time": self.metrics.end_time.isoformat() if self.metrics.end_time else None,
                "duration": self.metrics.duration,
                "memory_usage": self.metrics.memory_usage,
                "cpu_usage": self.metrics.cpu_usage
            }
        }
