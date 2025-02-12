from enum import Enum

class StepStatus(str, Enum):
    """Step status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    
class NodeType(str, Enum):
    """Node type enum."""
    AGENT = "agent"
    PROCESSOR = "processor"
    CONNECTOR = "connector"
    
class NodeState(str, Enum):
    """Node state enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    
class WorkflowStatus(str, Enum):
    """Workflow status enum."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"
    
class ConfigurationType(str, Enum):
    """Configuration type enum."""
    GENERIC = "generic"
    RESEARCH = "research"
    ASSISTANT = "assistant"
    EXPERT = "expert"
    CUSTOM = "custom"
    
class AgentMode(str, Enum):
    """Agent mode enum."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    INTERACTIVE = "interactive" 