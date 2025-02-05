"""Base types module."""

from enum import Enum

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
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
    INITIALIZED = "INITIALIZED"

class DictKeyType(str, Enum):
    """Dictionary key type enum."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    UUID = "uuid"
    ANY = "any"

class MessageType(str, Enum):
    """Message type enumeration."""
    COMMAND = "command"
    RESPONSE = "response"
    ERROR = "error"
    STATUS = "status"
    DATA = "data"
    RESULT = "result"
    LOG = "log"
    METRIC = "metric"
    EVENT = "event"
    CUSTOM = "custom" 