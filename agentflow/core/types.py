"""Core types module."""

from enum import Enum

class AgentStatus(Enum):
    """Agent status enum."""
    
    INITIALIZED = "initialized"
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    TERMINATED = "terminated" 