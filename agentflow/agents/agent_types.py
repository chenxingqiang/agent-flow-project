"""Agent types module."""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import uuid

# Re-export types from core.types
from ..core.types import AgentType, AgentMode, AgentStatus, ModelConfig, AgentConfig

__all__ = ['AgentType', 'AgentMode', 'AgentStatus', 'ModelConfig', 'AgentConfig']

class AgentType(str, Enum):
    """Agent type enum."""
    GENERIC = "generic"
    RESEARCH = "research"
    DATA_SCIENCE = "data_science"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"

class AgentMode(str, Enum):
    """Agent mode enum."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"

class AgentStatus(str, Enum):
    """Agent status enum."""
    IDLE = "idle"
    INITIALIZED = "initialized"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
