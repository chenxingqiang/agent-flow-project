"""Core module for agentflow."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.agent_types import AgentConfig, AgentMode, AgentType, AgentStatus
    from ..agents.agent import Agent, AgentState

from .workflow_types import WorkflowConfig, StepConfig, WorkflowStepType
from .workflow_executor import WorkflowExecutor
from .workflow_state import WorkflowStateManager, WorkflowStatus, StepStatus
from .exceptions import WorkflowExecutionError, StepExecutionError
from .workflow import Workflow

# Re-export types for convenience
if TYPE_CHECKING:
    from ..agents.agent_types import AgentConfig, AgentMode, AgentType, AgentStatus

__all__ = [
    'WorkflowConfig',
    'StepConfig',
    'WorkflowStepType',
    'WorkflowStateManager',
    'WorkflowExecutionError'
]
