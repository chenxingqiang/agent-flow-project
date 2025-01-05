"""Core module for agentflow."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.agent_types import AgentConfig, AgentMode, AgentType, AgentStatus
    from ..agents.agent import Agent, AgentState

from .workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType
from .workflow_executor import WorkflowExecutor
from .workflow_state import WorkflowStateManager, WorkflowStatus, StepStatus
from .exceptions import WorkflowExecutionError, StepExecutionError
from .workflow import Workflow

# Re-export types for convenience
if TYPE_CHECKING:
    from ..agents.agent_types import AgentConfig, AgentMode, AgentType, AgentStatus

# Rebuild models after all imports are done
WorkflowConfig.model_rebuild()

__all__ = [
    'WorkflowConfig',
    'WorkflowStep',
    'WorkflowStepType',
    'AgentType',
    'AgentMode',
    'WorkflowExecutor',
    'WorkflowStateManager',
    'WorkflowStatus',
    'StepStatus',
    'WorkflowExecutionError',
    'StepExecutionError',
    'AgentConfig',
    'AgentMode',
    'AgentType',
    'AgentStatus',
    'Workflow'
]
