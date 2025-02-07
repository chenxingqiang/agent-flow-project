"""State management module for AgentFlow."""

from typing import Dict, Any, List, Optional
from .base import BaseWorkflow
from enum import Enum
from pydantic import BaseModel, Field

class StateManager:
    """Manages workflow state."""
    
    def __init__(self):
        """Initialize state manager."""
        self._workflows: Dict[str, BaseWorkflow] = {}
        self._current_workflow: Optional[BaseWorkflow] = None
        
    def register_workflow(self, workflow: BaseWorkflow) -> None:
        """Register a workflow.
        
        Args:
            workflow: Workflow to register
        """
        self._workflows[workflow.id] = workflow
        self._current_workflow = workflow
        
    def get_workflow(self) -> Optional[BaseWorkflow]:
        """Get current workflow.
        
        Returns:
            Optional[BaseWorkflow]: Current workflow
        """
        return self._current_workflow
        
    def get_workflows(self) -> List[BaseWorkflow]:
        """Get all workflows.
        
        Returns:
            List[BaseWorkflow]: List of workflows
        """
        return list(self._workflows.values())

class AgentStatus(str, Enum):
    """Agent status."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class AgentState(BaseModel):
    """Agent state."""

    status: AgentStatus = AgentStatus.IDLE
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict) 