"""Workflow state management module."""

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator, model_validator
from dataclasses import dataclass, field
from .workflow_types import WorkflowStatus, StepStatus
from ..agents.agent_types import AgentStatus

class StepState:
    """Step state class."""
    
    def __init__(self):
        """Initialize step state."""
        self.status = StepStatus.PENDING
        self.result = None
        self.start_time = None
        self.end_time = None
        self.success_count = 0
        self.failure_count = 0
        self.retry_count = 0
        self.metadata = {}
        
    def start(self) -> None:
        """Start step execution."""
        self.start_time = datetime.now().timestamp()
        self.status = StepStatus.RUNNING
        
    def complete(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Complete step execution."""
        self.end_time = datetime.now().timestamp()
        self.status = StepStatus.COMPLETED
        self.result = result
        self.success_count += 1
        
    def fail(self, error: str) -> None:
        """Fail step execution."""
        self.end_time = datetime.now().timestamp()
        self.status = StepStatus.FAILED
        self.metadata['error'] = error
        self.failure_count += 1
        
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1
        
    def get_execution_time(self) -> Optional[float]:
        """Get execution time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class AgentState:
    """Agent state class."""
    
    def __init__(self):
        """Initialize agent state."""
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status: AgentStatus = AgentStatus.INITIALIZED
        self.error: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self.step_results: Dict[str, Any] = {}
        self.retry_count = 0
        self.max_retries = 3
        self.timeout = 60.0
        self.last_error: Optional[str] = None
        
    def start(self) -> None:
        """Start agent execution."""
        self.start_time = datetime.now()
        self.status = AgentStatus.RUNNING
        
    def complete(self) -> None:
        """Complete agent execution."""
        self.end_time = datetime.now()
        self.status = AgentStatus.COMPLETED
        
    def fail(self, error: str) -> None:
        """Fail agent execution."""
        self.end_time = datetime.now()
        self.status = AgentStatus.FAILED
        self.error = error
        self.last_error = error
        
    def stop(self) -> None:
        """Stop agent execution."""
        self.end_time = datetime.now()
        self.status = AgentStatus.STOPPED
        
    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to the state."""
        self.metrics[name] = value
        
    def add_step_result(self, step_id: str, result: Any) -> None:
        """Add a step result to the state."""
        self.step_results[step_id] = result
        
    def get_execution_time(self) -> Optional[float]:
        """Get execution time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def should_retry(self) -> bool:
        """Check if agent should retry execution."""
        return self.retry_count < self.max_retries and self.status == AgentStatus.FAILED

    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            self.status = AgentStatus.FAILED

class WorkflowState:
    """Workflow state class."""

    def __init__(self, step_states: Optional[Dict[str, StepState]] = None):
        """Initialize workflow state."""
        self.status = WorkflowStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.error = None
        self.metadata = {}
        self.step_states = step_states or {}
        
    def start(self) -> None:
        """Start workflow execution."""
        self.start_time = datetime.now().timestamp()
        self.status = WorkflowStatus.RUNNING
        
    def fail(self, error: str) -> None:
        """Fail workflow execution."""
        self.end_time = datetime.now().timestamp()
        self.status = WorkflowStatus.FAILED
        self.error = error

    def complete(self) -> None:
        """Complete workflow execution."""
        self.end_time = datetime.now().timestamp()
        self.status = WorkflowStatus.COMPLETED

    def get_step_state(self, step_id: str) -> Optional[StepState]:
        """Get step state."""
        return self.step_states.get(step_id)

    def add_step_state(self, step_id: str, state: StepState) -> None:
        """Add step state."""
        self.step_states[step_id] = state

    def get_execution_time(self) -> Optional[float]:
        """Get execution time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class WorkflowStateManager:
    """Workflow state manager class."""

    def __init__(self):
        """Initialize workflow state manager."""
        self.step_states = {}  # Map of step_id to StepState
        self.workflow_states = {}  # Map of workflow_id to WorkflowState
        self.workflow_result = None

    def initialize_workflow(self, workflow_id: str) -> None:
        """Initialize workflow state."""
        self.workflow_states[workflow_id] = WorkflowState()

    def update_workflow_status(self, workflow_id: str, status: WorkflowStatus) -> None:
        """Update workflow status."""
        if workflow_id not in self.workflow_states:
            self.initialize_workflow(workflow_id)
        
        workflow_state = self.workflow_states[workflow_id]
        workflow_state.status = status
        
        if status == WorkflowStatus.RUNNING:
            workflow_state.start_time = datetime.now().timestamp()
        elif status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
            workflow_state.end_time = datetime.now().timestamp()

    def get_workflow_status(self, workflow_id: str) -> WorkflowStatus:
        """Get workflow status."""
        if workflow_id not in self.workflow_states:
            return WorkflowStatus.PENDING
        return self.workflow_states[workflow_id].status

    def set_workflow_error(self, workflow_id: str, error: str) -> None:
        """Set workflow error."""
        if workflow_id not in self.workflow_states:
            self.initialize_workflow(workflow_id)
        self.workflow_states[workflow_id].error = error

    def update_step_status(self, workflow_id: str, step_id: str, status: StepStatus) -> None:
        """Update step status."""
        if workflow_id not in self.workflow_states:
            self.initialize_workflow(workflow_id)
        
        workflow_state = self.workflow_states[workflow_id]
        if step_id not in workflow_state.step_states:
            workflow_state.step_states[step_id] = StepState()
        
        step_state = workflow_state.step_states[step_id]
        step_state.status = status
        
        if status == StepStatus.RUNNING:
            step_state.start()
        elif status == StepStatus.COMPLETED:
            step_state.complete()
        elif status == StepStatus.FAILED:
            step_state.fail("")

    def get_step_status(self, workflow_id: str, step_id: str) -> Optional[StepStatus]:
        """Get step status."""
        if workflow_id not in self.workflow_states:
            return None
        workflow_state = self.workflow_states[workflow_id]
        if step_id not in workflow_state.step_states:
            return None
        return workflow_state.step_states[step_id].status

    def set_step_result(self, workflow_id: str, step_id: str, result: Dict[str, Any]) -> None:
        """Set step result."""
        if workflow_id not in self.workflow_states:
            self.initialize_workflow(workflow_id)
        
        workflow_state = self.workflow_states[workflow_id]
        if step_id not in workflow_state.step_states:
            workflow_state.step_states[step_id] = StepState()
        
        workflow_state.step_states[step_id].complete(result)

    def get_step_result(self, workflow_id: str, step_id: str) -> Optional[Dict[str, Any]]:
        """Get step result."""
        if workflow_id not in self.workflow_states:
            return None
        workflow_state = self.workflow_states[workflow_id]
        if step_id not in workflow_state.step_states:
            return None
        return workflow_state.step_states[step_id].result

    def get_workflow_result(self) -> Optional[Dict[str, Any]]:
        """Get workflow result."""
        return self.workflow_result

    def set_workflow_result(self, result: Dict[str, Any]) -> None:
        """Set workflow result."""
        self.workflow_result = result

    def initialize_step(self, step_id: str) -> None:
        """Initialize step state.
        
        Args:
            step_id: Step ID to initialize
        """
        if step_id not in self.step_states:
            step_state = StepState()
            self.step_states[step_id] = step_state
            self.workflow_states[step_id] = WorkflowState(step_states={step_id: step_state})

    def start_step(self, step_id: str) -> None:
        """Start step execution.
        
        Args:
            step_id: Step ID to start
        """
        if step_id in self.step_states:
            self.step_states[step_id].start()

    def complete_step(self, step_id: str, result: Dict[str, Any] = None) -> None:
        """Complete step execution.
        
        Args:
            step_id: Step ID to complete
            result: Optional step result
        """
        if step_id in self.step_states:
            self.step_states[step_id].complete(result)

    def fail_step(self, step_id: str, error: str = None) -> None:
        """Fail step execution.
        
        Args:
            step_id: Step ID to fail
            error: Optional error message
        """
        if step_id in self.step_states:
            self.step_states[step_id].fail(error)