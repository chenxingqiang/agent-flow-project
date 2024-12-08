from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class StepState:
    status: StepStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class WorkflowStateManager:
    def __init__(self):
        self.states: Dict[int, StepState] = {}
        self.workflow_status = WorkflowStatus.PENDING
        self.current_step = 0
        
    def initialize_step(self, step_num: int):
        """Initialize step state"""
        self.states[step_num] = StepState(status=StepStatus.PENDING)
        
    def initialize_step_state(self, step_num: int):
        """Initialize step state"""
        self.states[step_num] = StepState(status=StepStatus.PENDING)
        
    def start_step(self, step_num: int):
        """Mark step as running"""
        state = self.states.get(step_num)
        if state:
            state.status = StepStatus.RUNNING
            state.start_time = datetime.now()
            self.current_step = step_num
            if self.workflow_status == WorkflowStatus.PENDING:
                self.workflow_status = WorkflowStatus.RUNNING
            
    def complete_step(self, step_num: int, result: Dict[str, Any]):
        """Mark step as completed"""
        state = self.states.get(step_num)
        if state:
            state.status = StepStatus.COMPLETED
            state.end_time = datetime.now()
            state.result = result
            
    def fail_step(self, step_num: int, error: str):
        """Mark step as failed"""
        state = self.states.get(step_num)
        if state:
            state.status = StepStatus.FAILED
            state.end_time = datetime.now()
            state.error = error
            self.workflow_status = WorkflowStatus.FAILED
            
    def retry_step(self, step_num: int):
        """Mark step for retry"""
        state = self.states.get(step_num)
        if state:
            state.status = StepStatus.RETRYING
            state.retry_count += 1
            
    def increment_step_retry_count(self, step_num: int):
        """Increment retry count for a specific step"""
        state = self.states.get(step_num)
        if state:
            state.retry_count += 1
            
    def increment_step_success_count(self, step_num: int):
        """Increment success count for a specific step"""
        # This method can be expanded in future to track more metrics
        pass
        
    def get_workflow_status(self) -> WorkflowStatus:
        """Get current workflow status"""
        return self.workflow_status
        
    def update_workflow_status(self, status: WorkflowStatus):
        """Update workflow status"""
        self.workflow_status = status
        
    def get_current_step(self) -> int:
        """Get current step number"""
        return self.current_step
        
    def get_step_status(self, step_num: int) -> Optional[StepStatus]:
        """Get status of a specific step"""
        state = self.states.get(step_num)
        return state.status if state else None

    def get_step_retry_count(self, step_num: int) -> int:
        """Get retry count for a specific step"""
        state = self.states.get(step_num)
        return state.retry_count if state else 0