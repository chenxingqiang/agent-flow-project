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
        
    def initialize_step(self, step_num: int):
        """Initialize step state"""
        self.states[step_num] = StepState(status=StepStatus.PENDING)
        
    def start_step(self, step_num: int):
        """Mark step as running"""
        state = self.states.get(step_num)
        if state:
            state.status = StepStatus.RUNNING
            state.start_time = datetime.now()
            
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
            
    def retry_step(self, step_num: int):
        """Mark step for retry"""
        state = self.states.get(step_num)
        if state:
            state.status = StepStatus.RETRYING
            state.retry_count += 1 