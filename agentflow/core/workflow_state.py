from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

class StepStatus(Enum):
    """Step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class StepState:
    """State of a workflow step"""
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class WorkflowStateManager:
    """Manages workflow execution state"""

    def __init__(self):
        """Initialize workflow state manager"""
        self.reset_state()

    def reset_state(self):
        """Reset workflow state to initial values"""
        self.workflow_status = WorkflowStatus.PENDING
        self.step_states: Dict[str, StepState] = {}
        self.step_results: Dict[str, Any] = {}
        self.step_metadata: Dict[str, Dict[str, Any]] = {}

    def initialize_workflow(self):
        """Initialize workflow execution"""
        self.reset_state()
        self.workflow_status = WorkflowStatus.RUNNING

    def set_workflow_status(self, status: WorkflowStatus):
        """Set workflow status"""
        self.workflow_status = status

    def get_workflow_status(self) -> WorkflowStatus:
        """Get current workflow status"""
        return self.workflow_status

    def set_step_status(self, step_id: str, status: StepStatus):
        """Set status for a specific step"""
        if step_id not in self.step_states:
            self.step_states[step_id] = StepState(status=status)
        else:
            self.step_states[step_id].status = status
            if status == StepStatus.RUNNING:
                self.step_states[step_id].start_time = datetime.now()
            elif status in [StepStatus.SUCCESS, StepStatus.FAILED]:
                self.step_states[step_id].end_time = datetime.now()

    def get_step_status(self, step_id: str) -> StepStatus:
        """Get status of a specific step"""
        return self.step_states.get(step_id, StepState()).status

    def set_step_result(self, step_id: str, result: Dict[str, Any]):
        """Store result for a specific step"""
        if step_id not in self.step_states:
            self.step_states[step_id] = StepState()
        self.step_states[step_id].result = result
        self.step_results[step_id] = result

    def get_step_result(self, step_id: str) -> Dict[str, Any]:
        """Get result of a specific step"""
        return self.step_results.get(step_id, {})

    def reset_step_retry_count(self, step_id: str):
        """Reset retry count for a step"""
        if step_id not in self.step_states:
            self.step_states[step_id] = StepState()
        self.step_states[step_id].retry_count = 0

    def increment_retry_count(self, step_id: str):
        """Increment retry count for a step"""
        if step_id not in self.step_states:
            self.step_states[step_id] = StepState()
        self.step_states[step_id].retry_count += 1

    def get_step_retry_count(self, step_id: str) -> int:
        """Get current retry count for a step"""
        return self.step_states.get(step_id, StepState()).retry_count

    def update_step_metadata(self, step_id: str, metadata: Dict[str, Any]):
        """Update metadata for a step"""
        if step_id not in self.step_metadata:
            self.step_metadata[step_id] = {}
        self.step_metadata[step_id].update(metadata)

    def get_step_metadata(self, step_id: str) -> Dict[str, Any]:
        """Get metadata for a step"""
        return self.step_metadata.get(step_id, {})

    def initialize_step(self, step_id: str):
        """Initialize state for a step"""
        if step_id not in self.step_states:
            self.step_states[step_id] = StepState()
            self.step_results[step_id] = {}
            self.step_metadata[step_id] = {
                'start_time': None,
                'end_time': None,
                'error': None
            }
        return self.step_states[step_id]

    def start_step(self, step_id: str):
        """Start a step by updating its status and start time"""
        if step_id not in self.step_states:
            self.initialize_step(step_id)
        
        self.set_step_status(step_id, StepStatus.RUNNING)
        
        # Update metadata with start time
        if step_id not in self.step_metadata:
            self.step_metadata[step_id] = {}
        self.step_metadata[step_id]['start_time'] = datetime.now()
        
        return self.step_states[step_id]

    def get_step_success_count(self, step_id: str) -> int:
        """Get the number of times a step has been successfully executed"""
        step_state = self.step_states.get(step_id)
        if step_state and step_state.status == StepStatus.SUCCESS:
            return 1
        return 0