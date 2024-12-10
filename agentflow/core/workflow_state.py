from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

class StepStatus(Enum):
    """Workflow step status"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    SUCCESS = "SUCCESS"

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
    metadata: Optional[Dict[str, Any]] = field(default=None, init=False)

class WorkflowStateManager:
    """Manages the state of a workflow execution"""
    def __init__(self):
        self.step_states = {}
        self.retry_counts = {}
        self.success_counts = {}
        self.step_metadata = {}
        self.workflow_status = WorkflowStatus.PENDING
        
    def initialize_step(self, step_id: str):
        """Initialize state for a step"""
        # Initialize state if not exists
        if step_id not in self.step_states:
            self.step_states[step_id] = StepStatus.PENDING
            
        # Initialize retry count if not exists
        if step_id not in self.retry_counts:
            self.retry_counts[step_id] = 0
            
        # Initialize success count if not exists
        if step_id not in self.success_counts:
            self.success_counts[step_id] = 0
            
        # Initialize metadata if not exists
        if step_id not in self.step_metadata:
            self.step_metadata[step_id] = {
                'start_time': None,
                'end_time': None,
                'error': None,
                'output_format': None
            }
    
    def start_step(self, step_id: str):
        """Mark a step as running"""
        self.step_states[step_id] = StepStatus.RUNNING
        self.step_metadata[step_id]['start_time'] = datetime.now()
    
    def retry_step(self, step_id: str):
        """Mark a step for retry"""
        self.step_states[step_id] = StepStatus.RETRYING
    
    def increment_retry_count(self, step_id: str):
        """Increment the retry count for a step"""
        if step_id not in self.retry_counts:
            self.retry_counts[step_id] = 0
        self.retry_counts[step_id] += 1
    
    def get_step_retry_count(self, step_id: str) -> int:
        """Get the current retry count for a step"""
        return self.retry_counts.get(step_id, 0)
    
    def reset_step_retry_count(self, step_id: str):
        """Reset the retry count for a step"""
        self.retry_counts[step_id] = 0
    
    def update_step_status(self, step_id: str, status: StepStatus):
        """Update the status of a step"""
        self.step_states[step_id] = status
        if status in [StepStatus.COMPLETED, StepStatus.SUCCESS]:
            self.step_metadata[step_id]['end_time'] = datetime.now()
        
    def get_step_status(self, step_id: str) -> StepStatus:
        """Get the current status of a step"""
        return self.step_states.get(step_id, StepStatus.PENDING)
    
    def increment_step_success_count(self, step_id: str):
        """Increment the success count for a step"""
        if step_id not in self.success_counts:
            self.success_counts[step_id] = 0
        self.success_counts[step_id] += 1
    
    def get_step_success_count(self, step_id: str) -> int:
        """Get the success count for a step"""
        return self.success_counts.get(step_id, 0)
        
    def get_step_metadata(self, step_id: str) -> Dict[str, Any]:
        """Get the metadata for a step"""
        return self.step_metadata.get(step_id, {})
    
    def update_step_metadata(self, step_id: str, metadata: Dict[str, Any]):
        """Update metadata for a step"""
        if step_id not in self.step_metadata:
            self.step_metadata[step_id] = {}
        self.step_metadata[step_id].update(metadata)
        
    def initialize_workflow(self):
        """Initialize the workflow state"""
        self.workflow_status = WorkflowStatus.RUNNING
        
    def set_workflow_status(self, status: WorkflowStatus):
        """Set the overall workflow status"""
        self.workflow_status = status
        
    def get_workflow_status(self) -> WorkflowStatus:
        """Get the current workflow status"""
        return self.workflow_status
    
    def set_step_status(self, step_id: str, status: StepStatus):
        """Set the status of a specific step"""
        self.step_states[step_id] = status
        
        if status == StepStatus.SUCCESS:
            self.increment_step_success_count(step_id)
            self.step_metadata[step_id]['end_time'] = datetime.now()
        elif status == StepStatus.FAILED:
            self.step_metadata[step_id]['error'] = f"Step {step_id} failed"
            self.step_metadata[step_id]['end_time'] = datetime.now()
            
    def set_step_result(self, step_id: str, result: Dict[str, Any]):
        """Set the result for a specific step"""
        if step_id not in self.step_metadata:
            self.step_metadata[step_id] = {}
        self.step_metadata[step_id]['result'] = result
        
    def get_step_result(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get the result for a specific step"""
        return self.step_metadata.get(step_id, {}).get('result')