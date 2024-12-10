import pytest
from datetime import datetime
from agentflow.core.workflow_state import (
    WorkflowStateManager,
    StepStatus,
    StepState,
    WorkflowStatus
)

def test_workflow_state_initialization():
    """Test workflow state initialization"""
    manager = WorkflowStateManager()
    step_id = "step_1"
    
    # Initialize step
    manager.initialize_step(step_id)
    
    # Check initial state
    assert step_id in manager.step_states
    assert manager.get_step_status(step_id) == StepStatus.PENDING
    assert manager.get_step_retry_count(step_id) == 0
    
    # Check metadata initialization
    metadata = manager.get_step_metadata(step_id)
    assert metadata['start_time'] is None
    assert metadata['end_time'] is None
    assert metadata['error'] is None

def test_workflow_state_transitions():
    """Test workflow state transitions"""
    manager = WorkflowStateManager()
    step_id = "step_1"
    
    # Initialize and start step
    manager.initialize_step(step_id)
    manager.start_step(step_id)
    
    # Check running state
    assert manager.get_step_status(step_id) == StepStatus.RUNNING
    metadata = manager.get_step_metadata(step_id)
    assert metadata['start_time'] is not None
    
    # Test step success
    result = {"test": "result"}
    manager.set_step_status(step_id, StepStatus.SUCCESS)
    manager.set_step_result(step_id, result)
    
    assert manager.get_step_status(step_id) == StepStatus.SUCCESS
    assert manager.get_step_result(step_id) == result
    assert manager.get_step_success_count(step_id) == 1
    
    # Test step failure
    failure_step_id = "step_2"
    manager.initialize_step(failure_step_id)
    manager.start_step(failure_step_id)
    manager.set_step_status(failure_step_id, StepStatus.FAILED)
    
    assert manager.get_step_status(failure_step_id) == StepStatus.FAILED
    
    # Test retry mechanism
    manager.increment_retry_count(failure_step_id)
    assert manager.get_step_retry_count(failure_step_id) == 1
    
    # Test workflow status
    manager.initialize_workflow()
    assert manager.get_workflow_status() == WorkflowStatus.RUNNING
    
    manager.set_workflow_status(WorkflowStatus.COMPLETED)
    assert manager.get_workflow_status() == WorkflowStatus.COMPLETED