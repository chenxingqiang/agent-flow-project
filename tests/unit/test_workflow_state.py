import pytest
from datetime import datetime
from agentflow.core.workflow_state import (
    WorkflowStateManager,
    StepStatus,
    StepState
)

def test_workflow_state_initialization():
    """Test workflow state initialization"""
    manager = WorkflowStateManager()
    manager.initialize_step(1)
    
    assert 1 in manager.states
    assert manager.states[1].status == StepStatus.PENDING
    assert manager.states[1].retry_count == 0

def test_workflow_state_transitions():
    """Test workflow state transitions"""
    manager = WorkflowStateManager()
    manager.initialize_step(1)
    
    # Test start
    manager.start_step(1)
    assert manager.states[1].status == StepStatus.RUNNING
    assert isinstance(manager.states[1].start_time, datetime)
    
    # Test completion
    result = {"test": "result"}
    manager.complete_step(1, result)
    assert manager.states[1].status == StepStatus.COMPLETED
    assert manager.states[1].result == result
    assert isinstance(manager.states[1].end_time, datetime)
    
    # Test failure
    manager.initialize_step(2)
    manager.start_step(2)
    manager.fail_step(2, "Test error")
    assert manager.states[2].status == StepStatus.FAILED
    assert manager.states[2].error == "Test error"
    
    # Test retry
    manager.retry_step(2)
    assert manager.states[2].status == StepStatus.RETRYING
    assert manager.states[2].retry_count == 1 