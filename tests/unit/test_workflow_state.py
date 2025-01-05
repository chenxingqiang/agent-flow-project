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
    workflow_id = "workflow_1"
    step_id = "step_1"
    
    # Initialize workflow and step
    manager.initialize_workflow(workflow_id)
    manager.update_step_status(workflow_id, step_id, StepStatus.PENDING)
    
    # Check initial state
    assert manager.get_step_status(workflow_id, step_id) == StepStatus.PENDING
    
    # Check workflow state
    assert manager.get_workflow_status(workflow_id) == WorkflowStatus.PENDING

def test_workflow_state_transitions():
    """Test workflow state transitions"""
    manager = WorkflowStateManager()
    workflow_id = "workflow_1"
    step_id = "step_1"
    
    # Initialize workflow and step
    manager.initialize_workflow(workflow_id)
    manager.update_step_status(workflow_id, step_id, StepStatus.PENDING)
    
    # Start step
    manager.update_step_status(workflow_id, step_id, StepStatus.RUNNING)
    
    # Check running state
    assert manager.get_step_status(workflow_id, step_id) == StepStatus.RUNNING
    
    # Test step completion
    result = {"test": "result"}
    manager.update_step_status(workflow_id, step_id, StepStatus.COMPLETED)
    manager.set_step_result(workflow_id, step_id, result)
    
    assert manager.get_step_status(workflow_id, step_id) == StepStatus.COMPLETED
    assert manager.get_step_result(workflow_id, step_id) == result
    
    # Test step failure
    failure_step_id = "step_2"
    manager.update_step_status(workflow_id, failure_step_id, StepStatus.RUNNING)
    manager.update_step_status(workflow_id, failure_step_id, StepStatus.FAILED)
    
    assert manager.get_step_status(workflow_id, failure_step_id) == StepStatus.FAILED
    
    # Test workflow status transitions
    manager.update_workflow_status(workflow_id, WorkflowStatus.RUNNING)
    assert manager.get_workflow_status(workflow_id) == WorkflowStatus.RUNNING
    
    manager.update_workflow_status(workflow_id, WorkflowStatus.COMPLETED)
    assert manager.get_workflow_status(workflow_id) == WorkflowStatus.COMPLETED