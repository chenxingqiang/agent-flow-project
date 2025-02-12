"""Tests for base_types.py."""
import pytest
from agentflow.core.base_types import (
    AgentStatus,
    WorkflowStatus,
    StepStatus,
    WorkflowStepType,
    MessageRole,
)

def test_agent_status_values():
    """Test that AgentStatus enum has all required values."""
    assert AgentStatus.PENDING.value == "pending"
    assert AgentStatus.RUNNING.value == "running"
    assert AgentStatus.SUCCESS.value == "success"
    assert AgentStatus.FAILED.value == "failed"
    assert AgentStatus.CANCELLED.value == "cancelled"
    assert AgentStatus.PAUSED.value == "paused"
    assert AgentStatus.ERROR.value == "error"
    assert AgentStatus.INITIALIZED.value == "initialized"
    assert AgentStatus.COMPLETED.value == "completed"
    assert AgentStatus.STOPPED.value == "stopped"
    assert AgentStatus.WAITING.value == "waiting"

def test_workflow_status_values():
    """Test that WorkflowStatus enum has all required values."""
    assert WorkflowStatus.PENDING.value == "pending"
    assert WorkflowStatus.RUNNING.value == "running"
    assert WorkflowStatus.SUCCESS.value == "success"
    assert WorkflowStatus.FAILED.value == "failed"
    assert WorkflowStatus.CANCELLED.value == "cancelled"
    assert WorkflowStatus.PAUSED.value == "paused"
    assert WorkflowStatus.ERROR.value == "error"
    assert WorkflowStatus.INITIALIZED.value == "initialized"
    assert WorkflowStatus.COMPLETED.value == "completed"
    assert WorkflowStatus.STOPPED.value == "stopped"

def test_step_status_values():
    """Test that StepStatus enum has all required values."""
    assert StepStatus.PENDING.value == "pending"
    assert StepStatus.RUNNING.value == "running"
    assert StepStatus.SUCCESS.value == "success"
    assert StepStatus.FAILED.value == "failed"
    assert StepStatus.CANCELLED.value == "cancelled"
    assert StepStatus.PAUSED.value == "paused"
    assert StepStatus.ERROR.value == "error"
    assert StepStatus.INITIALIZED.value == "initialized"
    assert StepStatus.COMPLETED.value == "completed"
    assert StepStatus.STOPPED.value == "stopped"
    assert StepStatus.WAITING.value == "waiting"
    assert StepStatus.SKIPPED.value == "skipped"

def test_workflow_step_type_values():
    """Test that WorkflowStepType enum has all required values."""
    assert WorkflowStepType.SEQUENTIAL.value == "sequential"
    assert WorkflowStepType.DISTRIBUTED.value == "distributed"
    assert WorkflowStepType.CONDITIONAL.value == "conditional"

def test_message_role_values():
    """Test that MessageRole enum has all required values."""
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole.FUNCTION.value == "function"

def test_status_transitions():
    """Test that status transitions are valid."""
    # Test that we can convert string to enum
    assert AgentStatus("pending") == AgentStatus.PENDING
    assert WorkflowStatus("running") == WorkflowStatus.RUNNING
    assert StepStatus("completed") == StepStatus.COMPLETED

    # Test that we can convert enum to string
    assert str(AgentStatus.PENDING) == "pending"
    assert str(WorkflowStatus.RUNNING) == "running"
    assert str(StepStatus.COMPLETED) == "completed"

def test_invalid_status():
    """Test that invalid status values raise ValueError."""
    with pytest.raises(ValueError):
        AgentStatus("invalid")
    with pytest.raises(ValueError):
        WorkflowStatus("invalid")
    with pytest.raises(ValueError):
        StepStatus("invalid")
    with pytest.raises(ValueError):
        WorkflowStepType("invalid")
    with pytest.raises(ValueError):
        MessageRole("invalid") 