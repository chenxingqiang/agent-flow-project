import pytest
from datetime import datetime
from agentflow.core.workflow_state import WorkflowState
from agentflow.core.exceptions import WorkflowStateError
from agentflow.core.workflow_types import WorkflowStatus
from pydantic import ValidationError

@pytest.fixture
def workflow_state():
    return WorkflowState(
        workflow_id="test-workflow",
        name="Test Workflow",
        status=WorkflowStatus.PENDING
    )

class TestWorkflowState:
    def test_init_with_valid_params(self):
        state = WorkflowState(
            workflow_id="test-1",
            name="Test Workflow",
            status=WorkflowStatus.PENDING
        )
        assert state.workflow_id == "test-1"
        assert state.name == "Test Workflow"
        assert state.status == WorkflowStatus.PENDING
        assert isinstance(state.created_at, datetime)
        assert state.updated_at is None

    def test_init_with_invalid_status(self):
        with pytest.raises(ValueError):
            WorkflowState(
                workflow_id="test-1",
                name="Test",
                status="invalid_status"
            )

    def test_update_status(self, workflow_state):
        workflow_state.update_status(WorkflowStatus.RUNNING)
        assert workflow_state.status == WorkflowStatus.RUNNING
        assert len(workflow_state.state_history) == 1

    def test_invalid_status_update(self, workflow_state):
        with pytest.raises(ValueError):
            workflow_state.update_status("invalid_status")

    def test_to_dict(self, workflow_state):
        state_dict = workflow_state.to_dict()
        assert isinstance(state_dict, dict)
        assert state_dict["workflow_id"] == "test-workflow"
        assert state_dict["name"] == "Test Workflow"
        assert state_dict["status"] == WorkflowStatus.PENDING

    def test_from_dict(self):
        state_dict = {
            "workflow_id": "test-1",
            "name": "Test Workflow",
            "status": WorkflowStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "updated_at": None
        }
        state = WorkflowState.model_validate(state_dict)
        assert state.workflow_id == "test-1"
        assert state.name == "Test Workflow"
        assert state.status == WorkflowStatus.PENDING

    def test_validate_state(self, workflow_state):
        workflow_state.validate()  # Should not raise
        
        # Test with invalid status using update_status method
        with pytest.raises(ValueError) as exc_info:
            workflow_state.update_status("invalid")
        assert "Invalid status" in str(exc_info.value)
        
        # Reset to valid status
        workflow_state.update_status(WorkflowStatus.PENDING)
        workflow_state.validate()  # Should not raise

    def test_state_history(self, workflow_state):
        workflow_state.update_status(WorkflowStatus.RUNNING)
        workflow_state.update_status(WorkflowStatus.SUCCESS)
        assert len(workflow_state.state_history) == 2
        assert workflow_state.state_history[0]["status"] == WorkflowStatus.PENDING
        assert workflow_state.state_history[1]["status"] == WorkflowStatus.RUNNING