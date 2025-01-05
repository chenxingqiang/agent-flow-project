import pytest
from datetime import datetime
from agentflow.core.workflow_state import WorkflowState
from agentflow.core.exceptions import WorkflowStateError

@pytest.fixture
def workflow_state():
    return WorkflowState(
        workflow_id="test-workflow",
        name="Test Workflow",
        status="pending"
    )

class TestWorkflowState:
    def test_init_with_valid_params(self):
        state = WorkflowState(
            workflow_id="test-1",
            name="Test Workflow",
            status="pending"
        )
        assert state.workflow_id == "test-1"
        assert state.name == "Test Workflow"
        assert state.status == "pending"
        assert isinstance(state.created_at, datetime)
        assert state.updated_at is None

    def test_init_with_invalid_status(self):
        with pytest.raises(WorkflowStateError):
            WorkflowState(
                workflow_id="test-1",
                name="Test",
                status="invalid_status"
            )

    def test_update_status(self, workflow_state):
        workflow_state.update_status("running")
        assert workflow_state.status == "running"
        assert workflow_state.updated_at is not None

    def test_invalid_status_update(self, workflow_state):
        with pytest.raises(WorkflowStateError):
            workflow_state.update_status("invalid_status")

    def test_to_dict(self, workflow_state):
        state_dict = workflow_state.to_dict()
        assert isinstance(state_dict, dict)
        assert state_dict["workflow_id"] == "test-workflow"
        assert state_dict["name"] == "Test Workflow"
        assert state_dict["status"] == "pending"

    def test_from_dict(self):
        state_dict = {
            "workflow_id": "test-1",
            "name": "Test Workflow",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": None
        }
        state = WorkflowState.from_dict(state_dict)
        assert state.workflow_id == "test-1"
        assert state.name == "Test Workflow"
        assert state.status == "pending"

    def test_validate_state(self, workflow_state):
        workflow_state.validate()  # Should not raise
        
        # Test invalid state
        workflow_state._status = "invalid"  # Access private attr for testing
        with pytest.raises(WorkflowStateError):
            workflow_state.validate()

    def test_state_history(self, workflow_state):
        workflow_state.update_status("running")
        workflow_state.update_status("completed")
        
        history = workflow_state.status_history
        assert len(history) == 3
        assert history[0]["status"] == "pending"
        assert history[-1]["status"] == "completed"