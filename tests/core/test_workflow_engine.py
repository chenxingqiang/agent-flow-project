"""Unit tests for WorkflowEngine."""

import pytest
from agentflow.core.workflow import WorkflowEngine
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType
from agentflow.core.config import ConfigurationType
import asyncio

@pytest.fixture
def valid_workflow_def():
    """Return a valid workflow definition for testing."""
    return {
        "COLLABORATION": {
            "WORKFLOW": [
                {
                    "id": "step1",
                    "name": "test_step",
                    "type": ConfigurationType.TEST.value,
                    "config": {}
                }
            ]
        }
    }

@pytest.fixture
def workflow_config():
    """Return a workflow configuration for testing."""
    return WorkflowConfig(
        name="test_workflow",
        max_iterations=3,
        max_retries=3,
        timeout=60.0,
        steps=[]
    )

@pytest.fixture
def workflow_engine(valid_workflow_def, workflow_config):
    """Return a WorkflowEngine instance for testing."""
    return WorkflowEngine(valid_workflow_def, workflow_config)

def test_workflow_engine_initialization(workflow_engine):
    """Test WorkflowEngine initialization."""
    assert workflow_engine.workflow_def is not None
    assert workflow_engine.workflow_config is not None
    assert workflow_engine.state_manager is not None

def test_invalid_workflow_config():
    """Test WorkflowEngine initialization with invalid config."""
    with pytest.raises(ValueError, match="workflow_config must be an instance of WorkflowConfig"):
        WorkflowEngine(None, "invalid_config")

def test_invalid_workflow_definition():
    """Test WorkflowEngine initialization with invalid workflow definition."""
    with pytest.raises(ValueError, match="Workflow definition must contain COLLABORATION.WORKFLOW"):
        WorkflowEngine({}, WorkflowConfig(name="test", max_iterations=3, timeout=60.0))

@pytest.mark.asyncio
async def test_execute_with_invalid_max_retries(workflow_engine):
    """Test execute with invalid max_retries."""
    with pytest.raises(ValueError, match="max_retries must be greater than 0"):
        await workflow_engine.execute({}, max_retries=0)

@pytest.mark.asyncio
async def test_execute_basic_workflow(workflow_engine):
    """Test execution of a basic workflow."""
    context = {"input": "test"}
    result = await workflow_engine.execute(context)
    assert isinstance(result, dict)
