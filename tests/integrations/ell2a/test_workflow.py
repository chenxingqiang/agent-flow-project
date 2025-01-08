"""Tests for ELL2A workflow module."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from agentflow.ell2a.workflow import ELL2AWorkflow
from agentflow.core.ell2a_integration import ell2a_integration

@pytest.fixture
def sample_workflow():
    """Create a sample ELL2AWorkflow for testing."""
    return ELL2AWorkflow(
        name="test_workflow", 
        description="A workflow for testing"
    )

def test_workflow_initialization(sample_workflow):
    """Test workflow initialization."""
    assert sample_workflow.name == "test_workflow"
    assert sample_workflow.description == "A workflow for testing"
    assert sample_workflow.steps == []
    assert sample_workflow.id == "workflow_test_workflow"
    assert not sample_workflow._initialized

def test_workflow_add_step(sample_workflow):
    """Test adding steps to the workflow."""
    # Valid step
    valid_step = {
        "name": "step1", 
        "type": "test", 
        "config": {"param": "value"}
    }
    sample_workflow.add_step(valid_step)
    assert len(sample_workflow.steps) == 1
    assert sample_workflow.steps[0] == valid_step

    # Invalid step - missing required fields
    with pytest.raises(ValueError, match="Step missing required fields"):
        sample_workflow.add_step({"name": "incomplete"})

    # Invalid step - wrong type
    with pytest.raises(ValueError, match="Step must be a dictionary"):
        sample_workflow.add_step("not a dictionary")

@pytest.mark.asyncio
async def test_workflow_initialize(sample_workflow):
    """Test workflow initialization method."""
    # Mock the ell2a_integration.register_workflow method
    with patch('agentflow.core.ell2a_integration.ell2a_integration.register_workflow') as mock_register:
        await sample_workflow.initialize()
        
        # Verify registration
        mock_register.assert_called_once_with(
            "workflow_test_workflow", 
            sample_workflow
        )
        
        # Verify initialization flag
        assert sample_workflow._initialized is True

@pytest.mark.asyncio
async def test_workflow_run(sample_workflow):
    """Test workflow run method."""
    # Prepare a sample context
    context = {"input": "test_data"}
    
    # Add a step to the workflow
    sample_workflow.add_step({
        "name": "test_step", 
        "type": "test", 
        "config": {"param": "value"}
    })
    
    # Verify the run method exists and is async
    assert asyncio.iscoroutinefunction(sample_workflow.run)
    
    # Run the workflow and check it returns a dictionary
    result = await sample_workflow.run(context)
    assert isinstance(result, dict)