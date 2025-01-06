"""Tests for ELL2A integration functionality."""

import pytest
import asyncio
from typing import Dict, Any
from agentflow.core.ell2a_integration import ell2a_integration
from agentflow.core.config import WorkflowConfig
from agentflow.ell2a.lmp import LMPType
from agentflow.ell2a.workflow import ELL2AWorkflow

@pytest.fixture(autouse=True)
def cleanup_ell2a():
    """Clean up ELL2A integration state before each test."""
    ell2a_integration.cleanup()
    yield
    ell2a_integration.cleanup()

@pytest.fixture
def sample_workflow_config():
    """Create a sample workflow configuration."""
    return WorkflowConfig(
        id="test-workflow",
        name="Test Workflow",
        use_ell2a=True,
        ell2a_mode="complex",
        ell2a_config={
            "model": "test-model",
            "max_tokens": 1000,
            "temperature": 0.7,
            "tools": [],
            "stream": False,
            "complex": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "timeout": 30.0,
                "track_performance": True,
                "track_memory": True
            }
        }
    )

@pytest.mark.asyncio
async def test_ell2a_message_handling():
    """Test ELL2A message creation and handling."""
    # Create a message
    message = ell2a_integration.create_message(
        role="user",
        content="Test message",
        metadata={"type": LMPType.AGENT}
    )
    
    # Add message to context
    ell2a_integration.add_message(message)
    
    # Get messages
    messages = ell2a_integration.get_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Test message"
    assert messages[0]["metadata"]["type"] == LMPType.AGENT

@pytest.mark.asyncio
async def test_ell2a_workflow_registration():
    """Test ELL2A workflow registration and management."""
    workflow = ELL2AWorkflow(
        name="test-workflow",
        description="Test workflow"
    )
    
    # Register workflow
    ell2a_integration.register_workflow(workflow.id, workflow)
    
    # Verify registration
    workflows = ell2a_integration.list_workflows()
    assert workflow.id in workflows
    
    # Unregister workflow
    ell2a_integration.unregister_workflow(workflow.id)
    workflows = ell2a_integration.list_workflows()
    assert workflow.id not in workflows

@pytest.mark.asyncio
async def test_ell2a_function_decoration():
    """Test ELL2A function decoration."""
    
    @ell2a_integration.with_ell2a(mode="simple")
    async def test_function():
        return {"result": "success"}
    
    result = await test_function()
    assert result["result"] == "success"
    
    @ell2a_integration.with_ell2a(mode="complex")
    async def test_complex_function():
        return {"result": "complex success"}
    
    result = await test_complex_function()
    assert result["result"] == "complex success"

@pytest.mark.asyncio
async def test_ell2a_performance_tracking():
    """Test ELL2A performance tracking."""
    
    @ell2a_integration.track_function()
    async def tracked_function():
        await asyncio.sleep(0.1)  # Simulate work
        return {"result": "tracked"}
    
    result = await tracked_function()
    metrics = ell2a_integration.get_metrics()
    
    assert "tracked_function" in metrics
    assert metrics["tracked_function"]["calls"] > 0
    assert metrics["tracked_function"]["total_time"] > 0

@pytest.mark.asyncio
async def test_ell2a_mode_configuration(sample_workflow_config):
    """Test ELL2A mode configuration."""
    # Configure ELL2A
    ell2a_integration.configure(sample_workflow_config.ell2a_config)
    
    # Get mode config
    simple_config = ell2a_integration.get_mode_config("simple")
    complex_config = ell2a_integration.get_mode_config("complex")
    
    assert simple_config["model"] == "test-model"
    assert complex_config["track_performance"] == True
    assert complex_config["track_memory"] == True

@pytest.mark.asyncio
async def test_ell2a_error_handling():
    """Test ELL2A error handling."""
    
    @ell2a_integration.with_ell2a(mode="simple")
    async def failing_function():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError):
        await failing_function()
    
    metrics = ell2a_integration.get_metrics()
    assert "failing_function" in metrics
    assert metrics["failing_function"]["errors"] > 0

@pytest.mark.asyncio
async def test_ell2a_context_management():
    """Test ELL2A context management."""
    # Create and add messages
    messages = [
        ell2a_integration.create_message(
            role="user",
            content=f"Message {i}",
            metadata={"type": LMPType.AGENT}
        ) for i in range(3)
    ]
    
    for msg in messages:
        ell2a_integration.add_message(msg)
    
    # Get context
    context = ell2a_integration.get_context()
    assert len(context["messages"]) == 3
    
    # Clear context
    ell2a_integration.clear_context()
    context = ell2a_integration.get_context()
    assert len(context["messages"]) == 0

@pytest.mark.asyncio
async def test_ell2a_workflow_execution(sample_workflow_config):
    """Test ELL2A workflow execution."""
    workflow = ELL2AWorkflow(
        name=sample_workflow_config.name,
        description="Test workflow execution"
    )
    
    # Add a test step
    workflow.add_step({
        "name": "test_step",
        "type": "process",
        "config": {"param": "value"}
    })
    
    # Initialize and register workflow
    await workflow.initialize()
    ell2a_integration.register_workflow(workflow.id, workflow)
    
    # Execute workflow
    result = await workflow.run({
        "input": "test data",
        "config": sample_workflow_config.ell2a_config
    })
    
    assert result is not None
    ell2a_integration.unregister_workflow(workflow.id) 