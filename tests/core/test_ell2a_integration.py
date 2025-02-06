"""Tests for ELL2A integration functionality."""

import pytest
import asyncio
from typing import Dict, Any
from agentflow.core.ell2a_integration import ell2a_integration
from agentflow.core.workflow_types import WorkflowConfig, ErrorPolicy, WorkflowStep, StepConfig, WorkflowStepType
from agentflow.ell2a.types.message import Message, MessageRole, MessageType
from agentflow.ell2a.lmp import LMPType
from agentflow.ell2a.workflow import ELL2AWorkflow
import uuid

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
        id=str(uuid.uuid4()),
        name="Test Workflow",
        max_iterations=10,
        timeout=30,
        error_policy=ErrorPolicy(),
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="standard",
                    params={"test": "value"}
                )
            )
        ],
        distributed=False
    )

@pytest.mark.asyncio
async def test_ell2a_message_handling():
    """Test ELL2A message creation and handling."""
    # Create a message
    message = Message(
        role=MessageRole.USER,
        content="Test message",
        metadata={
            "role": MessageRole.USER,
            "type": MessageType.TEXT
        }
    )
    assert message.role == MessageRole.USER
    assert message.content == "Test message"
    assert message.metadata["type"] == MessageType.TEXT

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
        return Message(
            role=MessageRole.ASSISTANT,
            content="success",
            metadata={"type": "result"}
        )
    
    result = await test_function()
    assert result.content == "success"
    
    @ell2a_integration.with_ell2a(mode="complex")
    async def test_complex_function():
        return Message(
            role=MessageRole.ASSISTANT,
            content="complex success",
            metadata={"type": "result"}
        )
    
    result = await test_complex_function()
    assert result.content == "complex success"

@pytest.mark.asyncio
async def test_ell2a_performance_tracking():
    """Test ELL2A performance tracking."""
    
    @ell2a_integration.track_function()
    async def tracked_function():
        await asyncio.sleep(0.1)  # Simulate work
        return Message(
            role=MessageRole.ASSISTANT,
            content="tracked",
            metadata={"type": "result"}
        )
    
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
    # Create test messages
    messages = [
        Message(
            role=MessageRole.USER,
            content=f"Message {i}",
            type=MessageType.TEXT,
            metadata={
                "role": MessageRole.USER,
                "type": MessageType.TEXT
            }
        ) for i in range(3)
    ]
    
    # Process messages
    for msg in messages:
        await ell2a_integration.process_message(msg)
    
    # Get messages
    processed_messages = ell2a_integration.get_messages()
    assert len(processed_messages) == 3
    assert all(isinstance(msg, Message) for msg in processed_messages)

@pytest.mark.asyncio
async def test_ell2a_workflow_execution(sample_workflow_config):
    """Test ELL2A workflow execution."""
    # Create a message
    message = Message(
        role=MessageRole.USER,
        content="Test message",
        type=MessageType.TEXT,
        metadata={
            "role": MessageRole.USER,
            "type": MessageType.TEXT
        }
    )
    
    # Process message
    result = await ell2a_integration.process_message(message)
    assert result is not None
    assert isinstance(result, Message)
    assert result.type == MessageType.RESULT