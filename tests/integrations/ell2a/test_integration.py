"""Tests for ELL2A integration module."""

import pytest
import time
from agentflow.ell2a.integration import ELL2AIntegration
from agentflow.ell2a.types.message import Message, MessageRole

@pytest.fixture
def ell2a_integration():
    """Create an ELL2AIntegration instance for testing."""
    return ELL2AIntegration()

def test_ell2a_integration_initial_state(ell2a_integration):
    """Test the initial state of ELL2AIntegration."""
    assert ell2a_integration.enabled is True
    assert ell2a_integration.tracking_enabled is True
    assert ell2a_integration.config == {}
    assert ell2a_integration.metrics == {
        "function_calls": 0,
        "total_execution_time": 0.0,
        "errors": 0
    }

def test_ell2a_integration_configure(ell2a_integration):
    """Test configuration of ELL2AIntegration."""
    config = {
        "enabled": False,
        "tracking_enabled": False,
        "model": {"name": "test_model"}
    }
    
    ell2a_integration.configure(config)
    
    assert ell2a_integration.enabled is False
    assert ell2a_integration.tracking_enabled is False
    assert ell2a_integration.config == config

@pytest.mark.asyncio
async def test_ell2a_process_message(ell2a_integration):
    """Test message processing."""
    # Create an input message
    input_message = Message(
        role=MessageRole.USER,
        content="Test message"
    )
    
    # Process the message
    response = await ell2a_integration.process_message(input_message)
    
    # Verify metrics are updated
    assert ell2a_integration.metrics["function_calls"] == 1
    
    # Verify response
    assert isinstance(response, Message)
    assert response.role == MessageRole.ASSISTANT
    assert response.content == input_message.content
    assert "model" in response.metadata
    assert "timestamp" in response.metadata

def test_ell2a_integration_disabled():
    """Test integration when disabled."""
    integration = ELL2AIntegration()
    integration.configure({"enabled": False})
    
    # Create an input message
    input_message = Message(
        role=MessageRole.USER,
        content="Test message"
    )
    
    # Process the message
    async def test_disabled():
        response = await integration.process_message(input_message)
        assert response == input_message
    
    # Run the async test
    import asyncio
    asyncio.run(test_disabled())
