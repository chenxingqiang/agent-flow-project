"""
Tests for Agent functionality
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
import logging
import time
from typing import AsyncGenerator

# Add anext for Python < 3.10
try:
    from builtins import anext
except ImportError:
    async def anext(ait):
        return await ait.__anext__()

from agentflow.agents.agent import Agent, AgentStatus
from agentflow.core.config import AgentConfig, ModelConfig, WorkflowConfig
from agentflow.ell2a.integration import ELL2AIntegration
from agentflow.ell2a.types.message import Message, MessageRole

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio

# Add event loop fixture
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for tests."""
    try:
        policy = asyncio.get_event_loop_policy()
        loop = policy.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
    finally:
        loop.close()

@pytest.fixture
def mock_ell2a():
    """Mock ELL2A integration."""
    mock = MagicMock(spec=ELL2AIntegration)
    mock.enabled = True
    mock.tracking_enabled = True
    mock.config = {}
    mock.metrics = {
        "function_calls": 0,
        "total_execution_time": 0.0,
        "errors": 0
    }
    
    async def mock_process(message: Message) -> Message:
        if not isinstance(message, Message):
            raise TypeError(f"Expected Message, got {type(message)}")
            
        # Return default response if no side effect
        if not hasattr(mock.process_message, '_mock_side_effect') or mock.process_message._mock_side_effect is None:
            return Message(
                role=MessageRole.ASSISTANT,
                content="Test response",
                metadata={
                    "model": "test-model",
                    "timestamp": time.time()
                }
            )
            
        # Handle side effect
        side_effect = mock.process_message._mock_side_effect
        if isinstance(side_effect, Exception):
            raise side_effect
        elif isinstance(side_effect, type) and issubclass(side_effect, Exception):
            raise side_effect("Test error")
        elif callable(side_effect):
            result = side_effect(message)
            if asyncio.iscoroutine(result):
                return await result
            return result
        else:
            return side_effect
    
    mock.process_message = AsyncMock(wraps=mock_process)
    mock.configure = MagicMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def agent_config():
    """Create test agent config."""
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        type="generic",
        system_prompt="You are a test agent",
        model=ModelConfig(
            provider="default",
            name="gpt-3.5-turbo"
        ),
        workflow=WorkflowConfig(
            name="test-workflow",
            max_iterations=10,
            timeout=3600
        ),
        config={
            "algorithm": "PPO"
        }
    )

@pytest.fixture
async def agent(agent_config, mock_ell2a):
    """Create and initialize test agent."""
    _agent = None
    try:
        _agent = Agent(config=agent_config)
        _agent._ell2a = mock_ell2a
        await _agent.initialize()
        _agent.metadata["test_mode"] = True
        yield _agent
    finally:
        if _agent:
            await _agent.cleanup()

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization."""
    agent_instance = await anext(agent)
    assert isinstance(agent_instance, Agent)
    assert agent_instance.state.status == AgentStatus.IDLE
    assert agent_instance._initialized

@pytest.mark.asyncio
async def test_message_processing(agent, mock_ell2a):
    """Test message processing."""
    agent_instance = await anext(agent)
    assert isinstance(agent_instance, Agent)

    test_message = "Test message"
    response = await agent_instance.process_message(test_message)

    assert response == "Test response"
    assert len(agent_instance.history) == 2
    assert agent_instance.state.status == AgentStatus.IDLE

@pytest.mark.asyncio
async def test_error_handling(agent, mock_ell2a):
    """Test error handling."""
    agent_instance = await anext(agent)
    assert isinstance(agent_instance, Agent)

    # Set up mock to raise an exception
    mock_ell2a.process_message.side_effect = Exception("Test error")
    agent_instance._ell2a = mock_ell2a

    with pytest.raises(Exception):
        await agent_instance.process_message("Test message")

    assert agent_instance.state.status == AgentStatus.FAILED
    assert len(agent_instance.errors) == 1
    assert agent_instance.state.last_error == "Test error"

@pytest.mark.asyncio
async def test_error_limit(agent, mock_ell2a):
    """Test error limit handling."""
    agent_instance = await anext(agent)
    assert isinstance(agent_instance, Agent)

    # Set up mock to raise an exception
    mock_ell2a.process_message.side_effect = Exception("Test error")
    agent_instance._ell2a = mock_ell2a

    # Generate more errors than the limit
    for i in range(agent_instance.max_errors + 5):
        with pytest.raises(Exception):
            await agent_instance.process_message("Test message")

    # Check that errors list is limited
    assert len(agent_instance.errors) == agent_instance.max_errors
    assert agent_instance.state.status == AgentStatus.FAILED
