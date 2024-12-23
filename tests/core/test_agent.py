"""
Tests for Agent functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pytest_asyncio import fixture as async_fixture
from pytest_mock import mocker

from agentflow.core.agent import Agent
from agentflow.core.config_manager import AgentConfig, ModelConfig, WorkflowConfig

@pytest.fixture(autouse=True)
def mock_openai(mocker):
    """Mock OpenAI client"""
    def mock_create(**kwargs):
        mock_message = MagicMock()
        mock_message.content = "Test response"
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(total_tokens=10)
        return mock_response
    
    mock_completion = MagicMock()
    mock_completion.create = mock_create
    mock_chat = MagicMock(completions=mock_completion)
    mock_client = MagicMock(chat=mock_chat)
    
    mocker.patch('openai.OpenAI', return_value=mock_client)
    return mock_client

@async_fixture
async def agent_config():
    """Create test agent config"""
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        model=ModelConfig(
            provider="default",
            name="gpt-3.5-turbo"
        ),
        workflow=WorkflowConfig(
            name="test-workflow",
            max_iterations=10
        ),
        config={
            "algorithm": "PPO"
        }
    )

@async_fixture(scope="function")
async def agent(mock_openai):
    """Create test agent instance"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        model=ModelConfig(
            provider="default",
            name="gpt-3.5-turbo"
        ),
        workflow=WorkflowConfig(
            name="test-workflow",
            max_iterations=10
        ),
        config={
            "algorithm": "PPO"
        }
    )
    agent = Agent(config)
    await agent.initialize()
    yield agent
    await agent.cleanup()

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization"""
    assert agent._config_obj.id == "test-agent"
    assert agent.name == "Test Agent"
    assert agent.type == "GENERIC"
    assert agent.max_iterations == 10

@pytest.mark.asyncio
async def test_agent_process_message():
    """Test agent message processing"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        model=ModelConfig(
            provider="default",
            name="gpt-3.5-turbo"
        ),
        workflow=WorkflowConfig(
            name="test-workflow",
            max_iterations=10
        ),
        config={
            "algorithm": "PPO"
        }
    )
    agent = Agent(config)
    await agent.initialize()

    message = "Test message"
    response = agent.process_message(message)
    assert response == "Test response"

@pytest.mark.asyncio
async def test_agent_error_handling():
    """Test agent error handling"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        model=ModelConfig(
            provider="default",
            name="gpt-3.5-turbo"
        ),
        workflow=WorkflowConfig(
            name="test-workflow",
            max_iterations=10
        ),
        config={
            "algorithm": "PPO"
        }
    )
    agent = Agent(config)
    await agent.initialize()

    with pytest.raises(Exception):
        agent.process_message(None)

    history = agent.get_workflow_history()
    assert any(step.get('status') == 'error' for step in history)

    await agent.cleanup()

@pytest.mark.asyncio
async def test_agent_cleanup():
    """Test agent cleanup"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        model=ModelConfig(
            provider="default",
            name="gpt-3.5-turbo"
        ),
        workflow=WorkflowConfig(
            name="test-workflow",
            max_iterations=10
        ),
        config={
            "algorithm": "PPO"
        }
    )
    agent = Agent(config)
    await agent.initialize()

    await agent.cleanup()
    assert not agent._initialized
    assert not agent.history
    assert not agent.errors
    assert not agent._workflow_history

@pytest.mark.asyncio
async def test_workflow_history():
    """Test workflow history tracking"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        model=ModelConfig(
            provider="default",
            name="gpt-3.5-turbo"
        ),
        workflow=WorkflowConfig(
            name="test-workflow",
            max_iterations=10
        ),
        config={
            "algorithm": "PPO"
        }
    )
    agent = Agent(config)
    await agent.initialize()

    history = agent.get_workflow_history()
    assert isinstance(history, list)

    await agent.cleanup()

@pytest.mark.asyncio
async def test_workflow_error_tracking():
    """Test workflow error tracking"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        model=ModelConfig(
            provider="default",
            name="gpt-3.5-turbo"
        ),
        workflow=WorkflowConfig(
            name="test-workflow",
            max_iterations=10
        ),
        config={
            "algorithm": "PPO"
        }
    )
    agent = Agent(config)
    await agent.initialize()

    with pytest.raises(ValueError):
        agent.process_message(None)

    history = agent.get_workflow_history()
    assert any(step.get('status') == 'error' for step in history)

    await agent.cleanup()
