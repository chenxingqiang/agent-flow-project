"""
Tests for Agent functionality
"""

import pytest
from unittest.mock import MagicMock
import logging

from agentflow.core.agent import Agent
from agentflow.core.config_manager import AgentConfig, ModelConfig, WorkflowConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def mock_ray(mocker):
    """Mock Ray to prevent initialization"""
    mock_ray = mocker.MagicMock()
    mocker.patch('ray.init', return_value=None)
    return mock_ray

@pytest.fixture(autouse=True)
def mock_openai(mocker):
    """Mock OpenAI client"""
    mock_client = MagicMock()
    
    # Create a mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage.total_tokens = 10

    # Configure the mock to return the mock response
    mock_client.chat.completions.create.return_value = mock_response

    mocker.patch('openai.OpenAI', return_value=mock_client)
    return mock_client

@pytest.fixture
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

@pytest.fixture
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
    return agent

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization"""
    agent_instance = await agent
    assert agent_instance._config_obj.id == "test-agent"
    assert agent_instance._config_obj.name == "Test Agent"
    assert agent_instance._config_obj.agent_type == "generic"
    assert agent_instance._config_obj.system_prompt == "You are a test agent"
    assert agent_instance._config_obj.model.name == "gpt-3.5-turbo"
    assert agent_instance._config_obj.workflow.max_iterations == 10
    assert agent_instance._config_obj.config["algorithm"] == "PPO"

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

    # Create a mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage.total_tokens = 10

    # Create a mock client
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    # Set the mock client
    agent.client = mock_client

    message = "Test message"
    response = agent.process_message(message)
    
    # Debug logging
    logger.debug(f"Response type: {type(response)}")
    logger.debug(f"Response value: {repr(response)}")
    
    assert isinstance(response, str)
    assert response == "Test response"

    # Check workflow history
    history = agent.get_workflow_history()
    assert len(history) == 1
    assert history[0]['step'] == 'process_message'
    assert history[0]['status'] == 'success'
    assert history[0]['details']['message'] == message
    assert history[0]['details']['response'] == "Test response"
    assert history[0]['details']['tokens'] == 10

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
