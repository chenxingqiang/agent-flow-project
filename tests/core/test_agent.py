"""
Tests for Agent functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from pytest_asyncio import fixture as async_fixture

from agentflow.core.agent import Agent
from agentflow.core.config_manager import AgentConfig, ModelConfig

@async_fixture
async def agent_config():
    """Create test agent config"""
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        config={
            "algorithm": "PPO"
        }
    )

@async_fixture(scope="function")
async def agent():
    """Create test agent instance"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        config={
            "algorithm": "PPO"
        }
    )
    agent = await Agent.create(config)
    yield agent
    await agent.cleanup()

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization"""
    assert agent.config.id == "test-agent"
    assert agent.config.name == "Test Agent"
    assert agent.token_count == 0
    assert agent.last_latency == 0
    assert agent.memory_usage == 0

@pytest.mark.asyncio
async def test_agent_process_message():
    """Test agent message processing"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        config={
            "algorithm": "PPO"
        }
    )
    
    agent = await Agent.create(config)
    message = "Test message"
    
    response = await agent.process_message(message)
    assert response is not None

@pytest.mark.asyncio
async def test_agent_error_handling():
    """Test agent error handling"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        config={
            "algorithm": "PPO"
        }
    )
    
    agent = await Agent.create(config)
    
    # Test invalid message
    with pytest.raises(Exception):
        await agent.process_message(None)

    # Mock LLM error
    with patch("agentflow.core.agent.Agent._call_llm") as mock_llm:
        mock_llm.side_effect = Exception("Test error")
        
        with pytest.raises(Exception) as exc:
            await agent.process_message("message")
            
        assert str(exc.value) == "Test error"

@pytest.mark.asyncio
async def test_agent_cleanup():
    """Test agent cleanup"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        config={
            "algorithm": "PPO"
        }
    )
    
    agent = await Agent.create(config)
    await agent.cleanup()
    
    # Verify cleanup actions
    assert agent.history == []

@pytest.mark.asyncio
async def test_workflow_history():
    """Test workflow history tracking"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        config={
            "algorithm": "PPO"
        }
    )
    
    agent = await Agent.create(config)
    message = "Test message"
    
    # Process message and check history
    await agent.process_message(message)
    assert len(agent.history) > 0
    assert agent.history[0]["message"] == message

@pytest.mark.asyncio
async def test_workflow_error_tracking():
    """Test workflow error tracking"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        agent_type="generic",
        system_prompt="You are a test agent",
        config={
            "algorithm": "PPO"
        }
    )
    
    agent = await Agent.create(config)
    
    # Test error tracking
    with pytest.raises(Exception):
        await agent.process_message(None)
    
    assert len(agent.errors) > 0
    assert "error" in agent.errors[0]
