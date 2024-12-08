"""
Tests for Agent functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from agentflow.core.agent import Agent
from agentflow.core.config_manager import AgentConfig, ModelConfig

@pytest.fixture
def agent_config():
    """Create test agent configuration"""
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        type="test",
        model=ModelConfig(
            name="test-model",
            provider="test"
        ),
        system_prompt="You are a test agent"
    )

@pytest.fixture
def agent(agent_config):
    """Create test agent instance"""
    return Agent(agent_config)

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
        type="test",
        model=ModelConfig(
            name="test-model",
            provider="test"
        ),
        system_prompt="You are a test agent"
    )
    
    agent = Agent(config)
    
    # Mock LLM response
    with patch("agentflow.core.agent.Agent._call_llm") as mock_llm:
        mock_llm.return_value = {"response": "Test response"}
        
        result = await agent.process({"input": "Test input"})
        
        assert result["response"] == "Test response"
        assert agent.token_count > 0
        assert agent.last_latency > 0

@pytest.mark.asyncio
async def test_agent_error_handling():
    """Test agent error handling"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        type="test",
        model=ModelConfig(
            name="test-model",
            provider="test"
        ),
        system_prompt="You are a test agent"
    )
    
    agent = Agent(config)
    
    # Mock LLM error
    with patch("agentflow.core.agent.Agent._call_llm") as mock_llm:
        mock_llm.side_effect = Exception("Test error")
        
        with pytest.raises(Exception) as exc:
            await agent.process({"input": "Test input"})
            
        assert str(exc.value) == "Test error"

@pytest.mark.asyncio
async def test_agent_cleanup():
    """Test agent cleanup"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        type="test",
        model=ModelConfig(
            name="test-model",
            provider="test"
        ),
        system_prompt="You are a test agent"
    )
    
    agent = Agent(config)
    
    # Add some test data
    agent.token_count = 100
    agent.last_latency = 50
    
    await agent.cleanup()
    
    assert agent.token_count == 0
    assert agent.last_latency == 0
