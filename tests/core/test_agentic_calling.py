"""Unit tests for agentic calling functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any

from agentflow.core.workflow import WorkflowEngine, WorkflowInstance
from agentflow.core.types import AgentStatus
from agentflow.agents.agent import Agent
from agentflow.core.config import AgentConfig, ModelConfig, WorkflowConfig

@pytest.fixture
async def workflow_engine():
    """Create a workflow engine for testing."""
    engine = WorkflowEngine()
    await engine.initialize()
    try:
        yield engine
    finally:
        await engine.cleanup()

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
            id="test-workflow",
            name="test-workflow",
            max_iterations=10,
            timeout=3600
        ),
        config={
            "algorithm": "PPO"
        }
    )

@pytest.fixture
async def agent(agent_config):
    """Create and initialize test agent."""
    _agent = Agent(config=agent_config)
    await _agent.initialize()
    _agent.metadata["test_mode"] = True
    try:
        yield _agent
    finally:
        await _agent.cleanup()

@pytest.mark.asyncio
async def test_workflow_execution_basic(workflow_engine, agent):
    """Test basic workflow execution."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    workflow_id = await engine.register_workflow(agent_instance)
    input_data = {"message": "Test input"}
    
    result = await engine.execute_workflow(workflow_id, input_data)
    assert result is not None
    assert result["status"] == "success"
    assert "content" in result

@pytest.mark.asyncio
async def test_empty_workflow(workflow_engine):
    """Test execution of empty workflow."""
    engine = await anext(workflow_engine)
    with pytest.raises(ValueError):
        await engine.execute_workflow("nonexistent", {})

@pytest.mark.asyncio
async def test_workflow_error_handling(workflow_engine, agent):
    """Test error handling during workflow execution."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    workflow_id = await engine.register_workflow(agent_instance)
    
    # Test with invalid input
    with pytest.raises(Exception):
        await engine.execute_workflow(workflow_id, None)

@pytest.mark.asyncio
async def test_workflow_timeout(workflow_engine, agent):
    """Test workflow timeout functionality."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    workflow_id = await engine.register_workflow(agent_instance)
    input_data = {"message": "Test input"}
    
    # Mock process_message to take longer than timeout
    async def slow_process(*args, **kwargs):
        await asyncio.sleep(0.2)
        return "Test response"
    
    with patch.object(agent_instance, 'process_message', side_effect=slow_process):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                engine.execute_workflow(workflow_id, input_data),
                timeout=0.1
            )

@pytest.mark.asyncio
async def test_parallel_workflow_execution(workflow_engine, agent_config):
    """Test parallel workflow execution."""
    engine = await anext(workflow_engine)
    agents = []
    workflow_ids = []
    
    # Create and register multiple agents
    for _ in range(3):
        agent = Agent(config=agent_config)
        await agent.initialize()
        agent.metadata["test_mode"] = True
        agents.append(agent)
        
        workflow_id = await engine.register_workflow(agent)
        workflow_ids.append(workflow_id)
    
    # Execute workflows in parallel
    input_data = {"message": "Test input"}
    tasks = [
        engine.execute_workflow(workflow_id, input_data)
        for workflow_id in workflow_ids
    ]
    
    results = await asyncio.gather(*tasks)
    assert all(result is not None for result in results)
    assert all(result["status"] == "success" for result in results)
    
    # Cleanup agents
    for agent in agents:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_workflow_cleanup(workflow_engine, agent):
    """Test workflow cleanup."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    workflow_id = await engine.register_workflow(agent_instance)
    input_data = {"message": "Test input"}
    
    await engine.execute_workflow(workflow_id, input_data)
    await engine.cleanup()
    
    assert not engine._initialized
    assert not engine.workflows 