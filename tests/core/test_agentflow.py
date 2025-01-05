"""
Tests for AgentFlow system functionality
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
import logging
import time
from typing import Dict, Any, Optional

from agentflow.agents.agent import Agent
from agentflow.core.config import AgentConfig, ModelConfig, WorkflowConfig
from agentflow.core.workflow import WorkflowEngine
from agentflow.ell2a.integration import ELL2AIntegration
from agentflow.ell2a.types.message import Message, MessageRole
from agentflow.core.instruction_selector import InstructionSelector
from agentflow.core.isa.isa_manager import ISAManager
from agentflow.core.types import AgentStatus

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for tests."""
    try:
        loop = asyncio.new_event_loop()
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
        return Message(
            role=MessageRole.ASSISTANT,
            content="Test response",
            metadata={
                "model": "test-model",
                "timestamp": time.time()
            }
        )
    
    mock.process_message = AsyncMock(side_effect=mock_process)
    mock.configure = MagicMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def mock_isa_manager():
    """Mock ISA manager."""
    mock = MagicMock(spec=ISAManager)
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def mock_instruction_selector():
    """Mock instruction selector."""
    mock = MagicMock(spec=InstructionSelector)
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def workflow_config():
    """Create test workflow config."""
    return WorkflowConfig(
        id="test-workflow-id",
        name="test-workflow",
        max_iterations=10,
        timeout=3600
    )

@pytest.fixture
def model_config():
    """Create test model config."""
    return ModelConfig(
        provider="default",
        name="gpt-3.5-turbo"
    )

@pytest.fixture
def agent_config(model_config, workflow_config):
    """Create test agent config."""
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        type="generic",
        system_prompt="You are a test agent",
        model=model_config,
        workflow=workflow_config,
        config={
            "algorithm": "PPO"
        }
    )

@pytest.fixture
async def agent(agent_config, mock_ell2a):
    """Create and initialize test agent."""
    _agent = Agent(config=agent_config)
    _agent._ell2a = mock_ell2a
    await _agent.initialize()
    _agent.metadata["test_mode"] = True
    try:
        yield _agent
    finally:
        await _agent.cleanup()

@pytest.fixture
async def workflow_engine(mock_ell2a, mock_isa_manager, mock_instruction_selector):
    """Create workflow engine with mocked components."""
    engine = WorkflowEngine()
    engine._ell2a = mock_ell2a
    engine._isa_manager = mock_isa_manager
    engine._instruction_selector = mock_instruction_selector
    await engine.initialize()
    try:
        yield engine
    finally:
        await engine.cleanup()

@pytest.mark.asyncio
async def test_workflow_initialization(workflow_engine):
    """Test workflow engine initialization."""
    engine = await anext(workflow_engine)
    assert engine._initialized
    assert engine._ell2a is not None
    assert engine._isa_manager is not None
    assert engine._instruction_selector is not None

@pytest.mark.asyncio
async def test_agent_registration(workflow_engine, agent):
    """Test agent registration in workflow."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    workflow_id = await engine.register_workflow(agent_instance)
    assert workflow_id in engine.workflows
    assert engine.workflows[workflow_id].agent.id == agent_instance.id

@pytest.mark.asyncio
async def test_workflow_execution(workflow_engine, agent, mock_ell2a):
    """Test workflow execution."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    workflow_id = await engine.register_workflow(agent_instance)
    input_data = {"message": "Test input"}
    result = await engine.execute_workflow(workflow_id, input_data)
    
    assert result is not None
    assert "Test response" in result.get("content", "")
    assert mock_ell2a.process_message.called

@pytest.mark.asyncio
async def test_workflow_error_handling(workflow_engine, agent, mock_ell2a):
    """Test workflow error handling."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    workflow_id = await engine.register_workflow(agent_instance)
    
    # Set up mock to raise an exception
    mock_ell2a.process_message.side_effect = Exception("Test error")
    
    with pytest.raises(Exception):
        await engine.execute_workflow(workflow_id, {"message": "Test input"})
    
    workflow = engine.workflows[workflow_id]
    assert workflow.agent.state.status == AgentStatus.FAILED

@pytest.mark.asyncio
async def test_parallel_workflow_execution(workflow_engine, agent_config, mock_ell2a):
    """Test parallel workflow execution."""
    engine = await anext(workflow_engine)
    agents = []
    workflow_ids = []
    
    # Create and register multiple agents
    for _ in range(3):
        agent = Agent(config=agent_config)
        agent._ell2a = mock_ell2a
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
    assert all("Test response" in result.get("content", "") for result in results)
    
    # Cleanup agents
    for agent in agents:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_workflow_cleanup(workflow_engine, agent, mock_ell2a, mock_isa_manager, mock_instruction_selector):
    """Test workflow cleanup."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    workflow_id = await engine.register_workflow(agent_instance)
    
    # Execute workflow
    input_data = {"message": "Test input"}
    await engine.execute_workflow(workflow_id, input_data)
    
    # Store mock references
    ell2a = mock_ell2a
    isa_manager = mock_isa_manager
    instruction_selector = mock_instruction_selector
    
    # Cleanup
    await engine.cleanup()
    
    # Verify cleanup
    assert not engine._initialized
    assert not engine.workflows
    assert ell2a.cleanup.called
    assert isa_manager.cleanup.called
    assert instruction_selector.cleanup.called
