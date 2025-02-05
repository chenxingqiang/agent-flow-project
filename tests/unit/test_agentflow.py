"""
Tests for AgentFlow system functionality
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from agentflow.agents.agent import Agent
from agentflow.core.config import (
    AgentConfig, 
    ModelConfig, 
    WorkflowStepType
)
from agentflow.core.workflow_types import (
    Message,
    RetryPolicy,
    WorkflowConfig,
    WorkflowStep,
    StepConfig,
    ErrorPolicy
)
from agentflow.core.workflow_engine import WorkflowEngine
from agentflow.core.workflow_executor import WorkflowExecutor
from agentflow.core.exceptions import WorkflowExecutionError
from agentflow.ell2a.types.message import MessageRole, MessageType
from agentflow.core.isa.isa_manager import ISAManager
from agentflow.core.isa.instruction_selector import InstructionSelector
from agentflow.core.isa.instruction import Instruction
from agentflow.core.isa.types import InstructionType, InstructionStatus
from agentflow.core.isa.result import InstructionResult
from agentflow.core.types import AgentStatus
from pydantic import PrivateAttr

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="function")
def event_loop():
    """Create event loop for tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()

@pytest.fixture(scope="function")
def mock_ell2a():
    """Create a mock ELL2A integration."""
    mock = AsyncMock()
    mock.process_message.return_value = Message(
        content="Test response",
        metadata={
            "role": MessageRole.ASSISTANT,
            "type": "tool_result",
            "status": "success"
        }
    )
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    mock.configure = AsyncMock()
    return mock

@pytest.fixture(scope="function")
def mock_isa_manager():
    """Create a mock ISA manager."""
    mock = AsyncMock()
    mock.get_isa.return_value = "test_isa"
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture(scope="function")
def mock_instruction_selector():
    """Create a mock instruction selector."""
    mock = AsyncMock()
    mock.select_instruction.return_value = "test_instruction"
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture(scope="function")
def mock_state_manager():
    """Create a mock state manager."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture(scope="function")
def mock_metrics_manager():
    """Create a mock metrics manager."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture(scope="function")
async def agent(mock_ell2a, mock_isa_manager, mock_instruction_selector):
    """Create a test agent."""
    workflow_cfg = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="test_workflow",
        max_iterations=5,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=10,
            retry_policy=RetryPolicy(
                retry_delay=1.0,
                backoff=2.0,
                max_retries=3,
                max_delay=60.0,
                jitter=True
            ),
            continue_on_error=False
        ),
        steps=[
            WorkflowStep(
                id="step1",
                name="test_step",
                type=WorkflowStepType("transform"),
                required=True,
                optional=False,
                is_distributed=False,
                dependencies=[],
                config=StepConfig(
                    strategy="standard",
                    params={"substrategy": "standard"},
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )

    agent_config = AgentConfig(
        name="test_agent",
        type="generic",
        mode="sequential",
        model=ModelConfig(name="gpt-4", provider="openai"),
        workflow=workflow_cfg.model_dump()
    )
    agent = Agent(
        id=str(uuid.uuid4()),
        name=agent_config.name,
        type=agent_config.type,
        mode=agent_config.mode,
        config=agent_config.model_dump(),
        workflow=workflow_cfg.model_dump(),
        isa_manager=mock_isa_manager,
        instruction_selector=mock_instruction_selector
    )
    agent._ell2a = mock_ell2a
    agent._isa_manager = mock_isa_manager
    agent._instruction_selector = mock_instruction_selector
    await agent.initialize()
    return agent

@pytest.fixture(scope="function")
async def workflow_engine(mock_ell2a, mock_isa_manager, mock_instruction_selector, mock_state_manager, mock_metrics_manager):
    """Create a workflow engine for testing."""
    engine = WorkflowEngine()
    engine._ell2a = mock_ell2a
    engine._isa_manager = mock_isa_manager
    engine._instruction_selector = mock_instruction_selector
    engine.state_manager = mock_state_manager
    engine.metrics = mock_metrics_manager
    await engine.initialize()
    return engine

@pytest.mark.asyncio
async def test_workflow_initialization(workflow_engine):
    """Test workflow engine initialization."""
    engine = await workflow_engine
    assert engine is not None
    assert engine.state_manager is not None
    assert engine.metrics is not None

@pytest.mark.asyncio
async def test_agent_registration(workflow_engine, agent):
    """Test agent registration in workflow."""
    engine = await workflow_engine
    test_agent = await agent
    
    # Create workflow config from agent's workflow
    workflow_config = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="test_workflow",
        max_iterations=5,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=10,
            retry_policy=RetryPolicy(
                retry_delay=1.0,
                backoff=2.0,
                max_retries=3,
                max_delay=60.0,
                jitter=True
            ),
            continue_on_error=False
        ),
        steps=[
            WorkflowStep(
                id="step1",
                name="test_step",
                type=WorkflowStepType("transform"),
                required=True,
                optional=False,
                is_distributed=False,
                dependencies=[],
                config=StepConfig(
                    strategy="standard",
                    params={"substrategy": "standard"},
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )
    
    # Register workflow
    await engine.register_workflow(test_agent, workflow_config)
    assert test_agent.id in engine.workflows

@pytest.mark.asyncio
async def test_workflow_execution(workflow_engine, agent, mock_ell2a, mock_isa_manager):
    """Test workflow execution."""
    engine = await workflow_engine
    test_agent = await agent
    
    # Configure mock_ell2a
    mock_ell2a.process_message.return_value = Message(
        content="Test response",
        metadata={
            "role": MessageRole.ASSISTANT,
            "type": "tool_result",
            "status": "success"
        }
    )
    
    # Create workflow config
    async def test_transform(step_context):
        data = step_context.get("data", {})
        if isinstance(data, dict):
            data = data.get("data", "test")
        return {"data": f"processed_{data}"}

    workflow_config = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="test_workflow",
        max_iterations=5,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=10,
            retry_policy=RetryPolicy(
                retry_delay=1.0,
                backoff=2.0,
                max_retries=3,
                max_delay=60.0,
                jitter=True
            ),
            continue_on_error=False
        ),
        steps=[
            WorkflowStep(
                id="step1",
                name="test_step",
                type=WorkflowStepType("transform"),
                required=True,
                optional=False,
                is_distributed=False,
                dependencies=[],
                config=StepConfig(
                    strategy="standard",
                    params={"substrategy": "standard"},
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3,
                    execute=test_transform
                )
            )
        ]
    )
    
    # Register workflow
    await engine.register_workflow(test_agent, workflow_config)
    
    # Create a proper Message object
    message = Message(
        content="Test input",
        metadata={
            "role": MessageRole.USER,
            "type": MessageType.TEXT
        }
    )
    
    # Execute workflow
    result = await engine.execute_workflow(test_agent.id, message)
    assert result is not None
    assert isinstance(result, dict)
    assert result["status"] == "completed"

@pytest.mark.asyncio
async def test_workflow_error_handling(workflow_engine, mock_ell2a, mock_instruction_selector):
    """Test workflow error handling."""
    engine = await workflow_engine
    
    # Configure mock to raise error for this test
    error_message = "Test error"
    mock_ell2a.process_message.side_effect = Exception(error_message)
    
    async def test_execute(context: Dict[str, Any]) -> Dict[str, Any]:
        return await mock_ell2a.process_message(context)

    # Create workflow configuration
    workflow_config = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="test_workflow",
        max_iterations=5,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=10,
            retry_policy=RetryPolicy(
                retry_delay=1.0,
                backoff=2.0,
                max_retries=3,
                max_delay=60.0,
                jitter=True
            ),
            continue_on_error=False
        ),
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType("transform"),
                required=True,
                optional=False,
                is_distributed=False,
                dependencies=[],
                config=StepConfig(
                    strategy="custom",
                    params={},
                    execute=test_execute,
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )
    
    # Create agent configuration
    agent_config = AgentConfig(
        name="test_agent",
        type="generic",
        mode="sequential",
        model=ModelConfig(name="gpt-4", provider="openai"),
        workflow=workflow_config.model_dump()
    )
    
    # Create agent with workflow config
    agent = Agent(
        id=str(uuid.uuid4()),
        name=agent_config.name,
        type=agent_config.type,
        mode=agent_config.mode,
        config=agent_config.model_dump(),
        workflow=workflow_config.model_dump(),
        isa_manager=mock_ell2a,
        instruction_selector=mock_instruction_selector
    )
    await agent.initialize()
    
    # Register workflow
    await engine.register_workflow(agent, workflow_config)

    # Create test input data
    input_data = {
        "content": "Test input",
        "metadata": {
            "role": MessageRole.USER,
            "type": MessageType.TEXT
        }
    }

    # Test error handling
    with pytest.raises(WorkflowExecutionError, match=f"Step step-1 failed: {error_message}"):
        await engine.execute_workflow(agent.id, input_data)

@pytest.mark.asyncio
async def test_parallel_workflow_execution(workflow_engine, mock_ell2a, mock_isa_manager, mock_instruction_selector):
    """Test parallel workflow execution."""
    engine = await workflow_engine
    
    # Create multiple test agents
    num_agents = 3
    agents = []
    
    for i in range(num_agents):
        async def test_execute(context: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "test value"}

        step = WorkflowStep(
            id=f"step-{i}",
            name=f"step_{i}",
            type=WorkflowStepType("transform"),
            required=True,
            optional=False,
            is_distributed=False,
            dependencies=[],
            config=StepConfig(
                strategy="custom",
                params={},
                execute=test_execute,
                retry_delay=1.0,
                retry_backoff=2.0,
                max_retries=3
            )
        )

        workflow_cfg = WorkflowConfig(
            id=str(uuid.uuid4()),
            name=f"test_workflow_{i}",
            max_iterations=5,
            error_policy=ErrorPolicy(
                fail_fast=True,
                ignore_warnings=False,
                max_errors=10,
                retry_policy=RetryPolicy(
                    retry_delay=1.0,
                    backoff=2.0,
                    max_retries=3,
                    max_delay=60.0,
                    jitter=True
                ),
                continue_on_error=False
            ),
            steps=[step]
        )

        agent_config = AgentConfig(
            name=f"test_agent_{i}",
            type="generic",
            mode="parallel",
            model=ModelConfig(name="gpt-4", provider="openai"),
            workflow=workflow_cfg.model_dump()
        )

        agent = Agent(
            id=str(uuid.uuid4()),
            name=agent_config.name,
            type=agent_config.type,
            mode=agent_config.mode,
            config=agent_config.model_dump(),
            workflow=workflow_cfg.model_dump(),
            isa_manager=mock_isa_manager,
            instruction_selector=mock_instruction_selector
        )
        agent._ell2a = mock_ell2a
        agent._isa_manager = mock_isa_manager
        agent._instruction_selector = mock_instruction_selector
        await agent.initialize()
        agents.append(agent)

        # Register workflow
        await engine.register_workflow(agent, workflow_cfg)

@pytest.mark.asyncio
async def test_workflow_cleanup(workflow_engine, agent, mock_ell2a, mock_isa_manager, mock_instruction_selector):
    """Test workflow cleanup."""
    engine = await workflow_engine
    test_agent = await agent
    
    # Create workflow config
    workflow_config = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="test_workflow",
        max_iterations=5,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=10,
            retry_policy=RetryPolicy(
                retry_delay=1.0,
                backoff=2.0,
                max_retries=3,
                max_delay=60.0,
                jitter=True
            ),
            continue_on_error=False
        ),
        steps=[
            WorkflowStep(
                id="step1",
                name="test_step",
                type=WorkflowStepType("transform"),
                required=True,
                optional=False,
                is_distributed=False,
                dependencies=[],
                config=StepConfig(
                    strategy="standard",
                    params={"substrategy": "standard"},
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )
    
    # Register workflow
    await engine.register_workflow(test_agent, workflow_config)
    
    # Create a proper Message object
    message = Message(
        content="Test input",
        metadata={
            "role": MessageRole.USER,
            "type": MessageType.TEXT
        }
    )
    
    # Execute workflow
    result = await engine.execute_workflow(test_agent.id, message)
    assert result is not None
    
    # Cleanup workflow
    await engine.cleanup()
    assert len(engine.workflows) == 0

@pytest.mark.asyncio
async def test_workflow_registration(workflow_engine, mock_ell2a, mock_isa_manager, mock_instruction_selector):
    """Test workflow registration."""
    engine = await workflow_engine
    # Create agent
    agent_id = str(uuid.uuid4())
    agent_name = "Test Agent"
    
    config = AgentConfig(
        id=agent_id,
        name=agent_name,
        type="generic",
        version="1.0.0",
        mode="async",
        parameters={},
        metadata={"test_mode": True},
        model=ModelConfig(name="gpt-4", provider="openai"),
        system_prompt="You are a test agent for workflow engine testing"
    )

    async def test_execute(context: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": "test value"}

    workflow_cfg = WorkflowConfig(
        id="test-workflow-reg",
        name="test_workflow",
        max_iterations=5,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=10,
            retry_policy=RetryPolicy(
                retry_delay=1.0,
                backoff=2.0,
                max_retries=3,
                max_delay=60.0,
                jitter=True
            ),
            continue_on_error=False
        ),
        steps=[
            WorkflowStep(
                id="step1",
                name="test_step",
                type=WorkflowStepType("transform"),
                required=True,
                optional=False,
                is_distributed=False,
                dependencies=[],
                config=StepConfig(
                    strategy="standard",
                    params={"substrategy": "standard"},
                    execute=test_execute,
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )

    # Create agent instance
    agent = Agent(
        id=agent_id,
        name=agent_name,
        type=config.type,
        mode=config.mode,
        config=config.model_dump(),
        workflow=workflow_cfg.model_dump(),
        isa_manager=mock_isa_manager,
        instruction_selector=mock_instruction_selector
    )
    agent._ell2a = mock_ell2a
    agent._isa_manager = mock_isa_manager
    agent._instruction_selector = mock_instruction_selector
    await agent.initialize()

    # Register workflow
    await engine.register_workflow(agent, workflow_cfg)

    # Verify registration
    assert agent.id in engine.workflows
    assert engine.workflows[agent.id] == workflow_cfg

    # Test error handling - duplicate registration
    with pytest.raises(ValueError, match=f"Agent {agent.id} already registered"):
        await engine.register_workflow(agent, workflow_cfg)

    # Test error handling - invalid workflow
    with pytest.raises(ValueError, match="Agent must have a configuration"):
        invalid_agent = Agent(
            id=str(uuid.uuid4()),
            name="Invalid Agent",
            type="generic",
            mode="async",
            config={},  # Use empty config to trigger validation error
            workflow=None
        )
        # This line should not be reached as the Agent creation should raise the error
        await engine.register_workflow(invalid_agent, workflow_cfg)

@pytest.mark.asyncio
async def test_workflow_engine_execution():
    """Test workflow engine execution."""
    engine = WorkflowEngine()
    await engine.initialize()

    # Create agent
    agent_id = str(uuid.uuid4())
    agent_name = "Test Agent"
    
    config = AgentConfig(
        id=agent_id,
        name=agent_name,
        type="generic",
        version="1.0.0",
        mode="async",
        parameters={},
        metadata={"test_mode": True},
        model=ModelConfig(name="gpt-4", provider="openai"),
        system_prompt="You are a test agent for workflow engine testing"
    )

    async def test_execute(context: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(test_execute, 'side_effect'):
            raise test_execute.side_effect
        return {"result": "test value"}

    workflow_cfg = WorkflowConfig(
        id="test-workflow-exec",
        name="test_workflow",
        max_iterations=5,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=10,
            retry_policy=RetryPolicy(
                retry_delay=1.0,
                backoff=2.0,
                max_retries=3,
                max_delay=60.0,
                jitter=True
            ),
            continue_on_error=False
        ),
        steps=[
            WorkflowStep(
                id="step-1",
                name="Test Step 1",
                type=WorkflowStepType("transform"),
                required=True,
                optional=False,
                is_distributed=False,
                dependencies=[],
                config=StepConfig(
                    strategy="standard",
                    params={},
                    execute=test_execute,
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )

    # Create agent instance
    agent = Agent(
        id=agent_id,
        name=agent_name,
        type=config.type,
        mode=config.mode,
        config=config.model_dump(),
        workflow=workflow_cfg.model_dump(),
        isa_manager=None  # Mock will be set later
    )

    # Set up mocks
    mock_ell2a = AsyncMock()
    mock_ell2a.process_message.return_value = Message(
        content="Test response",
        metadata={
            "role": MessageRole.ASSISTANT,
            "type": MessageType.TOOL_RESULT,
            "status": "success"
        }
    )
    agent._ell2a = mock_ell2a

    mock_isa_manager = AsyncMock()
    mock_isa_manager.get_isa.return_value = "test_isa"
    agent._isa_manager = mock_isa_manager

    mock_instruction_selector = AsyncMock()
    mock_instruction_selector.select_instruction.return_value = "test_instruction"
    agent._instruction_selector = mock_instruction_selector

    await agent.initialize()

    # Register workflow
    await engine.register_workflow(agent, workflow_cfg)

    # Create test input data
    input_data = {
        "content": "Test input",
        "metadata": {
            "role": MessageRole.USER,
            "type": MessageType.TEXT
        }
    }

    # Execute workflow
    result = await engine.execute_workflow(agent.id, input_data)

    # Verify execution results
    assert result is not None
    assert isinstance(result, dict)
    assert result["status"] == "completed"
    assert "steps" in result
    assert len(result["steps"]) == 1
    assert result["steps"]["step-1"]["status"] == "success"

    # Test error handling
    test_execute.side_effect = Exception("Test error")
    with pytest.raises(WorkflowExecutionError, match="Step step-1 failed: Test error"):
        await engine.execute_workflow(agent.id, input_data)

    # Cleanup
    await engine.cleanup()
