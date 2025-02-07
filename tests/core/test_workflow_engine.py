"""Tests for workflow engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, Optional, Union, cast
from pydantic import ValidationError

from agentflow.core.workflow_engine import WorkflowEngine
from agentflow.core.workflow_types import (
    WorkflowConfig,
    WorkflowStep,
    WorkflowStepType,
    StepConfig,
    WorkflowStatus,
    RetryPolicy
)
from agentflow.core.exceptions import WorkflowExecutionError
from agentflow.agents.agent import Agent
from agentflow.core.config import AgentConfig, ModelConfig

@pytest.fixture
def workflow_config() -> WorkflowConfig:
    """Return a workflow configuration for testing."""
    return WorkflowConfig.model_validate(
        {
            "id": "test-workflow-1",
            "name": "test_workflow",
            "max_iterations": 3,
            "timeout": 60.0,
            "steps": [
                {
                    "id": "step1",
                    "name": "test_step",
                    "type": WorkflowStepType.TRANSFORM,
                    "description": "Test transformation step",
                    "config": {
                        "strategy": "custom",
                        "params": {"method": "test"}
                    }
                }
            ]
        },
        context={"is_initialization": True}
    )

@pytest.mark.asyncio
async def test_invalid_workflow_config():
    """Test WorkflowEngine initialization with invalid config."""
    engine = WorkflowEngine()
    with pytest.raises(ValueError, match="workflow_config must be an instance of WorkflowConfig, a dictionary, or None"):
        # Cast to Dict[str, Any] to satisfy type checker
        invalid_config = cast(Dict[str, Any], "invalid_config")
        await engine.initialize(workflow_def=None, workflow_config=invalid_config)

@pytest.mark.asyncio
async def test_invalid_workflow_definition():
    """Test WorkflowEngine initialization with invalid workflow definition."""
    engine = WorkflowEngine()
    with pytest.raises(WorkflowExecutionError, match="Empty workflow: no workflow steps defined in COLLABORATION.WORKFLOW"):
        await engine.initialize(workflow_def={})
        # Create and register a workflow to trigger validation
        agent_config = AgentConfig(
            name="test_agent",
            type="generic",
            model=ModelConfig(name="gpt-4", provider="openai")
        )
        agent = Agent(config=agent_config)
        await agent.initialize()
        await engine.register_workflow(agent, WorkflowConfig.model_validate(
            {"name": "test", "steps": []},
            context={"is_initialization": True}
        ))

@pytest.mark.asyncio
async def test_engine_initialization(workflow_config: WorkflowConfig):
    """Test workflow engine initialization."""
    engine = WorkflowEngine()
    await engine.initialize(workflow_def=None, workflow_config=workflow_config)
    assert engine is not None
    assert engine._initialized
    assert isinstance(engine.workflows, dict)
    assert len(engine.workflows) == 0

@pytest.mark.asyncio
async def test_execute_with_invalid_retries(workflow_config: WorkflowConfig):
    """Test workflow execution with invalid max retries."""
    engine = WorkflowEngine()
    await engine.initialize(workflow_def=None, workflow_config=workflow_config)
    
    # Create and register an agent with the workflow
    agent_config = AgentConfig(
        name="test_agent",
        type="generic",
        model=ModelConfig(name="gpt-4", provider="openai")
    )
    agent = Agent(config=agent_config)
    await agent.initialize()
    await engine.register_workflow(agent, workflow_config)
    
    # Create a new RetryPolicy with invalid max_retries
    with pytest.raises(ValidationError):
        workflow_config.error_policy.retry_policy = RetryPolicy(max_retries=-1)

@pytest.mark.asyncio
async def test_execute_basic_workflow(workflow_config: WorkflowConfig):
    """Test basic workflow execution."""
    engine = WorkflowEngine()
    await engine.initialize(workflow_def=None, workflow_config=workflow_config)
    with pytest.raises(WorkflowExecutionError, match="No workflow registered for agent test"):
        await engine.execute_workflow("test", {"data": "test"})
