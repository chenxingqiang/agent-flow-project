"""Test workflow types."""

import pytest
from typing import Dict, Any
from pydantic import ValidationError
from agentflow.core.workflow_types import (
    AgentConfig,
    WorkflowConfig,
    WorkflowStep,
    ModelConfig,
    RetryPolicy,
    ErrorPolicy,
    WorkflowStepType,
    StepConfig
)

@pytest.fixture
def test_workflow_def() -> Dict[str, Any]:
    """Test workflow definition."""
    return {
        "id": "test_workflow",
        "name": "test",
        "steps": [
            {
                "id": "step1",
                "name": "Step 1",
                "type": WorkflowStepType.TRANSFORM,
                "config": {
                    "strategy": "standard",
                    "params": {"key": "value"}
                }
            }
        ]
    }

@pytest.fixture
def test_agent_config() -> Dict[str, Any]:
    """Test agent configuration."""
    return {
        "name": "test_agent",
        "type": "research",
        "model": {
            "name": "gpt-4",
            "provider": "openai",
            "temperature": 0.5
        },
        "workflow": {
            "id": "test_workflow",
            "name": "test",
            "steps": [
                {
                    "id": "step1",
                    "name": "Step 1",
                    "type": WorkflowStepType.TRANSFORM,
                    "config": {
                        "strategy": "standard",
                        "params": {"key": "value"}
                    }
                }
            ]
        }
    }

def test_agent_config_basic(test_agent_config):
    """Test basic AgentConfig initialization and validation."""
    config = AgentConfig(**test_agent_config)
    assert config.name == "test_agent"
    assert config.type == "research"
    assert config.distributed is False
    assert isinstance(config.id, str)

def test_agent_config_with_workflow(test_workflow_def):
    """Test AgentConfig with workflow configuration."""
    config = AgentConfig(name="test_agent", workflow=test_workflow_def)
    assert isinstance(config.workflow, dict)
    assert config.workflow["id"] == "test_workflow"
    assert config.workflow["name"] == "test"

def test_agent_config_with_workflow_config(test_workflow_def):
    """Test AgentConfig with WorkflowConfig instance."""
    workflow = WorkflowConfig(**test_workflow_def)
    config = AgentConfig(name="test_agent", workflow=workflow.model_dump())
    assert isinstance(config.workflow, dict)
    assert config.workflow["id"] == test_workflow_def["id"]
    assert config.workflow["name"] == test_workflow_def["name"]

def test_agent_config_invalid_type():
    """Test AgentConfig with invalid type."""
    with pytest.raises(ValidationError):
        AgentConfig(name="test_agent", type="invalid_type")

def test_agent_config_model():
    """Test AgentConfig with model configuration."""
    model = ModelConfig(name="gpt-4", provider="openai", temperature=0.5)
    config = AgentConfig(name="test_agent", model=model)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.model, ModelConfig) and config.model.name == "gpt-4"

def test_agent_config_default_model():
    """Test AgentConfig default model configuration."""
    config = AgentConfig(name="test_agent")
    assert config.model is not None
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.model, ModelConfig) and config.model.name == "gpt-3.5-turbo"
    assert isinstance(config.model, ModelConfig) and config.model.provider == "openai"

def test_agent_config_dict_access():
    """Test AgentConfig dict-like access."""
    config = AgentConfig(name="test_agent")
    assert config["name"] == "test_agent"
    assert config.get("invalid_key", "default") == "default"
    assert "name" in config
    assert "invalid_key" not in config

def test_agent_config_is_distributed():
    """Test AgentConfig is_distributed property."""
    config = AgentConfig(name="test_agent", distributed=True)
    assert config.is_distributed is True

    workflow_config = {
        "id": "test-workflow",
        "name": "test",
        "distributed": True,
        "steps": [
            {
                "id": "step1",
                "name": "Step 1",
                "type": WorkflowStepType.TRANSFORM,
                "config": {
                    "strategy": "standard",
                    "params": {"key": "value"}
                }
            }
        ]
    }
    config = AgentConfig(name="test_agent", workflow=workflow_config)
    assert config.is_distributed is True

    workflow_config["distributed"] = False
    config = AgentConfig(name="test_agent", distributed=False, workflow=workflow_config)
    assert config.is_distributed is False

def test_workflow_step_validation():
    """Test WorkflowStep validation."""
    step = WorkflowStep(
        id="test_step",
        name="Test Step",
        type=WorkflowStepType.TRANSFORM,
        config=StepConfig(
            strategy="standard",
            params={"key": "value"}
        )
    )
    assert step.id == "test_step"
    assert step.name == "Test Step"
    assert step.type == WorkflowStepType.TRANSFORM

def test_workflow_config_validation():
    """Test WorkflowConfig validation."""
    config = WorkflowConfig(
        id="test_workflow",
        name="Test Workflow",
        steps=[
            WorkflowStep(
                id="step1",
                name="Step 1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="standard",
                    params={"key": "value"}
                )
            )
        ]
    )
    assert config.id == "test_workflow"
    assert config.name == "Test Workflow"
    assert len(config.steps) == 1

def test_workflow_step_invalid_type():
    """Test WorkflowStep with invalid type."""
    with pytest.raises(ValidationError):
        WorkflowStep(
            id="test_step",
            name="Test Step",
            type="invalid_type",  # type: ignore
            config=StepConfig(
                strategy="standard",
                params={"key": "value"}
            )
        )

def test_workflow_step_missing_name():
    """Test WorkflowStep with missing name."""
    with pytest.raises(ValidationError):
        WorkflowStep(
            id="test_step",
            name=None,  # type: ignore
            type=WorkflowStepType.TRANSFORM,
            config=StepConfig(
                strategy="standard",
                params={"key": "value"}
            )
        )

def test_workflow_config_empty_steps():
    """Test WorkflowConfig with empty steps list."""
    with pytest.raises(ValueError):
        WorkflowConfig(
            id="test_workflow",
            name="Test Workflow",
            steps=[]
        )

    config = WorkflowConfig(
        id="test_workflow",
        name="Test Workflow",
        steps=[
            WorkflowStep(
                id="step1",
                name="Step 1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="standard",
                    params={"key": "value"}
                )
            )
        ]
    )
    
    # Test that setting empty steps raises ValueError
    with pytest.raises(ValueError):
        config.steps = []

def test_workflow_config_get_step():
    """Test WorkflowConfig get_step method."""
    step = WorkflowStep(
        id="step1",
        name="Step 1",
        type=WorkflowStepType.TRANSFORM,
        config=StepConfig(
            strategy="standard",
            params={"key": "value"}
        )
    )
    config = WorkflowConfig(
        id="test_workflow",
        name="Test Workflow",
        steps=[step]
    )
    assert config.get_step("step1") == step
    assert config.get_step("nonexistent") is None

def test_retry_policy_validation():
    """Test RetryPolicy validation."""
    policy = RetryPolicy(
        retry_delay=1.0,
        backoff=2.0,
        max_retries=3
    )
    assert policy.retry_delay == 1.0
    assert policy.backoff == 2.0
    assert policy.max_retries == 3

def test_error_policy_validation():
    """Test ErrorPolicy validation."""
    policy = ErrorPolicy(
        fail_fast=True,
        ignore_warnings=False,
        max_errors=10
    )
    assert policy.fail_fast is True
    assert policy.ignore_warnings is False
    assert policy.max_errors == 10

def test_workflow_config_distributed_steps():
    """Test WorkflowConfig distributed steps."""
    step1 = WorkflowStep(
        id="step1",
        name="Step 1",
        type=WorkflowStepType.TRANSFORM,
        config=StepConfig(
            strategy="standard",
            params={"key": "value"}
        ),
        is_distributed=True
    )
    step2 = WorkflowStep(
        id="step2",
        name="Step 2",
        type=WorkflowStepType.ANALYZE,
        config=StepConfig(
            strategy="standard",
            params={"key": "value"}
        ),
        is_distributed=True
    )
    config = WorkflowConfig(
        id="test_workflow",
        name="Test Workflow",
        steps=[step1, step2]
    )
    assert len([step for step in config.steps if step.is_distributed]) == 2
