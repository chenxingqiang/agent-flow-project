"""Test agent run module."""

import pytest
import numpy as np
from agentflow.core.config import AgentConfig, ConfigurationType, AgentMode, ModelConfig
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig
from agentflow.core.exceptions import WorkflowExecutionError

@pytest.mark.asyncio
async def test_agent_run():
    """Test agent run functionality."""
    workflow = WorkflowConfig(
        name="test_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="feature_engineering",
                    params={
                        "method": "standard",
                        "with_mean": True,
                        "with_std": True
                    }
                )
            )
        ]
    )
    agent = AgentConfig(
        id="test-agent-1",
        name="test_agent",
        type=ConfigurationType.DATA_SCIENCE,
        mode=AgentMode.SEQUENTIAL,
        version="1.0.0",
        model_settings=ModelConfig(
            provider="openai",
            name="gpt-4",
            temperature=0.7,
            max_tokens=4096
        ),
        workflow=workflow
    )
    data = np.random.randn(10, 2)
    result = await agent.workflow.execute({"data": data})
    assert "step-1" in result
    assert "data" in result["step-1"]
    assert isinstance(result["step-1"]["data"], np.ndarray)
    assert result["step-1"]["data"].shape == data.shape

@pytest.mark.asyncio
async def test_agent_run_with_dependencies():
    """Test agent run with workflow dependencies."""
    workflow = WorkflowConfig(
        name="dependency_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="feature_engineering",
                    params={
                        "method": "standard",
                        "with_mean": True,
                        "with_std": True
                    }
                )
            ),
            WorkflowStep(
                id="step-2",
                name="step_2",
                type=WorkflowStepType.TRANSFORM,
                dependencies=["step-1"],
                config=StepConfig(
                    strategy="outlier_removal",
                    params={
                        "method": "isolation_forest",
                        "threshold": 0.1
                    }
                )
            )
        ]
    )
    agent = AgentConfig(
        id="test-agent-2",
        name="test_agent",
        type=ConfigurationType.DATA_SCIENCE,
        mode=AgentMode.SEQUENTIAL,
        version="1.0.0",
        model_settings=ModelConfig(
            provider="openai",
            name="gpt-4",
            temperature=0.7,
            max_tokens=4096
        ),
        workflow=workflow
    )
    data = np.random.randn(100, 2)  # More data points for outlier detection
    result = await agent.workflow.execute({"data": data})
    assert "step-1" in result
    assert "step-2" in result
    assert "data" in result["step-2"]
    assert isinstance(result["step-2"]["data"], np.ndarray)
    assert result["step-2"]["data"].shape[1] == data.shape[1]  # Same number of features
    assert result["step-2"]["data"].shape[0] <= data.shape[0]  # Some points may be removed as outliers

@pytest.mark.asyncio
async def test_agent_run_error_handling():
    """Test agent run error handling."""
    workflow = WorkflowConfig(
        name="error_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="invalid_strategy",
                    params={}
                )
            )
        ]
    )
    agent = AgentConfig(
        id="test-agent-3",
        name="test_agent",
        type=ConfigurationType.DATA_SCIENCE,
        mode=AgentMode.SEQUENTIAL,
        version="1.0.0",
        model_settings=ModelConfig(
            provider="openai",
            name="gpt-4",
            temperature=0.7,
            max_tokens=4096
        ),
        workflow=workflow
    )
    data = np.random.randn(10, 2)
    with pytest.raises(WorkflowExecutionError):
        await agent.workflow.execute({"data": data})
