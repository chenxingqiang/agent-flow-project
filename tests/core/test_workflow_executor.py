"""Test workflow executor module."""

import pytest
import asyncio
import numpy as np
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig
from agentflow.core.exceptions import WorkflowExecutionError
from agentflow.core.workflow_executor import WorkflowExecutor

@pytest.mark.asyncio
async def test_workflow_execution():
    """Test basic workflow execution."""
    async def custom_transform(data):
        await asyncio.sleep(0.1)
        return data

    config = WorkflowConfig(
        name="test_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"execute": custom_transform}
                )
            )
        ]
    )
    executor = WorkflowExecutor(config)
    data = np.random.randn(10, 2)
    result = await executor.execute({"data": data})
    assert "step-1" in result
    assert "data" in result["step-1"]
    assert isinstance(result["step-1"]["data"], np.ndarray)
    assert result["step-1"]["data"].shape == data.shape

@pytest.mark.asyncio
async def test_workflow_with_dependencies():
    """Test workflow execution with dependencies."""
    async def transform1(data):
        await asyncio.sleep(0.1)
        return data * 2

    async def transform2(data):
        await asyncio.sleep(0.1)
        return data + 1

    config = WorkflowConfig(
        name="dependency_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"execute": transform1}
                )
            ),
            WorkflowStep(
                id="step-2",
                name="step_2",
                type=WorkflowStepType.TRANSFORM,
                dependencies=["step-1"],
                config=StepConfig(
                    strategy="custom",
                    params={"execute": transform2}
                )
            )
        ]
    )
    executor = WorkflowExecutor(config)
    data = np.random.randn(10, 2)
    result = await executor.execute({"data": data})
    assert "step-1" in result
    assert "step-2" in result
    assert "data" in result["step-1"]
    assert "data" in result["step-2"]
    assert isinstance(result["step-1"]["data"], np.ndarray)
    assert isinstance(result["step-2"]["data"], np.ndarray)
    assert result["step-1"]["data"].shape == data.shape
    assert result["step-2"]["data"].shape == data.shape
    # Check that step-2 data is step-1 data + 1
    np.testing.assert_array_almost_equal(result["step-2"]["data"], result["step-1"]["data"] + 1)

@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflow error handling."""
    async def failing_transform(data):
        raise ValueError("Test error")

    config = WorkflowConfig(
        name="error_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"execute": failing_transform}
                )
            )
        ]
    )
    executor = WorkflowExecutor(config)
    data = np.random.randn(10, 2)
    with pytest.raises(WorkflowExecutionError):
        await executor.execute({"data": data})

@pytest.mark.asyncio
async def test_workflow_timeout():
    """Test workflow timeout handling."""
    async def slow_transform(data):
        await asyncio.sleep(2)  # Longer than timeout
        return data

    config = WorkflowConfig(
        name="timeout_workflow",
        max_iterations=5,
        timeout=1,  # Short timeout
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"execute": slow_transform}
                )
            )
        ]
    )
    executor = WorkflowExecutor(config)
    data = np.random.randn(10, 2)
    with pytest.raises(WorkflowExecutionError):
        await executor.execute({"data": data})
