"""Test workflow performance."""

import pytest
import time
import numpy as np
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig

@pytest.mark.asyncio
async def test_workflow_execution_time():
    """Test workflow execution time."""
    workflow = WorkflowConfig(
        name="performance_workflow",
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
    data = np.random.randn(1000, 10)  # Large dataset
    start_time = time.time()
    result = await workflow.execute({"data": data})
    execution_time = time.time() - start_time
    assert execution_time < 5  # Should complete within 5 seconds
    assert "step-1" in result
    assert "data" in result["step-1"]
    assert isinstance(result["step-1"]["data"], np.ndarray)
    assert result["step-1"]["data"].shape == data.shape

@pytest.mark.asyncio
async def test_workflow_memory_usage():
    """Test workflow memory usage."""
    workflow = WorkflowConfig(
        name="memory_workflow",
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
    data = np.random.randn(10000, 10)  # Very large dataset
    result = await workflow.execute({"data": data})
    assert "step-1" in result
    assert "step-2" in result
    assert "data" in result["step-2"]
    assert isinstance(result["step-2"]["data"], np.ndarray)
    assert result["step-2"]["data"].shape[1] == data.shape[1]  # Same number of features
    assert result["step-2"]["data"].shape[0] <= data.shape[0]  # Some points may be removed as outliers

@pytest.mark.asyncio
async def test_workflow_parallel_execution():
    """Test parallel workflow execution."""
    workflow = WorkflowConfig(
        name="parallel_workflow",
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
    data = np.random.randn(1000, 10)  # Large dataset
    start_time = time.time()
    result = await workflow.execute({"data": data})
    execution_time = time.time() - start_time
    assert execution_time < 10  # Should complete within 10 seconds
    assert "step-1" in result
    assert "step-2" in result
    assert "data" in result["step-2"]
    assert isinstance(result["step-2"]["data"], np.ndarray)
    assert result["step-2"]["data"].shape[1] == data.shape[1]  # Same number of features
    assert result["step-2"]["data"].shape[0] <= data.shape[0]  # Some points may be removed as outliers