"""Test workflow templates."""

import pytest
import numpy as np
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig

@pytest.mark.asyncio
async def test_feature_engineering_template():
    """Test feature engineering workflow template."""
    workflow = WorkflowConfig(
        name="feature_engineering_workflow",
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
    data = np.random.randn(10, 2)
    result = await workflow.execute({"data": data})
    assert "step-1" in result
    assert "output" in result["step-1"]
    assert "data" in result["step-1"]["output"]
    assert isinstance(result["step-1"]["output"]["data"], np.ndarray)
    assert result["step-1"]["output"]["data"].shape == data.shape

@pytest.mark.asyncio
async def test_outlier_removal_template():
    """Test outlier removal workflow template."""
    workflow = WorkflowConfig(
        name="outlier_removal_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
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
    data = np.random.randn(100, 2)  # More data points for outlier detection
    result = await workflow.execute({"data": data})
    assert "step-1" in result
    assert "output" in result["step-1"]
    assert "data" in result["step-1"]["output"]
    assert isinstance(result["step-1"]["output"]["data"], np.ndarray)
    assert result["step-1"]["output"]["data"].shape[1] == data.shape[1]  # Same number of features
    assert result["step-1"]["output"]["data"].shape[0] <= data.shape[0]  # Some points may be removed as outliers

@pytest.mark.asyncio
async def test_combined_template():
    """Test combined workflow template."""
    workflow = WorkflowConfig(
        name="combined_workflow",
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
    data = np.random.randn(100, 2)  # More data points for outlier detection
    result = await workflow.execute({"data": data})
    assert "step-1" in result
    assert "step-2" in result
    assert "output" in result["step-2"]
    assert "data" in result["step-2"]["output"]
    assert isinstance(result["step-2"]["output"]["data"], np.ndarray)
    assert result["step-2"]["output"]["data"].shape[1] == data.shape[1]  # Same number of features
    assert result["step-2"]["output"]["data"].shape[0] <= data.shape[0]  # Some points may be removed as outliers
