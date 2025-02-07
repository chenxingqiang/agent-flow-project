"""Test workflow templates."""

import pytest
import uuid
import numpy as np
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig
from agentflow.core.workflow_executor import WorkflowExecutor

@pytest.mark.asyncio
async def test_feature_engineering_template():
    """Test feature engineering workflow template."""
    workflow = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="feature_engineering_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Feature engineering step",
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
    executor = WorkflowExecutor(workflow)
    await executor.initialize()
    data = np.random.randn(10, 2)
    result = await executor.execute({"data": data})
    assert result["status"] == "completed"
    assert "step-1" in result["steps"]
    assert result["steps"]["step-1"]["status"] == "completed"

@pytest.mark.asyncio
async def test_outlier_removal_template():
    """Test outlier removal workflow template."""
    workflow = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="outlier_removal_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Outlier removal step",
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
    executor = WorkflowExecutor(workflow)
    await executor.initialize()
    data = np.random.randn(100, 2)
    result = await executor.execute({"data": data})
    assert result["status"] == "completed"
    assert "step-1" in result["steps"]
    assert result["steps"]["step-1"]["status"] == "completed"

@pytest.mark.asyncio
async def test_combined_template():
    """Test combined workflow template."""
    workflow = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="combined_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Feature engineering step",
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
                description="Outlier removal step",
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
    executor = WorkflowExecutor(workflow)
    await executor.initialize()
    data = np.random.randn(100, 2)
    result = await executor.execute({"data": data})
    assert result["status"] == "completed"
    assert "step-1" in result["steps"]
    assert "step-2" in result["steps"]
    assert result["steps"]["step-1"]["status"] == "completed"
    assert result["steps"]["step-2"]["status"] == "completed"
