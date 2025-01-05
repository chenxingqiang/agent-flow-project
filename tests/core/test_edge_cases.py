"""Test edge cases for workflow execution."""

import pytest
import numpy as np
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig
from agentflow.core.exceptions import WorkflowExecutionError

@pytest.mark.asyncio
async def test_empty_workflow():
    """Test workflow with no steps."""
    workflow = WorkflowConfig(
        name="empty_workflow",
        max_iterations=5,
        timeout=30,
        steps=[]
    )
    data = np.random.randn(10, 2)
    result = await workflow.execute({"data": data})
    assert isinstance(result, dict)
    assert len(result) == 0  # No steps, so no results

@pytest.mark.asyncio
async def test_invalid_step_type():
    """Test workflow with invalid step type."""
    workflow = WorkflowConfig(
        name="invalid_workflow",
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
    data = np.random.randn(10, 2)
    with pytest.raises(WorkflowExecutionError):
        await workflow.execute({"data": data})

@pytest.mark.asyncio
async def test_circular_dependencies():
    """Test workflow with circular dependencies."""
    workflow = WorkflowConfig(
        name="circular_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                dependencies=["step-2"],
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
    data = np.random.randn(10, 2)
    with pytest.raises(WorkflowExecutionError):
        await workflow.execute({"data": data})

@pytest.mark.asyncio
async def test_missing_dependencies():
    """Test workflow with missing dependencies."""
    workflow = WorkflowConfig(
        name="missing_deps_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                dependencies=["nonexistent-step"],
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
    with pytest.raises(WorkflowExecutionError):
        await workflow.execute({"data": data})
