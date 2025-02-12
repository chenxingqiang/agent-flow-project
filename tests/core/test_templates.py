"""Test workflow templates."""

import pytest
import uuid
import numpy as np
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig
from agentflow.core.workflow_executor import WorkflowExecutor
from agentflow.core.processors.transformers import TransformProcessor

async def create_transform_function(strategy: str, params: dict):
    """Create a transform function for the given strategy."""
    processor = TransformProcessor(StepConfig(strategy=strategy, params=params))
    
    async def transform_fn(step: WorkflowStep, context: dict) -> dict:
        result = await processor.process(context)
        return {"data": result.data, "metadata": result.metadata}
    
    return transform_fn

@pytest.mark.asyncio
async def test_feature_engineering_template():
    """Test feature engineering workflow template."""
    transform_fn = await create_transform_function(
        strategy="feature_engineering",
        params={
            "method": "standard",
            "with_mean": True,
            "with_std": True
        }
    )
    
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
                        "with_std": True,
                        "execute": transform_fn
                    }
                )
            )
        ]
    )
    executor = WorkflowExecutor(workflow)
    await executor.initialize()
    data = np.random.randn(10, 2)
    result = await executor.execute({"data": data})
    assert result is not None
    assert "steps" in result
    assert "step-1" in result["steps"]
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_outlier_removal_template():
    """Test outlier removal workflow template."""
    transform_fn = await create_transform_function(
        strategy="outlier_removal",
        params={
            "method": "isolation_forest",
            "threshold": 0.1
        }
    )
    
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
                        "threshold": 0.1,
                        "execute": transform_fn
                    }
                )
            )
        ]
    )
    executor = WorkflowExecutor(workflow)
    await executor.initialize()
    data = np.random.randn(100, 2)
    result = await executor.execute({"data": data})
    assert result is not None
    assert "steps" in result
    assert "step-1" in result["steps"]
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_combined_template():
    """Test combined workflow template."""
    feature_eng_fn = await create_transform_function(
        strategy="feature_engineering",
        params={
            "method": "standard",
            "with_mean": True,
            "with_std": True
        }
    )
    
    outlier_removal_fn = await create_transform_function(
        strategy="outlier_removal",
        params={
            "method": "isolation_forest",
            "threshold": 0.1
        }
    )
    
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
                        "with_std": True,
                        "execute": feature_eng_fn
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
                        "threshold": 0.1,
                        "execute": outlier_removal_fn
                    }
                )
            )
        ]
    )
    executor = WorkflowExecutor(workflow)
    await executor.initialize()
    data = np.random.randn(100, 2)
    result = await executor.execute({"data": data})
    assert result is not None
    assert "steps" in result
    assert "step-1" in result["steps"]
    assert "step-2" in result["steps"]
    assert result["status"] == "success"
