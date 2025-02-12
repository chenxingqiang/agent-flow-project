"""Test workflow performance."""

import pytest
import time
from typing import Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig

async def feature_engineering_transform(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
    """Feature engineering transform function.
    
    Args:
        step: The workflow step being executed
        context: The execution context containing the data
        
    Returns:
        Dict containing the transformed data
    """
    data = context["data"]
    scaler = StandardScaler(
        with_mean=step.config.params["with_mean"],
        with_std=step.config.params["with_std"]
    )
    transformed_data = scaler.fit_transform(data)
    return {"data": transformed_data}

async def outlier_removal_transform(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
    """Outlier removal transform function.
    
    Args:
        step: The workflow step being executed
        context: The execution context containing the data
        
    Returns:
        Dict containing the transformed data
    """
    data = context["data"]
    iso_forest = IsolationForest(
        contamination=step.config.params["threshold"],
        random_state=42
    )
    predictions = iso_forest.fit_predict(data)
    # Keep only non-outlier points (predictions == 1)
    filtered_data = data[predictions == 1]
    return {"data": filtered_data}

@pytest.mark.asyncio
async def test_workflow_execution_time():
    """Test workflow execution time."""
    workflow = WorkflowConfig(
        id="test-workflow-1",
        name="performance_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Feature engineering step for performance testing",
                config=StepConfig(
                    strategy="feature_engineering",
                    params={
                        "method": "standard",
                        "with_mean": True,
                        "with_std": True,
                        "execute": feature_engineering_transform
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
    assert "step-1" in result["steps"]  # Check that step-1 is in steps
    assert result["status"] == "success"  # Verify overall workflow status

@pytest.mark.asyncio
async def test_workflow_memory_usage():
    """Test workflow memory usage."""
    workflow = WorkflowConfig(
        id="test-workflow-2",
        name="memory_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Feature engineering step for memory testing",
                config=StepConfig(
                    strategy="feature_engineering",
                    params={
                        "method": "standard",
                        "with_mean": True,
                        "with_std": True,
                        "execute": feature_engineering_transform
                    }
                )
            ),
            WorkflowStep(
                id="step-2",
                name="step_2",
                type=WorkflowStepType.TRANSFORM,
                description="Outlier removal step for memory testing",
                dependencies=["step-1"],
                config=StepConfig(
                    strategy="outlier_removal",
                    params={
                        "method": "isolation_forest",
                        "threshold": 0.1,
                        "execute": outlier_removal_transform
                    }
                )
            )
        ]
    )
    data = np.random.randn(10000, 10)  # Very large dataset
    result = await workflow.execute({"data": data})
    assert "step-1" in result["steps"]  # Check that step-1 is in steps
    assert result["status"] == "success"  # Verify overall workflow status

@pytest.mark.asyncio
async def test_workflow_parallel_execution():
    """Test parallel workflow execution."""
    workflow = WorkflowConfig(
        id="test-workflow-3",
        name="parallel_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Feature engineering step for parallel testing",
                config=StepConfig(
                    strategy="feature_engineering",
                    params={
                        "method": "standard",
                        "with_mean": True,
                        "with_std": True,
                        "execute": feature_engineering_transform
                    }
                )
            ),
            WorkflowStep(
                id="step-2",
                name="step_2",
                type=WorkflowStepType.TRANSFORM,
                description="Outlier removal step for parallel testing",
                config=StepConfig(
                    strategy="outlier_removal",
                    params={
                        "method": "isolation_forest",
                        "threshold": 0.1,
                        "execute": outlier_removal_transform
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
    assert "step-1" in result["steps"]  # Check that step-1 is in steps
    assert result["status"] == "success"  # Verify overall workflow status