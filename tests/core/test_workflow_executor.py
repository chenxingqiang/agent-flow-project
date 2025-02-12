"""Test workflow executor module."""

import pytest
import asyncio
import numpy as np
import uuid
from typing import Dict, Any
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig, ErrorPolicy, RetryPolicy
from agentflow.core.exceptions import WorkflowExecutionError
from agentflow.core.workflow_executor import WorkflowExecutor

@pytest.mark.asyncio
async def test_workflow_execution():
    """Test basic workflow execution."""
    async def custom_transform(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Custom transform function.
        
        Args:
            step: The workflow step being executed
            context: The execution context
            
        Returns:
            Dict containing the transformed data
        """
        await asyncio.sleep(0.1)
        return {"data": context["data"]}

    config = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="test_workflow",
        max_iterations=5,
        timeout=30,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=1,
            retry_policy=RetryPolicy(
                max_retries=3,
                retry_delay=1.0,
                backoff=2.0,
                max_delay=60.0
            )
        ),
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Test transformation step",
                config=StepConfig(
                    strategy="custom",
                    params={"execute": custom_transform}
                )
            )
        ]
    )
    executor = WorkflowExecutor(config)
    await executor.initialize()
    data = np.random.randn(10, 2)
    result = await executor.execute({"data": data})
    assert "step-1" in result["steps"]
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_workflow_with_dependencies():
    """Test workflow execution with dependencies."""
    async def transform1(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """First transform function.
        
        Args:
            step: The workflow step being executed
            context: The execution context
            
        Returns:
            Dict containing the transformed data
        """
        await asyncio.sleep(0.1)
        data = context["data"]
        return {"data": data * 2}

    async def transform2(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Second transform function.
        
        Args:
            step: The workflow step being executed
            context: The execution context
            
        Returns:
            Dict containing the transformed data
        """
        await asyncio.sleep(0.1)
        data = context["data"]
        return {"data": data + 1}

    config = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="dependency_workflow",
        max_iterations=5,
        timeout=30,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=1,
            retry_policy=RetryPolicy(
                max_retries=3,
                retry_delay=1.0,
                backoff=2.0,
                max_delay=60.0
            )
        ),
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="First transformation step",
                config=StepConfig(
                    strategy="custom",
                    params={"execute": transform1}
                )
            ),
            WorkflowStep(
                id="step-2",
                name="step_2",
                type=WorkflowStepType.TRANSFORM,
                description="Second transformation step",
                dependencies=["step-1"],
                config=StepConfig(
                    strategy="custom",
                    params={"execute": transform2}
                )
            )
        ]
    )
    executor = WorkflowExecutor(config)
    await executor.initialize()
    data = np.random.randn(10, 2)
    result = await executor.execute({"data": data})
    assert "step-1" in result["steps"]
    assert "step-2" in result["steps"]
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflow error handling."""
    async def failing_transform(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Transform function that raises an error.
        
        Args:
            step: The workflow step being executed
            context: The execution context
            
        Raises:
            ValueError: Always raises this error for testing
        """
        raise ValueError("Test error")

    config = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="error_workflow",
        max_iterations=5,
        timeout=30,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=1,
            retry_policy=RetryPolicy(
                max_retries=0,  # Disable retries for this test
                retry_delay=1.0,
                backoff=2.0,
                max_delay=60.0
            )
        ),
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Failing transformation step",
                config=StepConfig(
                    strategy="custom",
                    params={"execute": failing_transform}
                )
            )
        ]
    )
    executor = WorkflowExecutor(config)
    await executor.initialize()
    data = np.random.randn(10, 2)
    
    # Explicitly raise WorkflowExecutionError
    with pytest.raises(WorkflowExecutionError) as exc_info:
        await executor.execute({"data": data})
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_workflow_timeout():
    """Test workflow timeout handling."""
    async def slow_transform(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Transform function that takes a long time.
        
        Args:
            step: The workflow step being executed
            context: The execution context
            
        Returns:
            Dict containing the transformed data
        """
        await asyncio.sleep(2)  # Longer than timeout
        return {"data": context["data"]}

    config = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="timeout_workflow",
        max_iterations=5,
        timeout=0.1,  # Very short timeout
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=1,
            retry_policy=RetryPolicy(
                max_retries=0,  # Disable retries for this test
                retry_delay=1.0,
                backoff=2.0,
                max_delay=60.0
            )
        ),
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Slow transformation step",
                config=StepConfig(
                    strategy="custom",
                    params={"execute": slow_transform}
                )
            )
        ]
    )
    executor = WorkflowExecutor(config)
    await executor.initialize()
    data = np.random.randn(10, 2)
    
    # Explicitly raise TimeoutError
    with pytest.raises(TimeoutError) as exc_info:
        await executor.execute({"data": data})
    assert "Workflow execution timed out" in str(exc_info.value)
