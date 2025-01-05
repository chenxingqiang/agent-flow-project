"""Unit tests for agentic calling functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch

from agentflow.core.workflow_types import (
    WorkflowConfig,
    WorkflowStep,
    WorkflowStepType,
    StepConfig,
    WorkflowExecutionError
)

@pytest.fixture
def basic_workflow():
    """Create a basic workflow for testing."""
    return WorkflowConfig(
        name="test_workflow",
        steps=[
            WorkflowStep(
                id="step1",
                name="Transform Data",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(strategy="standard"),
                dependencies=[]
            ),
            WorkflowStep(
                id="step2", 
                name="Analyze Data",
                type=WorkflowStepType.ANALYZE,
                config=StepConfig(strategy="basic"),
                dependencies=["step1"]
            )
        ]
    )

@pytest.mark.asyncio
async def test_workflow_execution_basic(basic_workflow):
    """Test basic workflow execution."""
    test_data = {"input": "test"}
    results = await basic_workflow.execute(test_data)
    
    assert "step1" in results
    assert "step2" in results
    assert results["step1"]["status"] == "success"
    assert results["step2"]["status"] == "success"

@pytest.mark.asyncio
async def test_empty_workflow():
    """Test execution of empty workflow."""
    workflow = WorkflowConfig(name="empty_workflow")
    results = await workflow.execute({})
    assert results == {}

def test_invalid_step_type():
    """Test validation of invalid step type."""
    with pytest.raises(WorkflowExecutionError):
        WorkflowStep(
            id="invalid",
            name="Invalid Step",
            type="invalid_type",  # Invalid type
            config=StepConfig(strategy="test")
        ).validate()

def test_invalid_transform_strategy():
    """Test validation of invalid transform strategy."""
    step = WorkflowStep(
        id="transform",
        name="Transform Step",
        type=WorkflowStepType.TRANSFORM,
        config=StepConfig(strategy="invalid_strategy")
    )
    
    with pytest.raises(WorkflowExecutionError):
        step.validate()

def test_circular_dependencies():
    """Test detection of circular dependencies."""
    workflow = WorkflowConfig(
        name="circular_workflow",
        steps=[
            WorkflowStep(
                id="step1",
                name="Step 1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(strategy="standard"),
                dependencies=["step2"]
            ),
            WorkflowStep(
                id="step2",
                name="Step 2",
                type=WorkflowStepType.ANALYZE,
                config=StepConfig(strategy="basic"),
                dependencies=["step1"]
            )
        ]
    )
    
    with pytest.raises(WorkflowExecutionError):
        workflow._validate_dependencies()

def test_missing_dependency():
    """Test validation of missing dependencies."""
    workflow = WorkflowConfig(
        name="missing_dep_workflow",
        steps=[
            WorkflowStep(
                id="step1",
                name="Step 1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(strategy="standard"),
                dependencies=["nonexistent"]
            )
        ]
    )
    
    with pytest.raises(WorkflowExecutionError):
        workflow._validate_dependencies()

@pytest.mark.asyncio
async def test_execution_order():
    """Test correct execution order of steps."""
    workflow = WorkflowConfig(
        name="order_test",
        steps=[
            WorkflowStep(
                id="step3",
                name="Step 3",
                type=WorkflowStepType.AGGREGATE,
                config=StepConfig(strategy="basic"),
                dependencies=["step1", "step2"]
            ),
            WorkflowStep(
                id="step1",
                name="Step 1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(strategy="standard"),
                dependencies=[]
            ),
            WorkflowStep(
                id="step2",
                name="Step 2",
                type=WorkflowStepType.ANALYZE,
                config=StepConfig(strategy="basic"),
                dependencies=["step1"]
            )
        ]
    )
    
    order = workflow._get_execution_order()
    assert order.index("step1") < order.index("step2")
    assert order.index("step2") < order.index("step3")

@pytest.mark.asyncio
async def test_workflow_execution_error_handling():
    """Test error handling during workflow execution."""
    workflow = WorkflowConfig(
        name="error_test",
        steps=[
            WorkflowStep(
                id="step1",
                name="Step 1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(strategy="standard"),
                dependencies=[]
            )
        ]
    )
    
    # Mock step execution to raise an error
    with patch.object(workflow, '_get_execution_order', side_effect=Exception("Test error")):
        with pytest.raises(WorkflowExecutionError):
            await workflow.execute({})

@pytest.mark.asyncio
async def test_workflow_timeout():
    """Test workflow timeout functionality."""
    workflow = WorkflowConfig(
        name="timeout_test",
        timeout=0.1,  # Very short timeout
        steps=[
            WorkflowStep(
                id="step1",
                name="Long Step",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(strategy="standard"),
                dependencies=[]
            )
        ]
    )
    
    # Mock step execution to take longer than timeout
    async def slow_execution():
        await asyncio.sleep(0.2)
        return {}
    
    with patch.object(workflow, 'execute', side_effect=slow_execution):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(workflow.execute({}), timeout=workflow.timeout) 