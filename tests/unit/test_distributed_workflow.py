"""Test distributed workflow."""

import pytest
import ray
from typing import Dict, Any
import asyncio

from agentflow.core.research_workflow import ResearchDistributedWorkflow
from agentflow.core.workflow_types import WorkflowStepType
from agentflow.core.exceptions import WorkflowExecutionError

@pytest.fixture
def setup_ray():
    """Setup Ray."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield None
    if ray.is_initialized():
        ray.shutdown()

@pytest.fixture
def test_workflow_def() -> Dict[str, Any]:
    """Test workflow definition."""
    return {
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "HIERARCHICAL"
            },
            "WORKFLOW": {
                "step_1": {
                    "id": "step_1",
                    "name": "First test step",
                    "description": "First test step",
                    "input": ["test_input"],
                    "output": ["test_output"],
                    "agent_config": {},
                    "type": "research_execution"
                },
                "step_2": {
                    "id": "step_2",
                    "name": "Second test step",
                    "description": "Second test step",
                    "input": ["test_input"],
                    "output": ["test_output"],
                    "agent_config": {},
                    "dependencies": ["step_1"],
                    "type": "research_execution"
                }
            }
        }
    }

@pytest.fixture
def workflow_config() -> Dict[str, Any]:
    """Test configuration."""
    return {
        "max_retries": 3,
        "timeout": 3600,
        "max_iterations": 5,
        "logging_level": "INFO",
        "required_fields": [],
        "error_handling": {},
        "retry_policy": {
            "max_retries": 3,
            "retry_delay": 0.1,
            "retry_backoff": 1.5
        },
        "error_policy": None,
        "is_distributed": True,
        "distributed": True,
        "steps": [],
        "metadata": {},
        "agents": {}
    }

@pytest.mark.asyncio
async def test_distributed_workflow_execution(setup_ray, test_workflow_def, workflow_config):
    """Test distributed workflow execution."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Execute workflow
    input_data = {"test_input": "test_value"}
    result = await workflow.execute_async.remote(input_data)
    assert result["status"] == "success"
    assert "results" in result
    assert "step_1" in result["results"]
    assert "step_2" in result["results"]

@pytest.mark.asyncio
async def test_distributed_workflow_error_handling(setup_ray, test_workflow_def, workflow_config):
    """Test distributed workflow error handling."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Test with invalid input
    with pytest.raises(ray.exceptions.RayTaskError) as exc_info:
        await workflow.execute_async.remote("invalid_input")
    assert "dict" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_distributed_workflow_retry_mechanism(setup_ray, test_workflow_def, workflow_config):
    """Test distributed workflow retry mechanism."""
    # Modify config to test retry
    workflow_config["step_1_config"]["max_retries"] = 2
    workflow_config["step_1_config"]["retry_delay"] = 0.1
    
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Mock a failing input that should trigger retries
    failing_input = {
        "test_input": "fail_on_first_try",
        "_test_fail_count": 1  # This will make the first attempt fail
    }
    
    # Execute workflow - should succeed after retry
    result = await workflow.execute_async.remote(failing_input)
    assert result["status"] == "success"
    assert result["results"]["step_1"]["retry_count"] == 1

@pytest.mark.asyncio
async def test_distributed_workflow_step_dependencies(setup_ray, test_workflow_def, workflow_config):
    """Test distributed workflow step dependencies."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Execute workflow
    input_data = {"test_input": "test_value"}
    result = await workflow.execute_async.remote(input_data)
    
    # Verify step execution order through timestamps
    step1_time = result["results"]["step_1"]["metadata"]["timestamp"]
    step2_time = result["results"]["step_2"]["metadata"]["timestamp"]
    assert step1_time < step2_time, "Step 2 executed before its dependency (Step 1)"

@pytest.mark.asyncio
async def test_distributed_workflow_parallel_execution(setup_ray, test_workflow_def, workflow_config):
    """Test parallel execution of independent steps."""
    # Modify workflow to have parallel steps
    test_workflow_def["WORKFLOW"]["step_3"] = {
        "step": 3,
        "name": "Test Step 3",
        "description": "Parallel step",
        "input": ["test_input"],
        "type": WorkflowStepType.RESEARCH_EXECUTION,
        "agent_config": {}
    }
    
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Execute workflow
    input_data = {"test_input": "test_value"}
    result = await workflow.execute_async.remote(input_data)
    
    # Verify parallel execution
    assert result["status"] == "success"
    assert "step_3" in result["results"]
    step1_time = result["results"]["step_1"]["metadata"]["timestamp"]
    step3_time = result["results"]["step_3"]["metadata"]["timestamp"]
    time_diff = abs(step3_time - step1_time)
    assert time_diff < 0.5, "Independent steps were not executed in parallel"
