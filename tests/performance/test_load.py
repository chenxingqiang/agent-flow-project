"""Test load performance."""

import pytest
import ray
import time
from typing import Dict, Any
import asyncio

from agentflow.core.research_workflow import ResearchDistributedWorkflow
from agentflow.core.workflow_types import WorkflowStepType

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
        "INPUT": ["test_input"],
        "OUTPUT": ["test_output"],
        "WORKFLOW": {
            "step_1": {
                "step": 1,
                "name": "Research Step",
                "description": "Research test step",
                "input": ["test_input"],
                "type": WorkflowStepType.RESEARCH_EXECUTION,
                "agent_config": {}
            },
            "step_2": {
                "step": 2,
                "name": "Document Step",
                "description": "Document test step",
                "input": ["test_input"],
                "dependencies": ["step_1"],
                "type": WorkflowStepType.DOCUMENT_GENERATION,
                "agent_config": {}
            }
        }
    }

@pytest.fixture
def workflow_config() -> Dict[str, Any]:
    """Test configuration."""
    return {
        "id": "test-workflow-1",
        "name": "Test Workflow",
        "max_iterations": 5,
        "timeout": 300,
        "steps": [
            {
                "id": "step-1",
                "name": "Research Step",
                "type": WorkflowStepType.RESEARCH_EXECUTION,
                "description": "Execute research analysis step",
                "config": {
                    "strategy": "standard",
                    "params": {
                        "research_topic": "AI Ethics",
                        "depth": "comprehensive"
                    }
                }
            }
        ],
        "error_policy": {
            "fail_fast": True,
            "ignore_warnings": False,
            "max_errors": 10,
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff": 2.0,
                "max_delay": 60.0
            }
        }
    }

@pytest.mark.asyncio
async def test_workflow_load(setup_ray, test_workflow_def, workflow_config):
    """Test workflow under load."""
    # Create multiple workflow instances
    workflows = []
    num_workflows = 5
    
    for _ in range(num_workflows):
        workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
        assert isinstance(workflow, ray.actor.ActorHandle)
        workflows.append(workflow)
    
    # Execute workflows in parallel
    start_time = time.time()
    input_data = {"test_input": "test_value"}
    
    result_refs = [workflow.execute_async.remote(input_data) for workflow in workflows]
    results = []
    for ref in result_refs:
        result = ray.get(ref)
        results.append(result)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Verify results
    for result in results:
        assert result["status"] == "success"
        assert "results" in result
    
    # Check performance metrics
    assert execution_time < 30  # Maximum allowed execution time in seconds
    assert len(results) == num_workflows

@pytest.mark.asyncio
async def test_workflow_concurrent_execution(setup_ray, test_workflow_def, workflow_config):
    """Test concurrent workflow execution."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Execute multiple requests concurrently
    num_requests = 3
    input_data = {"test_input": "test_value"}
    
    start_time = time.time()
    result_refs = [workflow.execute_async.remote(input_data) for _ in range(num_requests)]
    results = []
    for ref in result_refs:
        result = ray.get(ref)
        results.append(result)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Verify results
    for result in results:
        assert result["status"] == "success"
        assert "results" in result
    
    # Check performance metrics
    assert execution_time < 20  # Maximum allowed execution time in seconds
    assert len(results) == num_requests

@pytest.mark.asyncio
async def test_workflow_memory_usage(setup_ray, test_workflow_def, workflow_config):
    """Test workflow memory usage under load."""
    # Create and execute multiple workflows
    num_workflows = 10
    workflows = []
    results = []
    
    start_time = time.time()
    
    for _ in range(num_workflows):
        workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
        assert isinstance(workflow, ray.actor.ActorHandle)
        workflows.append(workflow)
        
        input_data = {"test_input": "test_value"}
        result = await workflow.execute_async.remote(input_data)
        results.append(result)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Verify results
    for result in results:
        assert result["status"] == "success"
        assert "results" in result
    
    # Check performance metrics
    assert execution_time < 60  # Maximum allowed execution time in seconds
    assert len(results) == num_workflows
