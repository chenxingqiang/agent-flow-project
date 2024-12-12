import pytest
import ray
import asyncio
import time
import psutil
import statistics
from typing import Dict, Any
from unittest.mock import MagicMock

from agentflow.core.distributed_workflow import (
    DistributedWorkflow,
    ResearchDistributedWorkflow,
    DistributedWorkflowStep
)
from agentflow.core.workflow_state import (
    WorkflowStateManager,
    StepStatus,
    WorkflowStatus
)
from agentflow.core.exceptions import (
    WorkflowExecutionError,
    WorkflowValidationError,
    StepExecutionError,
    StepTimeoutError
)

@pytest.fixture
def test_workflow():
    """Create a test workflow configuration"""
    return {
        "WORKFLOW": [
            {
                "step": 1,
                "type": "research",
                "name": "Step 1",
                "description": "Research step",
                "input": ["research_topic", "deadline", "academic_level"],
                "output": {"type": "research"},
                "agent_config": {}
            },
            {
                "step": 2,
                "type": "document",
                "name": "Step 2", 
                "description": "Document generation step",
                "input": ["WORKFLOW.1"],
                "output": {"type": "document"},
                "agent_config": {}
            }
        ]
    }

@pytest.fixture
def test_config():
    """Create a test configuration"""
    return {
        "max_execution_time": 5.0,
        "max_concurrent_time": 10.0,
        "max_retries": 3,
        "step_1_config": {
            "step_id": "step_1",
            "step": 1,
            "type": "research",
            "timeout": 30,
            "max_retries": 3,
            "preprocessors": [],
            "postprocessors": [],
            "input": ["research_topic", "deadline", "academic_level"],
            "output": {"type": "research"}
        },
        "step_2_config": {
            "step_id": "step_2",
            "step": 2,
            "type": "document", 
            "timeout": 30,
            "max_retries": 3,
            "preprocessors": [],
            "postprocessors": [],
            "input": ["WORKFLOW.1"],
            "output": {"type": "document"}
        }
    }

@pytest.fixture
def mock_ray_workflow_step(mocker):
    """Create a mock Ray workflow step"""
    class MockRayWorkflowStep:
        def __init__(self, step_id, config=None):
            self.step_id = step_id
            self.config = config or {}
        
        def _validate_input(self, input_data):
            """Override input validation to always pass"""
            return True
        
        async def execute(self, input_data):
            """Mock execute method that returns a simple result"""
            # Bypass input validation
            return {"result": f"Mock result for {self.step_id}", "input": input_data}
        
        @classmethod
        def remote(cls, step_id=None, config=None):
            """Simulate Ray's remote method"""
            return cls(step_id, config)
    
    # Create a Ray actor mock
    MockRayActor = mocker.patch('ray.remote', autospec=True)
    MockRayActor.return_value = MockRayWorkflowStep
    
    return MockRayWorkflowStep

@pytest.fixture
def mock_workflow(mock_ray_workflow_step, test_workflow, test_config):
    """Create a mock workflow with distributed steps"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    
    # Override the distributed steps with mock steps
    workflow.distributed_steps = {
        f"step_{step['step']}": mock_ray_workflow_step.remote(
            step_id=f"step_{step['step']}", 
            config=test_config.get(f"step_{step['step']}_config", {})
        ) for step in test_workflow.get('WORKFLOW', [])
    }
    
    # Override the input validation method for each step
    for step_id, step in workflow.distributed_steps.items():
        step._validate_input = lambda input_data: True
    
    # Override workflow-level input validation
    workflow.validate_input = lambda input_data: None
    
    # Override step input preparation to ensure required fields
    def _prepare_step_input(self, step, input_data, previous_results=None):
        # Ensure all required input fields are present
        required_fields = step.get('input', [])
        step_input = {field: input_data.get(field, f'Mock {field}') for field in required_fields}
        
        # Add any additional configuration or metadata
        step_input.update({
            'name': step.get('name', 'Mock Step'),
            'description': step.get('description', 'Mock Step Description'),
            'agent_config': step.get('agent_config', {}),
            'input': required_fields
        })
        
        return step_input
    
    workflow._prepare_step_input = _prepare_step_input
    
    return workflow

@pytest.mark.asyncio
async def test_workflow_execution_time(mock_workflow):
    """Test workflow execution time is within acceptable range"""
    input_data = {
        "research_topic": "Performance Testing",
        "deadline": "2024-12-31",
        "academic_level": "PhD",
        "STUDENT_NEEDS": "Performance Testing Needs",
        "LANGUAGE": "English", 
        "TEMPLATE": "Research Template"
    }
    
    start_time = time.time()
    result = await mock_workflow.execute_async(input_data)
    execution_time = time.time() - start_time
    
    assert result is not None
    assert execution_time < 5.0  # Maximum execution time in seconds

@pytest.mark.asyncio
async def test_concurrent_workflow_execution(mock_ray_workflow_step, test_workflow, test_config):
    """Test concurrent workflow execution with varying concurrency levels"""
    # Create a single workflow with mock distributed steps
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    
    # Override the distributed steps with mock steps
    workflow.distributed_steps = {
        f"step_{step['step']}": mock_ray_workflow_step.remote(
            step_id=f"step_{step['step']}", 
            config=test_config.get(f"step_{step['step']}_config", {})
        ) for step in test_workflow.get('WORKFLOW', [])
    }
    
    # Override the input validation method for each step
    for step_id, step in workflow.distributed_steps.items():
        step._validate_input = lambda input_data: True
    
    # Override workflow-level input validation
    workflow.validate_input = lambda input_data: None
    
    # Override step input preparation to ensure required fields
    def _prepare_step_input(self, step, input_data, previous_results=None):
        # Ensure all required input fields are present
        required_fields = step.get('input', [])
        step_input = {field: input_data.get(field, f'Mock {field}') for field in required_fields}
        
        # Add any additional configuration or metadata
        step_input.update({
            'name': step.get('name', 'Mock Step'),
            'description': step.get('description', 'Mock Step Description'),
            'agent_config': step.get('agent_config', {}),
            'input': required_fields
        })
        
        return step_input
    
    workflow._prepare_step_input = _prepare_step_input

    async def run_workflow_with_metrics(workflow, input_data, iteration):
        process = psutil.Process()
        start_cpu = process.cpu_percent()
        start_memory = process.memory_info().rss
        
        start_time = time.time()
        result = await workflow.execute_async({
            **input_data,
            "iteration": iteration  # Vary input data
        })
        execution_time = time.time() - start_time
        
        end_cpu = process.cpu_percent()
        end_memory = process.memory_info().rss
        
        return {
            'result': result,
            'execution_time': execution_time,
            'cpu_usage': end_cpu - start_cpu,
            'memory_delta': end_memory - start_memory
        }
    
    # Test different concurrency levels
    concurrency_levels = [2, 5, 10]
    base_input_data = {
        "research_topic": "Concurrent Testing",
        "deadline": "2024-12-31",
        "academic_level": "PhD",
        "STUDENT_NEEDS": "Performance Testing Needs",
        "LANGUAGE": "English",
        "TEMPLATE": "Research Template"
    }
    
    for num_workflows in concurrency_levels:
        tasks = [
            run_workflow_with_metrics(workflow, base_input_data, i)
            for i in range(num_workflows)
        ]
        
        start_time = time.time()
        metrics_list = await asyncio.gather(*tasks)
        
        # Validate metrics
        assert len(metrics_list) == num_workflows
        
        # Check execution times
        execution_times = [metrics['execution_time'] for metrics in metrics_list]
        assert all(time < 5.0 for time in execution_times)
        
        # Optional: Check other metrics
        cpu_usages = [metrics['cpu_usage'] for metrics in metrics_list]
        memory_deltas = [metrics['memory_delta'] for metrics in metrics_list]
        
        # Optional additional assertions
        def result_to_hashable(result):
            # If result is a dict, convert it to a sorted tuple of items
            if isinstance(result, dict):
                return tuple(sorted((k, str(v)) for k, v in result.items()))
            return result
        
        # Get unique results
        unique_results = {result_to_hashable(metrics['result']) for metrics in metrics_list}
        assert len(unique_results) == 1, f"Expected all results to be the same, got {unique_results}"

if __name__ == "__main__":
    pytest.main([__file__])