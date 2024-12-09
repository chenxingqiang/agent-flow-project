import pytest
import asyncio
import time
import statistics
import psutil
import ray
from typing import List, Dict, Any
from agentflow.core.distributed_workflow import ResearchDistributedWorkflow

@pytest.fixture
def mock_ray_workflow_step():
    """Create a mock Ray remote step for testing"""
    @ray.remote
    class MockDistributedStep:
        def __init__(self):
            self._mock_result = None

        def set_mock_result(self, result):
            self._mock_result = result

        async def execute(self, input_data):
            # Simulated remote method that returns a predefined result
            return self._mock_result or {'result': input_data, 'processed': True}

    return MockDistributedStep

@pytest.fixture
def mock_workflow(mock_ray_workflow_step, test_workflow):
    """Create a mock workflow with predefined distributed steps"""
    workflow = ResearchDistributedWorkflow(config={}, workflow_def=test_workflow)
    
    # Create mock steps and set default results
    mock_steps = {}
    for step in test_workflow.get('WORKFLOW', []):
        mock_step = mock_ray_workflow_step.remote()
        mock_step.set_mock_result.remote({'result': {'step': step['step']}, 'processed': True})
        mock_steps[step['step']] = mock_step
    
    workflow.distributed_steps = mock_steps
    return workflow

@pytest.mark.asyncio
async def test_workflow_execution_time(mock_workflow, test_config):
    """Test workflow execution time with warmup and multiple iterations"""
    input_data = {
        "research_topic": "Performance Testing",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    
    # Warmup run
    await mock_workflow.execute_async(input_data)
    
    # Multiple iterations for statistical significance
    execution_times = []
    for _ in range(5):
        start_time = time.time()
        result = await mock_workflow.execute_async(input_data)
        execution_times.append(time.time() - start_time)
    
    avg_time = statistics.mean(execution_times)
    max_time = max(execution_times)
    
    # More flexible performance assertions
    assert avg_time < test_config.get('max_execution_time', 5.0)
    assert max_time < test_config.get('max_execution_time', 5.0) * 1.5
    assert result is not None

@pytest.mark.asyncio
async def test_concurrent_workflow_execution(mock_ray_workflow_step, test_workflow, test_config):
    """Test concurrent workflow execution with varying concurrency levels"""
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
        "academic_level": "PhD"
    }
    
    for num_workflows in concurrency_levels:
        workflows = [
            ResearchDistributedWorkflow(config={}, workflow_def=test_workflow)
            for _ in range(num_workflows)
        ]
        
        # Manually set distributed steps for each workflow
        for workflow in workflows:
            mock_steps = {}
            for step in test_workflow.get('WORKFLOW', []):
                mock_step = mock_ray_workflow_step.remote()
                mock_step.set_mock_result.remote({'result': {'step': step['step']}, 'processed': True})
                mock_steps[step['step']] = mock_step
            
            workflow.distributed_steps = mock_steps
        
        tasks = [
            run_workflow_with_metrics(workflow, base_input_data, i) 
            for i, workflow in enumerate(workflows)
        ]
        
        start_time = time.time()
        metrics_list = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        avg_execution_time = statistics.mean(m['execution_time'] for m in metrics_list)
        max_execution_time = max(m['execution_time'] for m in metrics_list)
        avg_cpu_usage = statistics.mean(m['cpu_usage'] for m in metrics_list)
        total_memory_delta = sum(m['memory_delta'] for m in metrics_list)
        
        # Flexible assertions based on configuration
        max_concurrent_time = test_config.get('max_concurrent_time', 10.0)
        assert avg_execution_time < max_concurrent_time
        assert max_execution_time < max_concurrent_time * 1.5
        assert all(m['result'] is not None for m in metrics_list)

if __name__ == "__main__":
    pytest.main([__file__])