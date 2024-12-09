import pytest
import asyncio
import time
import statistics
import psutil
import ray
from typing import List, Dict, Any
from agentflow.core.distributed_workflow import ResearchDistributedWorkflow

@pytest.fixture
def test_workflow():
    """Fixture for a test workflow definition"""
    return {
        'WORKFLOW': [
            {'input': ['research_topic', 'deadline', 'academic_level'], 'output': {'type': 'research'}, 'step': 1},
            {'input': ['WORKFLOW.1'], 'output': {'type': 'document'}, 'step': 2}
        ]
    }

@pytest.fixture
def test_config():
    """Fixture for test configuration"""
    return {
        'max_execution_time': 5.0,
        'max_concurrent_time': 10.0
    }

@ray.remote
class MockDistributedStep:
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a distributed step with configurable processing time.
        
        Args:
            input_data (Dict[str, Any]): Input data for the step
        
        Returns:
            Dict[str, Any]: Processed result with simulated processing time
        """
        # Simulate processing time
        processing_time = input_data.get('processing_time', 0.1)
        await asyncio.sleep(processing_time)
        
        return {
            'result': input_data,
            'processing_time': processing_time
        }

@pytest.fixture
def mock_ray_workflow_step():
    """Create a mock Ray remote step for testing"""
    return MockDistributedStep

@pytest.fixture
def mock_workflow(mock_ray_workflow_step, test_workflow):
    """Create a mock workflow with distributed steps"""
    workflow = ResearchDistributedWorkflow(config={}, workflow_def=test_workflow)
    workflow.distributed_steps = {
        step['step']: mock_ray_workflow_step.remote() 
        for step in test_workflow.get('WORKFLOW', [])
    }
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
            workflow.distributed_steps = {
                step['step']: mock_ray_workflow_step.remote() 
                for step in test_workflow.get('WORKFLOW', [])
            }
        
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