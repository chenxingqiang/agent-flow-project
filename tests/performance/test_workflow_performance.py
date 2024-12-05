import pytest
import asyncio
import time
import statistics
import psutil
from typing import List
from agentflow.core.distributed_workflow import ResearchDistributedWorkflow

@pytest.mark.asyncio
async def test_workflow_execution_time(test_workflow, test_config):
    """Test workflow execution time with warmup and multiple iterations"""
    workflow = ResearchDistributedWorkflow(config=test_config, workflow_def=test_workflow)
    
    input_data = {
        "research_topic": "Performance Testing",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    
    # Warmup run
    await workflow.execute_async(input_data)
    
    # Multiple iterations for statistical significance
    execution_times = []
    for _ in range(5):
        start_time = time.time()
        result = await workflow.execute_async(input_data)
        execution_times.append(time.time() - start_time)
    
    avg_time = statistics.mean(execution_times)
    max_time = max(execution_times)
    
    # More flexible performance assertions
    assert avg_time < test_config.get('max_execution_time', 5.0)
    assert max_time < test_config.get('max_execution_time', 5.0) * 1.5
    assert result is not None

@pytest.mark.asyncio
async def test_concurrent_workflow_execution(test_workflow, test_config):
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
            ResearchDistributedWorkflow(config=test_config, workflow_def=test_workflow)
            for _ in range(num_workflows)
        ]
        
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