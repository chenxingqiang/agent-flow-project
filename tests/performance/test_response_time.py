import pytest
import time
import asyncio
import statistics
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock
import ray
from agentflow.core.research_workflow import ResearchWorkflow, DistributedStep
from agentflow.core.rate_limiter import ModelRateLimiter

@pytest.fixture
def mock_openai():
    with patch('openai.ChatCompletion.create') as mock:
        mock.return_value = {
            'choices': [{
                'message': {
                    'content': 'Test research results for mock API call',
                    'role': 'assistant'
                }
            }],
            'usage': {'total_tokens': 100}
        }
        yield mock

@pytest.fixture
def test_workflow_def():
    return {
        "name": "test_research_workflow",
        "description": "Test research workflow",
        "required_inputs": ["research_topic", "deadline", "academic_level"],
        "steps": [
            {
                "step": 1,
                "type": "research",
                "description": "Conduct research",
                "outputs": ["result", "methodology", "recommendations"]
            }
        ]
    }

@pytest.fixture
def test_workflow(test_workflow_def):
    return ResearchWorkflow(test_workflow_def)

@pytest.mark.performance
def test_single_workflow_response_time(test_workflow, mock_openai):
    """Test response time for a single workflow execution"""
    input_data = {
        "research_topic": "Performance Testing",
        "deadline": "2024-12-31",
        "academic_level": "PhD",
        "timestamp": "2023-12-01T12:00:00Z"
    }
    
    # Warm-up run
    test_workflow.execute(input_data)
    
    # Test run
    start_time = time.time()
    result = test_workflow.execute(input_data)
    end_time = time.time()
    
    response_time = end_time - start_time
    assert response_time < 1.0  # Should complete within 1 second with mocked API
    assert result is not None
    assert result["status"] == "completed"

@pytest.mark.performance
def test_sequential_workflow_response_time(test_workflow, mock_openai):
    """Test response time for sequential workflow executions"""
    input_data = {
        "research_topic": "Sequential Performance Testing",
        "deadline": "2024-12-31",
        "academic_level": "PhD",
        "timestamp": "2023-12-01T12:00:00Z"
    }
    
    num_runs = 10
    response_times = []
    
    # Sequential executions
    for i in range(num_runs):
        input_data["research_topic"] = f"Sequential Test {i}"
        start_time = time.time()
        result = test_workflow.execute(input_data)
        end_time = time.time()
        response_times.append(end_time - start_time)
        assert result["status"] == "completed"
    
    # Calculate statistics
    avg_time = statistics.mean(response_times)
    max_time = max(response_times)
    p95_time = sorted(response_times)[int(0.95 * num_runs)]
    
    assert avg_time < 0.5  # Average should be under 0.5s
    assert max_time < 1.0  # Max should be under 1s
    assert p95_time < 0.8  # 95th percentile should be under 0.8s

@pytest.mark.performance
def test_concurrent_workflow_response_time(test_workflow_def, mock_openai):
    """Test response time for concurrent workflow executions"""
    num_concurrent = 5
    num_runs_per_workflow = 3
    
    def run_workflow():
        workflow = ResearchWorkflow(test_workflow_def)
        response_times = []
        
        for i in range(num_runs_per_workflow):
            input_data = {
                "research_topic": f"Concurrent Test {i}",
                "deadline": "2024-12-31",
                "academic_level": "PhD",
                "timestamp": "2023-12-01T12:00:00Z"
            }
            
            start_time = time.time()
            result = workflow.execute(input_data)
            end_time = time.time()
            
            response_times.append(end_time - start_time)
            assert result["status"] == "completed"
            
        return response_times
    
    # Run concurrent workflows
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        all_response_times = list(executor.map(lambda _: run_workflow(), range(num_concurrent)))
    
    # Flatten response times
    flat_response_times = [t for times in all_response_times for t in times]
    
    # Calculate statistics
    avg_time = statistics.mean(flat_response_times)
    max_time = max(flat_response_times)
    p95_time = sorted(flat_response_times)[int(0.95 * len(flat_response_times))]
    
    assert avg_time < 1.0  # Average should be under 1s
    assert max_time < 2.0  # Max should be under 2s
    assert p95_time < 1.5  # 95th percentile should be under 1.5s

@pytest.mark.performance
@pytest.mark.distributed
def test_distributed_step_response_time():
    """Test response time for distributed step execution"""
    ray.init(ignore_reinit_error=True)
    
    try:
        num_steps = 5
        steps = [DistributedStep.remote(i, {"type": "research"}) for i in range(num_steps)]
        
        input_data = {
            "research_topic": "Distributed Performance Test",
            "deadline": "2024-12-31",
            "academic_level": "PhD"
        }
        
        # Warm-up run
        ray.get([step.execute.remote(input_data) for step in steps])
        
        # Test run
        start_time = time.time()
        results = ray.get([step.execute.remote(input_data) for step in steps])
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_step = total_time / num_steps
        
        assert avg_time_per_step < 0.5  # Average time per step should be under 0.5s
        assert all(isinstance(r, dict) for r in results)
        assert all("result" in r for r in results)
        
    finally:
        ray.shutdown()

@pytest.mark.performance
def test_rate_limiter_performance(test_workflow, mock_openai):
    """Test performance impact of rate limiting"""
    rate_limiter = ModelRateLimiter(max_retries=3, retry_delay=0.1)
    test_workflow.rate_limiter = rate_limiter
    
    input_data = {
        "research_topic": "Rate Limiter Performance Test",
        "deadline": "2024-12-31",
        "academic_level": "PhD",
        "timestamp": "2023-12-01T12:00:00Z"
    }
    
    # Test with successful execution
    start_time = time.time()
    result = test_workflow.execute(input_data)
    normal_time = time.time() - start_time
    
    # Test with retry
    with patch.object(rate_limiter, 'execute_with_retry') as mock_retry:
        mock_retry.side_effect = [Exception("Rate limit"), Exception("Rate limit"), {"result": "Success"}]
        
        start_time = time.time()
        result = test_workflow.execute(input_data)
        retry_time = time.time() - start_time
    
    assert retry_time > normal_time  # Retry should take longer
    assert retry_time < normal_time + 1.0  # But not too much longer with mocked API
    assert result is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])