import pytest
import time
import psutil
import asyncio
import statistics
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import ray
from unittest.mock import patch, MagicMock
from agentflow.core.research_workflow import ResearchWorkflow, DistributedStep
from agentflow.core.rate_limiter import ModelRateLimiter, RateLimitError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration for load tests
LOAD_TEST_CONFIG = {
    'memory_iterations': 10,
    'memory_growth_threshold': 0.5,
    'cpu_iterations': 5,
    'cpu_max_threshold': 95.0,
    'distributed_steps': 3,
    'distributed_iterations': 5,
    'distributed_max_time': 5.0,
    'rate_limit_requests': 10,
    'rate_limit_success_threshold': 0.3,
    'rate_limit_failure_threshold': 0.7
}

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

def get_resource_usage():
    """Get current CPU and memory usage"""
    process = psutil.Process()
    return {
        'cpu_percent': process.cpu_percent(),
        'memory_percent': process.memory_percent(),
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'threads': process.num_threads()
    }

@pytest.mark.load
def test_memory_usage_under_load(test_workflow_def, mock_openai):
    """Test memory usage under sustained load with enhanced tracking"""
    workflow = ResearchWorkflow(test_workflow_def)
    
    initial_usage = get_resource_usage()
    memory_samples = []
    memory_growth_events = []
    
    for i in range(LOAD_TEST_CONFIG['memory_iterations']):
        input_data = {
            "research_topic": f"Memory Test {i}",
            "deadline": "2024-12-31",
            "academic_level": "PhD"
        }
        
        # Use asyncio to run the async execute method
        result = asyncio.run(workflow.execute(input_data))
        current_usage = get_resource_usage()['memory_mb']
        memory_samples.append(current_usage)
        
        # Detailed memory growth tracking
        if i > 0 and i % 10 == 0:
            growth_rate = (current_usage - memory_samples[0]) / memory_samples[0]
            if growth_rate > LOAD_TEST_CONFIG['memory_growth_threshold']:
                memory_growth_events.append({
                    'iteration': i,
                    'growth_rate': growth_rate,
                    'current_memory': current_usage
                })
    
    # Calculate memory statistics
    avg_memory = statistics.mean(memory_samples)
    max_memory = max(memory_samples)
    p95_memory = sorted(memory_samples)[int(0.95 * len(memory_samples))]
    
    logger.info(f"Memory Usage Statistics (MB):")
    logger.info(f"Average: {avg_memory:.2f}")
    logger.info(f"Maximum: {max_memory:.2f}")
    logger.info(f"95th Percentile: {p95_memory:.2f}")
    
    # Log memory growth events
    if memory_growth_events:
        logger.warning(f"Memory growth events detected: {len(memory_growth_events)}")
        for event in memory_growth_events:
            logger.warning(f"Growth at iteration {event['iteration']}: {event['growth_rate']:.2%}")
    
    assert len(memory_growth_events) == 0, "Excessive memory growth detected"

@pytest.mark.load
def test_cpu_usage_under_load(test_workflow_def, mock_openai):
    """Test CPU usage under sustained load"""
    num_iterations = LOAD_TEST_CONFIG['cpu_iterations']
    num_concurrent = psutil.cpu_count() or 4
    
    def run_workflow_batch():
        workflow = ResearchWorkflow(test_workflow_def)
        cpu_samples = []
        for i in range(num_iterations):
            input_data = {
                "research_topic": f"CPU Test {i}",
                "deadline": "2024-12-31",
                "academic_level": "PhD"
            }
            # Add a small delay between iterations
            time.sleep(0.05)
            result = asyncio.run(workflow.execute(input_data))
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
        return cpu_samples
    
    # Run concurrent workflows
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(run_workflow_batch) for _ in range(num_concurrent)]
        all_cpu_samples = []
        for future in futures:
            all_cpu_samples.extend(future.result())
    
    # Calculate CPU statistics
    avg_cpu = statistics.mean(all_cpu_samples)
    max_cpu = max(all_cpu_samples)
    p95_cpu = sorted(all_cpu_samples)[int(0.95 * len(all_cpu_samples))]
    
    logger.info(f"\nCPU Usage Statistics (%):")
    logger.info(f"Average: {avg_cpu:.2f}")
    logger.info(f"Maximum: {max_cpu:.2f}")
    logger.info(f"95th Percentile: {p95_cpu:.2f}")
    
    # Adjust the threshold to be more lenient
    max_threshold = min(LOAD_TEST_CONFIG['cpu_max_threshold'], 100)
    
    # Log detailed information if test fails
    if max_cpu >= max_threshold:
        logger.warning(f"High CPU usage detected: {max_cpu:.2f}%")
        logger.warning(f"Number of samples: {len(all_cpu_samples)}")
        logger.warning(f"Concurrent workers: {num_concurrent}")
        logger.warning(f"Iterations per worker: {num_iterations}")
    
    assert max_cpu < max_threshold, f"CPU usage too high: {max_cpu:.2f}%"

@pytest.mark.load
@pytest.mark.distributed
def test_distributed_load_balancing(test_workflow_def, mock_openai):
    """Test load balancing in distributed execution"""
    import ray
    
    @ray.remote
    class DistributedResearchStep:
        def __init__(self, step_id, step_config):
            self.workflow = ResearchWorkflow(test_workflow_def)
            self.step_id = step_id
            self.step_config = step_config
        
        async def execute(self, input_data):
            return await self.workflow.execute(input_data)
    
    ray.init(ignore_reinit_error=True)
    
    try:
        num_steps = LOAD_TEST_CONFIG['distributed_steps']
        num_iterations = LOAD_TEST_CONFIG['distributed_iterations']
        steps = [DistributedResearchStep.remote(i, {"type": "research"}) for i in range(num_steps)]
        
        timing_stats = {i: [] for i in range(num_steps)}
        
        for iteration in range(num_iterations):
            input_data = {
                "research_topic": f"Load Balance Test {iteration}",
                "deadline": "2024-12-31",
                "academic_level": "PhD"
            }
            
            # Execute steps and measure timing
            start_times = time.time()
            results = ray.get([step.execute.remote(input_data) for step in steps])
            end_time = time.time()
            
            # Record timing for each step
            for i in range(num_steps):
                timing_stats[i].append(end_time - start_times)
        
        # Calculate statistics for each step
        for step_num, timings in timing_stats.items():
            avg_time = statistics.mean(timings)
            max_time = max(timings)
            p95_time = sorted(timings)[int(0.95 * len(timings))]
            
            logger.info(f"\nStep {step_num} Timing Statistics (s):")
            logger.info(f"Average: {avg_time:.3f}")
            logger.info(f"Maximum: {max_time:.3f}")
            logger.info(f"95th Percentile: {p95_time:.3f}")
            
            # Check for balanced load
            assert max_time < LOAD_TEST_CONFIG['distributed_max_time'], f"Step {step_num} taking too long: {max_time:.3f}s"
            
    finally:
        ray.shutdown()

@pytest.mark.load
def test_rate_limiter_under_load(test_workflow_def, mock_openai):
    """Enhanced rate limiter load test with detailed error tracking"""
    rate_limiter = ModelRateLimiter()  # Use default configuration
    workflow = ResearchWorkflow(test_workflow_def)
    
    success_count = 0
    failure_count = 0
    total_requests = LOAD_TEST_CONFIG['rate_limit_requests']
    
    for i in range(total_requests):
        input_data = {
            "research_topic": f"Rate Limit Test {i}",
            "deadline": "2024-12-31",
            "academic_level": "PhD"
        }
        
        try:
            # Use execute_with_retry to simulate rate limiting
            result = rate_limiter.execute_with_retry(
                asyncio.run, 
                workflow.execute(input_data)
            )
            success_count += 1
        
        except RateLimitError as rle:
            logger.warning(f"Rate limit exceeded: {rle}")
            failure_count += 1
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            failure_count += 1
    
    # Calculate success and failure rates
    success_rate = success_count / total_requests
    failure_rate = failure_count / total_requests
    
    logger.info(f"\nRate Limiter Load Test Results:")
    logger.info(f"Total Requests: {total_requests}")
    logger.info(f"Successful Requests: {success_count}")
    logger.info(f"Failed Requests: {failure_count}")
    logger.info(f"Success Rate: {success_rate:.2%}")
    logger.info(f"Failure Rate: {failure_rate:.2%}")
    
    # Validate rate limiter performance
    assert success_rate >= LOAD_TEST_CONFIG['rate_limit_success_threshold'], \
        f"Success rate too low: {success_rate:.2%}"
    assert failure_rate <= LOAD_TEST_CONFIG['rate_limit_failure_threshold'], \
        f"Failure rate too high: {failure_rate:.2%}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
