import pytest
from unittest.mock import patch, MagicMock
import ray
from agentflow.core.base_workflow import BaseWorkflow
from agentflow.core.research_workflow import ResearchWorkflow, DistributedStep
from agentflow.core.rate_limiter import ModelRateLimiter
import tenacity
import os
import ell
from ell.types import Message, ContentBlock

@pytest.fixture(autouse=True)
def setup_ray():
    """Initialize Ray for testing"""
    if not ray.is_initialized():
        ray.init(local_mode=True)
    yield
    if ray.is_initialized():
        ray.shutdown()

@pytest.fixture
def mock_ell():
    """Mock ell LLM calls"""
    with patch('ell.simple') as mock:
        def decorator(func):
            def wrapper(*args, **kwargs):
                return {
                    "messages": [
                        Message(role="system", content=[ContentBlock(text="You are a research assistant.")]),
                        Message(role="user", content=[ContentBlock(text="Research topic: Test")]),
                        Message(role="assistant", content=[ContentBlock(text="Research findings...")])
                    ],
                    "result": ["Research finding 1", "Research finding 2"],
                    "methodology": ["Systematic literature review", "Qualitative analysis"],
                    "recommendations": ["Further research needed", "Explore alternative approaches"]
                }
            return wrapper
        mock.return_value = decorator
        yield mock

@pytest.fixture
def test_workflow_def():
    """Create test workflow definition"""
    return {
        "name": "test_research_workflow",
        "description": "Test research workflow",
        "required_inputs": ["research_topic", "deadline", "academic_level"],
        "steps": [
            {
                "step": 1,
                "type": "research",
                "description": "Conduct research",
                "outputs": ["messages", "result", "methodology", "recommendations"]  # Updated to include all expected outputs
            }
        ]
    }

@pytest.fixture
def test_workflow(test_workflow_def):
    """Create test workflow instance"""
    with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
        return ResearchWorkflow(test_workflow_def)

def test_workflow_execution(test_workflow, mock_ell):
    """Test workflow execution with valid input"""
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD",
        "timestamp": "2023-12-01T12:00:00Z"
    }

    results = test_workflow.execute(input_data)
    assert results["status"] == "completed"
    assert "messages" in results
    assert "result" in results
    assert "methodology" in results
    assert "recommendations" in results
    assert isinstance(results["messages"], list)
    assert all(isinstance(msg, Message) for msg in results["messages"])

def test_workflow_step_processing(test_workflow, mock_ell):
    """Test individual step processing"""
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }

    result = test_workflow.process_step(1, input_data)
    assert result["status"] == "completed"
    assert "messages" in result
    assert "result" in result
    assert "methodology" in result
    assert "recommendations" in result
    assert isinstance(result["messages"], list)

def test_workflow_state_management(test_workflow, mock_ell):
    """Test workflow state management"""
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }

    test_workflow.initialize_state()
    test_workflow.update_state({"key": "value"})
    
    result = test_workflow.process_step(1, input_data)
    assert result["status"] == "completed"
    assert "messages" in result
    assert "result" in result

def test_rate_limiter_integration(test_workflow, mock_ell):
    """Test rate limiter integration"""
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }

    result = test_workflow.process_step(1, input_data)
    assert result["status"] == "completed"
    assert "messages" in result
    assert "result" in result

@pytest.mark.distributed
def test_distributed_step_execution(test_workflow):
    """Test distributed step execution"""
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }

    result = test_workflow.execute_distributed(input_data)
    assert result["status"] == "completed"
    assert "messages" in result
    assert "result" in result
    assert isinstance(result["messages"], list)

@pytest.mark.distributed
def test_distributed_step_error_handling(test_workflow):
    """Test distributed step error handling"""
    input_data = {}  # Invalid input

    with pytest.raises(ValueError) as exc_info:
        test_workflow.execute_distributed(input_data)
    assert "Missing or empty research inputs" in str(exc_info.value)

@pytest.mark.distributed
def test_distributed_step_retry_mechanism(test_workflow):
    """Test distributed step retry mechanism"""
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }

    with patch('ray.get', side_effect=Exception("Test error")):
        with pytest.raises(ValueError) as exc_info:
            test_workflow.execute_distributed(input_data)
        assert "Missing required input" in str(exc_info.value)

def test_workflow_error_propagation(test_workflow):
    """Test error propagation in workflow"""
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }

    with patch.object(test_workflow, '_process_step_impl', side_effect=ValueError("Test error")):
        with pytest.raises(ValueError) as exc_info:
            test_workflow.process_step(1, input_data)
        assert "Test error" in str(exc_info.value)

def test_workflow_step_validation(test_workflow):
    """Test step output validation"""
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }

    with patch.object(test_workflow, '_process_step_impl', return_value={"status": "completed"}):
        with pytest.raises(ValueError) as exc_info:
            test_workflow.process_step(1, input_data)
        assert "missing required output field" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main([__file__])
