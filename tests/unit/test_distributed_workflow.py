import pytest
import ray
from unittest.mock import Mock, patch
from agentflow.core.distributed_workflow import (
    DistributedWorkflow, 
    ResearchDistributedWorkflow,
    DistributedWorkflowStep
)

@pytest.fixture
def test_workflow():
    """Create a mock workflow definition for testing"""
    return {
        "WORKFLOW": [
            {
                "step": 1,
                "input": ["research_topic", "deadline", "academic_level"],
                "output": {"type": "research"}
            },
            {
                "step": 2,
                "input": ["WORKFLOW.1"],
                "output": {"type": "document"}
            }
        ]
    }

@pytest.fixture
def test_config():
    """Create a mock configuration for testing"""
    return {
        "logging_level": "INFO",
        "max_iterations": 10,
        "step_1_config": {
            "preprocessors": [],
            "postprocessors": []
        },
        "step_2_config": {
            "preprocessors": [],
            "postprocessors": []
        }
    }

@pytest.fixture
def mock_research_step():
    """Create a mock research step function"""
    def research_step(input_data):
        return {
            "research_results": f"Research on {input_data['research_topic']}",
            "academic_level": input_data['academic_level']
        }
    return research_step

@pytest.fixture
def mock_document_step():
    """Create a mock document generation step function"""
    def document_step(input_data):
        prev_result = input_data['step_1_result']
        return {
            "document": f"Document based on {prev_result['result']['research_results']}"
        }
    return document_step

def test_distributed_workflow_initialization(test_workflow, test_config):
    """Test distributed workflow initialization"""
    workflow = ResearchDistributedWorkflow(config=test_config, workflow_def=test_workflow)
    
    assert workflow is not None
    assert len(workflow.workflow_def["WORKFLOW"]) == 2
    assert workflow.config == test_config
    assert isinstance(workflow.distributed_steps, dict)
    assert len(workflow.distributed_steps) == 2

def test_distributed_workflow_execution(test_workflow, test_config):
    """Test distributed workflow execution"""
    workflow = ResearchDistributedWorkflow(config=test_config, workflow_def=test_workflow)
    
    input_data = {
        "research_topic": "Distributed AI Systems",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    
    result = workflow.execute(input_data)
    
    assert result is not None
    assert 1 in result
    assert 2 in result

@pytest.mark.asyncio
async def test_distributed_workflow_async_execution(test_workflow, test_config):
    """Test async distributed workflow execution"""
    workflow = ResearchDistributedWorkflow(config=test_config, workflow_def=test_workflow)
    
    input_data = {
        "research_topic": "Async Distributed Systems",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    
    result = await workflow.execute_async(input_data)
    assert result is not None
    assert 1 in result
    assert 2 in result

def test_distributed_workflow_error_handling(test_workflow, test_config):
    """Test error handling in distributed workflow"""
    workflow = ResearchDistributedWorkflow(config=test_config, workflow_def=test_workflow)
    
    # Test with empty input
    with pytest.raises(ValueError, match="Empty input data"):
        workflow.execute({})
    
    # Test with missing research topic
    with pytest.raises(ValueError, match="Missing or empty inputs: research_topic"):
        workflow.execute({
            "deadline": "2024-12-31",
            "academic_level": "PhD"
        })

def test_distributed_workflow_step_config(test_workflow, test_config):
    """Test workflow step configuration"""
    def uppercase_topic_preprocessor(input_data):
        if 'research_topic' in input_data:
            input_data['research_topic'] = input_data['research_topic'].upper()
        return input_data
    
    def add_metadata_postprocessor(result):
        result['metadata'] = {'processed': True}
        return result
    
    test_config['step_1_config']['preprocessors'] = [uppercase_topic_preprocessor]
    test_config['step_1_config']['postprocessors'] = [add_metadata_postprocessor]
    
    workflow = ResearchDistributedWorkflow(config=test_config, workflow_def=test_workflow)
    
    input_data = {
        "research_topic": "Distributed AI Systems",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    
    result = workflow.execute(input_data)
    assert result is not None
    assert result[1]['research_topic'] == "DISTRIBUTED AI SYSTEMS"
    assert result[1]['metadata']['processed'] is True

def test_distributed_workflow_step_execution():
    """Test individual distributed workflow step execution"""
    def mock_step_func(input_data):
        return {"result": f"Processed {input_data['data']}"}
    
    step_config = {
        'step_function': mock_step_func,
        'step_number': 1,
        'input_keys': ['data'],
        'output_type': 'test'
    }
    step = DistributedWorkflowStep.remote(step_config)
    result = ray.get(step.execute.remote({"data": "test_input"}))
    
    assert result is not None
    assert result["step_num"] == 1
    assert result["result"]["result"] == "Processed test_input"
    assert "timestamp" in result["metadata"]
    assert "worker_id" in result["metadata"]

def test_distributed_workflow_step_error_handling():
    """Test error handling in distributed workflow step"""
    def failing_step_func(input_data):
        raise ValueError("Step execution failed")
    
    step_config = {
        'step_function': failing_step_func,
        'step_number': 1,
        'input_keys': ['data'],
        'output_type': 'test'
    }
    step = DistributedWorkflowStep.remote(step_config)
    
    with pytest.raises(ray.exceptions.RayTaskError) as exc_info:
        ray.get(step.execute.remote({"data": "test_input"}))
    assert "Step execution failed" in str(exc_info.value)

def test_distributed_workflow_input_preparation(test_workflow, test_config):
    """Test workflow input preparation"""
    workflow = ResearchDistributedWorkflow(config=test_config, workflow_def=test_workflow)
    
    # Test with direct input
    step = test_workflow["WORKFLOW"][0]
    input_data = {
        "research_topic": "Test Topic",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    prepared_input = workflow._prepare_step_input(step, input_data, {})
    
    assert prepared_input["research_topic"] == "Test Topic"
    assert prepared_input["deadline"] == "2024-12-31"
    assert prepared_input["academic_level"] == "PhD"
    
    # Test with workflow reference input
    step = test_workflow["WORKFLOW"][1]
    previous_results = {
        "step_1": {"result": "Previous step result"}
    }
    prepared_input = workflow._prepare_step_input(step, input_data, previous_results)
    assert prepared_input["previous_step_result"] == previous_results["step_1"]

def test_complete_workflow_execution(test_workflow, test_config, 
                                   mock_research_step, mock_document_step):
    """Test complete workflow execution with mock steps"""
    # Configure step functions
    test_config['step_1_config']['step_function'] = mock_research_step
    test_config['step_2_config']['step_function'] = mock_document_step
    
    workflow = ResearchDistributedWorkflow(config=test_config, workflow_def=test_workflow)
    
    input_data = {
        "research_topic": "Distributed Systems",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    
    result = workflow.execute(input_data)
    
    assert result[1]['result']['research_results'].startswith("Research on Distributed Systems")
    assert result[2]['result']['document'].startswith("Document based on")

@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for tests"""
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    yield
    
    # Shutdown Ray after tests
    if ray.is_initialized():
        ray.shutdown()

if __name__ == "__main__":
    pytest.main([__file__])
