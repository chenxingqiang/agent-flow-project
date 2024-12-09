import pytest
import ray
from unittest.mock import Mock, patch
from agentflow.core.distributed_workflow import (
    DistributedWorkflow, 
    ResearchDistributedWorkflow,
    DistributedWorkflowStep
)
from agentflow.core.config import StepConfig, AgentConfig
from agentflow.core.workflow_state import WorkflowStateManager, StepStatus, WorkflowStatus
from agentflow.core.retry import RetryConfig
import asyncio
import time

@pytest.fixture
def test_workflow():
    """Create a mock workflow definition for testing"""
    return {
        "name": "test_distributed_workflow",
        "description": "Test distributed workflow",
        "execution_policies": {
            "required_fields": ["research_topic", "deadline", "academic_level"],
            "default_status": "pending",
            "error_handling": {
                "missing_input_error": "Missing required inputs",
                "missing_field_error": "Missing required fields"
            }
        },
        "WORKFLOW": [
            StepConfig(
                id="step_1",
                step=1,
                name="Research Step",
                agents=["research_agent"],
                input_type="research_topic",
                output_type="research_findings",
                input=["research_topic", "deadline", "academic_level"],
                output={"type": "research"}
            ).model_dump(),
            StepConfig(
                id="step_2",
                step=2,
                name="Document Step",
                agents=["document_agent"],
                input_type="research_findings",
                output_type="document",
                input=["WORKFLOW.1"],
                output={"type": "document"}
            ).model_dump()
        ],
        "agents": [
            AgentConfig(
                id="research_agent",
                name="Research Agent",
                model={
                    "name": "mock_research_model",
                    "provider": "default"
                }
            ).model_dump(),
            AgentConfig(
                id="document_agent",
                name="Document Agent",
                model={
                    "name": "mock_document_model",
                    "provider": "default"
                }
            ).model_dump()
        ]
    }

@pytest.fixture
def test_config():
    """Create a mock configuration for testing"""
    return {
        "logging_level": "INFO",
        "max_iterations": 10,
        "max_retries": 3,
        "retry_delay": 1.0,
        "retry_backoff": 2.0,
        "step_1_config": {
            "preprocessors": [],
            "postprocessors": [],
            "step_function": None,
            "step_number": 1,
            "input_keys": ["research_topic", "deadline", "academic_level"],
            "output_type": "research"
        },
        "step_2_config": {
            "preprocessors": [],
            "postprocessors": [],
            "step_function": None,
            "step_number": 2,
            "input_keys": ["WORKFLOW.1"],
            "output_type": "document"
        }
    }

@pytest.fixture
def minimal_distributed_workflow():
    """Create a minimal distributed workflow definition without execution policies."""
    return {
        "name": "minimal_distributed_workflow",
        "description": "Minimal distributed workflow",
        "WORKFLOW": [
            StepConfig(
                id="step_1",
                step=1,
                name="Basic Step",
                agents=["basic_agent"],
                input_type="text",
                output_type="text",
                input=["input_text"],
                output={"type": "text"}
            ).model_dump()
        ],
        "agents": [
            AgentConfig(
                id="basic_agent",
                name="Basic Agent",
                model={
                    "name": "test-model",
                    "provider": "ray"  # Update the provider to a valid one
                }
            ).model_dump()
        ]
    }

@pytest.fixture
def mock_research_step():
    """Create a mock research step actor"""
    @ray.remote
    class MockResearchStep:
        async def execute(self, input_data):
            return {
                'result': {
                    'research_findings': "Mock research findings"
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
    return MockResearchStep.remote()

@pytest.fixture
def mock_document_step():
    """Create a mock document generation step actor"""
    @ray.remote
    class MockDocumentStep:
        async def execute(self, input_data):
            return {
                'result': {
                    'document': "Mock document"
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
    return MockDocumentStep.remote()

@pytest.fixture
def failing_step():
    """Create a failing step actor"""
    @ray.remote
    class FailingStep:
        async def execute(self, input_data):
            raise RuntimeError("Step execution failed")
    return FailingStep.remote()

@pytest.mark.asyncio
async def test_distributed_workflow_initialization(test_workflow, test_config):
    """Test distributed workflow initialization"""
    workflow = ResearchDistributedWorkflow(config=test_config, workflow_def=test_workflow)
    
    assert workflow is not None
    assert len(workflow.workflow_def["WORKFLOW"]) == 2
    assert workflow.config == test_config
    assert isinstance(workflow.distributed_steps, dict)
    assert len(workflow.distributed_steps) == 2
    assert workflow.state_manager is not None
    assert workflow.retry_config is not None
    assert workflow.retry_config.max_retries == 3

@pytest.mark.asyncio
async def test_minimal_distributed_workflow(minimal_distributed_workflow, test_config):
    """Test distributed workflow initialization without execution policies."""
    workflow = DistributedWorkflow(minimal_distributed_workflow, test_config)
    assert workflow.required_fields == []
    assert workflow.error_handling == {}
    assert workflow.default_status is None
    assert len(workflow.workflow_steps) == 1

@pytest.mark.asyncio
async def test_distributed_workflow_execution(test_workflow, test_config, mock_research_step, mock_document_step):
    """Test distributed workflow execution"""
    workflow = ResearchDistributedWorkflow(test_config, test_workflow)
    workflow.distributed_steps[1] = mock_research_step
    workflow.distributed_steps[2] = mock_document_step
    
    input_data = {
        "research_topic": "AI Testing",
        "deadline": "2023-12-31",
        "academic_level": "PhD"
    }
    
    results = await workflow.execute(input_data)
    
    assert len(results) == 2
    assert results[1]['result']['research_findings'] == "Mock research findings"
    assert results[2]['result']['document'] == "Mock document"
    assert 'metadata' in results[1]
    assert 'metadata' in results[2]
    assert 'timestamp' in results[1]['metadata']
    assert 'timestamp' in results[2]['metadata']

@pytest.mark.asyncio
async def test_distributed_workflow_error_handling(test_workflow, test_config, failing_step):
    """Test error handling in distributed workflow"""
    workflow = ResearchDistributedWorkflow(test_config, test_workflow)
    workflow.distributed_steps[1] = failing_step
    
    input_data = {
        "research_topic": "AI Testing",
        "deadline": "2023-12-31",
        "academic_level": "PhD"
    }
    
    with pytest.raises(ValueError) as exc_info:
        await workflow.execute(input_data)
    
    assert "Step 1 execution failed" in str(exc_info.value)
    assert "Step execution failed" in str(exc_info.value)
    assert workflow.state_manager.get_step_status(1) == StepStatus.FAILED

@pytest.mark.asyncio
async def test_distributed_workflow_async_execution(test_workflow, test_config, mock_research_step, mock_document_step):
    """Test async execution of workflow steps"""
    # Add async preprocessor and postprocessor
    async def async_preprocessor(data):
        await asyncio.sleep(0.1)
        data['preprocessed'] = True
        return data
        
    async def async_postprocessor(data):
        await asyncio.sleep(0.1)
        data['postprocessed'] = True
        return data
    
    test_config['step_1_config']['preprocessors'] = [async_preprocessor]
    test_config['step_1_config']['postprocessors'] = [async_postprocessor]
    
    workflow = ResearchDistributedWorkflow(test_config, test_workflow)
    workflow.distributed_steps[1] = mock_research_step
    workflow.distributed_steps[2] = mock_document_step
    
    input_data = {
        "research_topic": "AI Testing",
        "deadline": "2023-12-31",
        "academic_level": "PhD"
    }
    
    results = await workflow.execute(input_data)
    
    assert results[1]['preprocessed']
    assert results[1]['postprocessed']

@pytest.mark.asyncio
async def test_distributed_workflow_retry_mechanism(test_workflow, test_config, failing_step):
    """Test retry mechanism with failing step"""
    test_config['max_retries'] = 2
    test_config['retry_delay'] = 0.1
    test_config['retry_backoff'] = 1.5
    
    workflow = ResearchDistributedWorkflow(test_config, test_workflow)
    workflow.distributed_steps[1] = failing_step
    
    input_data = {
        "research_topic": "AI Testing",
        "deadline": "2023-12-31",
        "academic_level": "PhD"
    }
    
    with pytest.raises(ValueError) as exc_info:
        await workflow.execute(input_data)
    
    assert "Step 1 execution failed" in str(exc_info.value)
    # Check that retries were attempted
    assert workflow.state_manager.get_step_retry_count(1) == 2

@pytest.mark.asyncio
async def test_distributed_workflow_step_dependency(test_workflow, test_config):
    """Test step dependency validation and execution order"""
    # Add a third step that depends on both step 1 and 2
    test_workflow['WORKFLOW'].append({
        'step': 3,
        'name': 'Summary Step',
        'agents': ['summary_agent'],
        'input': ['WORKFLOW.1', 'WORKFLOW.2'],
        'output': {'type': 'summary'}
    })
    
    @ray.remote
    class MockSummaryStep:
        async def execute(self, input_data):
            # Get the results from steps 1 and 2
            step1_result = input_data.get('WORKFLOW.1', {}).get('result', {})
            step2_result = input_data.get('WORKFLOW.2', {}).get('result', {})
            
            return {
                'result': {
                    'summary': f"Summary of {step1_result.get('research_findings', '')} and {step2_result.get('document', '')}"
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
    
    workflow = ResearchDistributedWorkflow(test_config, test_workflow)
    workflow.distributed_steps[3] = MockSummaryStep.remote()
    
    input_data = {
        "research_topic": "AI Testing",
        "deadline": "2023-12-31",
        "academic_level": "PhD"
    }
    
    results = await workflow.execute(input_data)
    
    assert len(results) == 3
    assert 'summary' in results[3]['result']
    # Verify execution order through timestamps
    assert results[3]['metadata']['timestamp'] > results[1]['metadata']['timestamp']
    assert results[3]['metadata']['timestamp'] > results[2]['metadata']['timestamp']

@pytest.mark.asyncio
async def test_distributed_workflow_timeout(test_workflow, test_config):
    """Test workflow step timeout"""
    @ray.remote
    class SlowStep:
        async def execute(self, input_data):
            await asyncio.sleep(2)  # Sleep for 2 seconds
            return {
                'result': 'Slow result',
                'metadata': {
                    'timestamp': time.time()
                }
            }
    
    test_config['step_timeout'] = 0.1  # Set timeout to 0.1 seconds
    workflow = ResearchDistributedWorkflow(test_config, test_workflow)
    workflow.distributed_steps[1] = SlowStep.remote()
    
    input_data = {
        "research_topic": "AI Testing",
        "deadline": "2023-12-31",
        "academic_level": "PhD"
    }
    
    with pytest.raises(ValueError) as exc_info:
        await workflow.execute(input_data)
    
    assert "Step 1 execution timed out" in str(exc_info.value)
    assert workflow.state_manager.get_step_status(1) == StepStatus.FAILED

@pytest.mark.asyncio
async def test_distributed_workflow_parallel_execution(test_workflow, test_config):
    """Test parallel execution of independent steps"""
    execution_order = []
    
    @ray.remote
    class OrderedStep:
        async def execute(self, input_data):
            await asyncio.sleep(0.1)
            step_num = input_data.get('step_num')
            execution_order.append(step_num)
            return {
                'result': f'Result from step {step_num}',
                'metadata': {
                    'timestamp': time.time(),
                    'step_num': step_num
                }
            }
    
    # Create two independent steps
    test_workflow['WORKFLOW'] = [
        {
            'step': 1,
            'name': 'Step 1',
            'input': ['input_1'],
            'output': {'type': 'output_1'}
        },
        {
            'step': 2,
            'name': 'Step 2',
            'input': ['input_2'],
            'output': {'type': 'output_2'}
        }
    ]
    
    test_config['max_concurrent_steps'] = 2
    workflow = ResearchDistributedWorkflow(test_config, test_workflow)
    workflow.distributed_steps[1] = OrderedStep.remote()
    workflow.distributed_steps[2] = OrderedStep.remote()
    
    input_data = {
        'input_1': {'step_num': 1},
        'input_2': {'step_num': 2}
    }
    
    results = await workflow.execute(input_data)
    
    assert len(results) == 2
    assert 'metadata' in results[1]
    assert 'metadata' in results[2]
    # Since steps are independent and run in parallel,
    # both should complete around the same time
    assert abs(results[1]['metadata']['timestamp'] - results[2]['metadata']['timestamp']) < 0.2

if __name__ == "__main__":
    pytest.main([__file__])
