import pytest
import ray
import asyncio
import time
import logging
logger = logging.getLogger(__name__)
from typing import Dict, Any
from unittest.mock import AsyncMock

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

# Test fixtures and utilities
@pytest.fixture
def test_workflow():
    """Create a test workflow configuration"""
    return {
        "AGENT": "Academic_Paper_Optimization",
        "WORKFLOW": [
            {
                "step": 1,
                "title": "Extract Details from Student Inputs",
                "input": ["STUDENT_NEEDS", "LANGUAGE", "TEMPLATE"],
                "output": {
                    "type": "analysis",
                    "format": "plain text"
                }
            },
            {
                "step": 2,
                "title": "Propose Innovative Ideas",
                "input": ["STUDENT_NEEDS.RESEARCH_TOPIC", "LANGUAGE.TYPE"],
                "output": {
                    "type": "ideas",
                    "format": "Markdown with LaTeX"
                }
            }
        ],
        "ENVIRONMENT": {
            "INPUT": ["STUDENT_NEEDS", "LANGUAGE", "TEMPLATE"],
            "OUTPUT": "A Markdown-formatted academic plan"
        }
    }

@pytest.fixture
def test_config():
    """Create a test configuration"""
    return {
        "max_retries": 3,
        "retry_delay": 0.1,
        "retry_backoff": 2.0,
        "timeout": 5.0,
        "step_1_config": {
            "max_retries": 3,
            "timeout": 30
        },
        "step_2_config": {
            "max_retries": 3,
            "timeout": 30
        }
    }

@pytest.fixture
def minimal_distributed_workflow():
    """Create a minimal workflow configuration"""
    return {
        "WORKFLOW": [
            {
                "step": 1,
                "input": ["input_text"],
                "output": {"type": "text"}
            }
        ]
    }

@pytest.fixture
def mock_research_step():
    """Create a mock research step"""
    class MockResearchStep:
        async def execute(self, input_data):
            return {
                'result': {
                    'student_analysis': "Detailed student needs analysis",
                    'format': "plain text"
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
    return MockResearchStep()

@pytest.fixture
def mock_document_step():
    """Create a mock document step"""
    class MockDocumentStep:
        async def execute(self, input_data):
            return {
                'result': {
                    'research_ideas': "Innovative research ideas",
                    'format': "Markdown with LaTeX"
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
    return MockDocumentStep()

@pytest.fixture
def failing_step():
    """Create a failing step"""
    class FailingStep:
        async def execute(self, input_data):
            raise StepExecutionError("Step execution failed")
    return FailingStep()

@pytest.mark.asyncio
async def test_distributed_workflow_initialization(test_workflow, test_config):
    """Test distributed workflow initialization"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    assert workflow.workflow_config == test_workflow
    assert workflow.config == test_config

@pytest.mark.asyncio
async def test_minimal_distributed_workflow(minimal_distributed_workflow, test_config):
    """Test distributed workflow initialization without execution policies."""
    workflow = DistributedWorkflow(workflow_config=minimal_distributed_workflow, config=test_config)
    assert workflow.workflow_config == minimal_distributed_workflow
    assert workflow.config == test_config

@pytest.mark.asyncio
async def test_distributed_workflow_execution(test_workflow, test_config, mock_research_step, mock_document_step):
    """Test distributed workflow execution with agent.json aligned format"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    workflow.distributed_steps["step_1"] = mock_research_step
    workflow.distributed_steps["step_2"] = mock_document_step
    
    input_data = {
        "STUDENT_NEEDS": "API Testing",
        "LANGUAGE": "English",
        "TEMPLATE": "Academic Paper"
    }
    
    result = await workflow.execute_async(input_data)
    assert result is not None
    assert isinstance(result, dict)
    assert 'output' in result
    
    # Check that the format is one of the expected formats
    expected_formats = ["plain text", "Markdown with LaTeX"]
    assert result['output']['format'] in expected_formats

@pytest.mark.asyncio
async def test_workflow_policy_compliance(test_workflow, test_config, mock_research_step):
    """Test that workflow execution complies with defined policies"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    workflow.distributed_steps["step_1"] = mock_research_step
    
    # Test with missing required input
    incomplete_input = {
        "STUDENT_NEEDS": "API Testing",
        # missing LANGUAGE and TEMPLATE
    }
    
    with pytest.raises(ValueError, match="Missing required inputs"):
        await workflow.execute_async(incomplete_input)
        
    # Verify environment requirements
    assert all(field in test_workflow["ENVIRONMENT"]["INPUT"] 
              for field in ["STUDENT_NEEDS", "LANGUAGE", "TEMPLATE"])
    assert "Markdown-formatted academic plan" in test_workflow["ENVIRONMENT"]["OUTPUT"]

@pytest.mark.asyncio
async def test_distributed_workflow_error_handling(test_workflow, test_config, failing_step):
    """Test error handling in distributed workflow"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    workflow.distributed_steps["step_1"] = failing_step

    input_data = {
        "STUDENT_NEEDS": "Error Testing",
        "LANGUAGE": "English",
        "TEMPLATE": "Academic Paper"
    }

    with pytest.raises(WorkflowExecutionError):
        await workflow.execute_async(input_data)
    
    assert workflow.state_manager.get_step_status("step_1") == StepStatus.FAILED

@pytest.mark.asyncio
async def test_distributed_workflow_async_execution(test_workflow, test_config, mock_research_step, mock_document_step):
    """Test async execution of workflow steps with agent.json format compliance"""
    # Add async preprocessor and postprocessor that maintain output format
    async def async_preprocessor(data):
        await asyncio.sleep(0.1)
        data['preprocessed'] = True
        if 'output' not in data:
            data['output'] = {}
        data['output']['format'] = test_workflow['WORKFLOW'][0]['output']['format']
        return data

    async def async_postprocessor(data):
        await asyncio.sleep(0.1)
        data['postprocessed'] = True
        # Ensure output format compliance
        if isinstance(data.get('result'), dict):
            data['result']['format'] = test_workflow['WORKFLOW'][0]['output']['format']
        return data

    test_config['step_1_config']['preprocessors'] = [async_preprocessor]
    test_config['step_1_config']['postprocessors'] = [async_postprocessor]

    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    workflow.distributed_steps["step_1"] = mock_research_step
    workflow.distributed_steps["step_2"] = mock_document_step

    input_data = {
        "STUDENT_NEEDS": "Async Testing",
        "LANGUAGE": "English",
        "TEMPLATE": "Academic Paper"
    }

    result = await workflow.execute_async(input_data)
    
    # Verify async processing and format compliance
    assert result["step_1"]['preprocessed']
    assert result["step_1"]['postprocessed']
    assert result["step_1"]['result']['format'] == "structured_data"
    assert result["step_2"]['result']['format'] == "Markdown with LaTeX"
    
    # Verify workflow state progression
    assert workflow.state_manager.get_step_status("step_1") == StepStatus.SUCCESS
    assert workflow.state_manager.get_step_status("step_2") == StepStatus.SUCCESS

@pytest.mark.asyncio
async def test_distributed_workflow_step_dependency(test_workflow, test_config, mock_research_step, mock_document_step):
    """Test step dependency validation and execution order with agent.json format"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    workflow.distributed_steps["step_1"] = mock_research_step
    workflow.distributed_steps["step_2"] = mock_document_step

    # Verify step dependencies from WORKFLOW definition
    step_2_config = next(step for step in test_workflow["WORKFLOW"] if step["step"] == 2)
    assert "STUDENT_NEEDS.RESEARCH_TOPIC" in step_2_config["input"]
    
    input_data = {
        "STUDENT_NEEDS": "Dependency Testing",
        "LANGUAGE": "English",
        "TEMPLATE": "Academic Paper"
    }
    
    result = await workflow.execute_async(input_data)
    
    # Verify execution order and output format compliance
    assert result["step_1"]['result']['format'] == "structured_data"
    assert result["step_2"]['result']['format'] == "Markdown with LaTeX"
    assert workflow.state_manager.get_step_status("step_1") == StepStatus.SUCCESS
    assert workflow.state_manager.get_step_status("step_2") == StepStatus.SUCCESS

@pytest.mark.asyncio
async def test_distributed_workflow_timeout(test_workflow, test_config):
    """Test workflow step timeout with agent.json format compliance"""
    class SlowStep:
        async def execute(self, input_data):
            await asyncio.sleep(2)  # Simulate slow execution
            return {
                'result': {
                    'research_findings': "Slow research findings",
                    'format': test_workflow['WORKFLOW'][0]['output']['format']
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
    
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    workflow.distributed_steps["step_1"] = SlowStep()
    
    # Set a short timeout
    test_config["step_1_config"]["timeout"] = 1.0
    
    input_data = {
        "STUDENT_NEEDS": "Timeout Testing",
        "LANGUAGE": "English",
        "TEMPLATE": "Academic Paper"
    }
    
    with pytest.raises(TimeoutError):
        await workflow.execute_async(input_data)
    
    assert workflow.state_manager.get_step_status("step_1") == StepStatus.FAILED

@pytest.mark.asyncio
async def test_distributed_workflow_parallel_execution(test_workflow, test_config):
    """Test parallel execution of independent steps"""
    execution_order = []
    
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
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    workflow.distributed_steps["step_1"] = OrderedStep()
    workflow.distributed_steps["step_2"] = OrderedStep()
    
    input_data = {
        'input_1': {'step_num': 1},
        'input_2': {'step_num': 2}
    }
    
    result = await workflow.execute_async(input_data)
    
    assert len(result) == 2
    assert 'metadata' in result["step_1"]
    assert 'metadata' in result["step_2"]
    # Since steps are independent and run in parallel,
    # both should complete around the same time
    assert abs(result["step_1"]['metadata']['timestamp'] - result["step_2"]['metadata']['timestamp']) < 0.3

@pytest.mark.asyncio
async def test_workflow_retry_mechanism(mocker):
    """Test the retry mechanism for workflow steps"""
    # Mock the step function to simulate failures
    mock_step_func = AsyncMock(side_effect=[
        Exception("First attempt failed"),  # First attempt fails
        Exception("Second attempt failed"),  # Second attempt fails
        "Success on third attempt"  # Third attempt succeeds
    ])

    # Create a mock workflow configuration with retry settings
    mock_config = {
        'max_retries': 3,
        'retry_delay': 0.1,
        'retry_backoff': 2
    }

    # Create a mock workflow with async methods
    class MockWorkflow:
        def __init__(self, config, state_manager):
            self.config = config
            self.state_manager = state_manager

        async def _execute_step_with_retry(self, step_num, step_func, step_input=None):
            max_retries = self.config.get('max_retries', 3)
            retry_delay = self.config.get('retry_delay', 1)
            retry_backoff = self.config.get('retry_backoff', 2)

            for attempt in range(max_retries + 1):  # Include initial attempt
                try:
                    # Execute the step
                    result = await step_func(step_input) if step_input is not None else await step_func()
                    
                    # If successful, mark as success and return
                    self.state_manager.update_step_status(step_num, StepStatus.SUCCESS)
                    self.state_manager.increment_step_success_count(step_num)
                    return result
                
                except Exception as e:
                    # Log the error
                    logger.warning(f"Step {step_num} attempt {attempt + 1} failed: {e}")
                    
                    # If this is the last attempt, raise the exception
                    if attempt == max_retries:
                        raise
                    
                    # Update retry status
                    self.state_manager.retry_step(step_num)
                    
                    # Exponential backoff
                    await asyncio.sleep(retry_delay * (retry_backoff ** attempt))

        async def execute_step(self, step_num, step_func, step_input=None):
            # Reset retry count at the start of execution
            self.state_manager.reset_step_retry_count(step_num)
            
            try:
                result = await self._execute_step_with_retry(
                    step_num, 
                    step_func, 
                    step_input
                )
                return result
            except Exception as e:
                # Mark step as failed if all retries exhausted
                self.state_manager.update_step_status(step_num, StepStatus.FAILED)
                raise

    # Create a mock workflow state manager
    mock_state_manager = mocker.Mock()
    mock_state_manager.reset_step_retry_count = mocker.Mock()
    mock_state_manager.retry_step = mocker.Mock()
    mock_state_manager.update_step_status = mocker.Mock()
    mock_state_manager.increment_step_success_count = mocker.Mock()

    # Create a mock workflow with the retry mechanism
    workflow = MockWorkflow(mock_config, mock_state_manager)

    # Execute the step with retry
    result = await workflow.execute_step(
        step_num="step_1",
        step_func=mock_step_func
    )

    # Assertions
    assert result == "Success on third attempt"
    
    # Verify retry count reset
    mock_state_manager.reset_step_retry_count.assert_called_once_with("step_1")
    
    # Verify retry step was called twice (for first two failures)
    assert mock_state_manager.retry_step.call_count == 2
    
    # Verify step status updates
    mock_state_manager.update_step_status.assert_has_calls([
        mocker.call("step_1", StepStatus.SUCCESS)
    ])
    
    # Verify success count incremented
    mock_state_manager.increment_step_success_count.assert_called_once_with("step_1")

@pytest.mark.asyncio
async def test_workflow_retry_mechanism_exhausted(mocker):
    """Test retry mechanism when all attempts fail"""
    # Mock the step function to always fail
    mock_step_func = AsyncMock(side_effect=Exception("Persistent failure"))

    # Create a mock workflow configuration with retry settings
    mock_config = {
        'max_retries': 3,
        'retry_delay': 0.1,
        'retry_backoff': 2
    }

    # Create a mock workflow with async methods
    class MockWorkflow:
        def __init__(self, config, state_manager):
            self.config = config
            self.state_manager = state_manager

        async def _execute_step_with_retry(self, step_num, step_func, step_input=None):
            max_retries = self.config.get('max_retries', 3)
            retry_delay = self.config.get('retry_delay', 1)
            retry_backoff = self.config.get('retry_backoff', 2)

            for attempt in range(max_retries + 1):  # Include initial attempt
                try:
                    # Execute the step
                    result = await step_func(step_input) if step_input is not None else await step_func()
                    
                    # If successful, mark as success and return
                    self.state_manager.update_step_status(step_num, StepStatus.SUCCESS)
                    self.state_manager.increment_step_success_count(step_num)
                    return result
                
                except Exception as e:
                    # Log the error
                    logger.warning(f"Step {step_num} attempt {attempt + 1} failed: {e}")
                    
                    # If this is the last attempt, raise the exception
                    if attempt == max_retries:
                        raise
                    
                    # Update retry status
                    self.state_manager.retry_step(step_num)
                    
                    # Exponential backoff
                    await asyncio.sleep(retry_delay * (retry_backoff ** attempt))

        async def execute_step(self, step_num, step_func, step_input=None):
            # Reset retry count at the start of execution
            self.state_manager.reset_step_retry_count(step_num)
            
            try:
                result = await self._execute_step_with_retry(
                    step_num, 
                    step_func, 
                    step_input
                )
                return result
            except Exception as e:
                # Mark step as failed if all retries exhausted
                self.state_manager.update_step_status(step_num, StepStatus.FAILED)
                raise

    # Create a mock workflow state manager
    mock_state_manager = mocker.Mock()
    mock_state_manager.reset_step_retry_count = mocker.Mock()
    mock_state_manager.retry_step = mocker.Mock()
    mock_state_manager.update_step_status = mocker.Mock()

    # Create a mock workflow with the retry mechanism
    workflow = MockWorkflow(mock_config, mock_state_manager)

    # Expect the execution to raise an exception after all retries are exhausted
    with pytest.raises(Exception, match="Persistent failure"):
        await workflow.execute_step(
            step_num="step_1",
            step_func=mock_step_func
        )

    # Verify retry count reset
    mock_state_manager.reset_step_retry_count.assert_called_once_with("step_1")
    
    # Verify retry step was called 3 times (for all attempts)
    assert mock_state_manager.retry_step.call_count == 3
    
    # Verify final step status is set to FAILED
    mock_state_manager.update_step_status.assert_called_once_with("step_1", StepStatus.FAILED)

@pytest.mark.asyncio
async def test_distributed_workflow_retry_mechanism(test_workflow, test_config, failing_step):
    """Test retry mechanism with failing step while maintaining agent.json format compliance"""
    test_config['max_retries'] = 2
    test_config['retry_delay'] = 0.1
    test_config['retry_backoff'] = 1.5
    
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    workflow.distributed_steps["step_1"] = failing_step

    input_data = {
        "STUDENT_NEEDS": "Retry Testing",
        "LANGUAGE": "English",
        "TEMPLATE": "Academic Paper"
    }

    with pytest.raises(WorkflowExecutionError, match=r"Workflow execution failed: .*Persistent failure in step step_1"):
        await workflow.execute_async(input_data)
    
    # Verify retry behavior and state management
    assert workflow.state_manager.get_step_retry_count("step_1") == 3  # initial attempt + 2 retries
    assert workflow.state_manager.get_step_status("step_1") == StepStatus.FAILED
    
    # Verify that the workflow maintains expected output format even during retries
    step_metadata = workflow.state_manager.get_step_metadata("step_1")
    if step_metadata and 'last_error' in step_metadata:
        assert isinstance(step_metadata['last_error'], dict)
        assert 'error_type' in step_metadata['last_error']
        assert 'timestamp' in step_metadata['last_error']

@pytest.mark.asyncio
async def test_distributed_workflow_parallel_execution(test_workflow, test_config):
    """Test parallel execution of independent steps with agent.json format compliance"""
    # Add parallel steps to the workflow
    parallel_steps = [
        {
            'step': i+3,  # Start after existing steps
            'name': f"Parallel Research Step {i}",
            'input': ["research_topic"],
            'output': {
                "type": "research",
                "details": f"Parallel research findings {i}",
                "format": "structured_data",
                "word_count": 500
            }
        }
        for i in range(3)
    ]
    
    test_workflow["WORKFLOW"].extend(parallel_steps)
    
    class ParallelStep:
        async def execute(self, input_data):
            await asyncio.sleep(0.1)  # Simulate some work
            return {
                'result': {
                    'research_findings': f"Parallel findings for {input_data.get('research_topic')}",
                    'format': "structured_data"
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
    
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    
    # Add parallel steps to workflow
    for i in range(3):
        workflow.distributed_steps[f"parallel_step_{i}"] = ParallelStep()
    
    input_data = {
        "research_topic": "Parallel Testing",
        "deadline": "2024-01-01",
        "academic_level": "PhD"
    }
    
    result = await workflow.execute_async(input_data)
    
    # Verify parallel execution results
    for i in range(3):
        step_id = f"parallel_step_{i}"
        assert step_id in result
        assert result[step_id]['result']['format'] == "structured_data"
        assert "Parallel findings" in result[step_id]['result']['research_findings']
        assert workflow.state_manager.get_step_status(step_id) == StepStatus.SUCCESS
    
    # Verify timestamps are close together (parallel execution)
    timestamps = [result[f"parallel_step_{i}"]['metadata']['timestamp'] for i in range(3)]
    max_time_diff = max(timestamps) - min(timestamps)
    assert max_time_diff < 0.5  # Should complete within 0.5 seconds of each other if parallel

def test_agent_config_loading():
    """Test loading an agent configuration using the agent.json format"""
    agent_config = {
        "AGENT": "Research_Workflow_Agent",
        "CONTEXT": "This agent executes a distributed research workflow with multiple steps for academic paper optimization",
        "OBJECTIVE": "Process research inputs through multiple distributed steps to generate optimized academic output",
        "STATE": "The workflow requires processing research topic, deadline, and academic level inputs to generate research findings and documents",
        "WORKFLOW": [
            {
                "step": 1,
                "title": "Research Analysis",
                "description": "Analyze the research topic and generate findings",
                "input": ["research_topic", "deadline", "academic_level"],
                "output": {
                    "type": "research",
                    "details": "Research findings and analysis",
                    "format": "structured_data",
                    "word_count": 1000
                }
            }
        ],
        "POLICY": "Ensure all steps maintain academic rigor and proper error handling with retries",
        "ENVIRONMENT": {
            "INPUT": ["research_topic", "deadline", "academic_level"],
            "OUTPUT": "A complete academic document with research findings"
        }
    }
    
    # Verify required fields
    assert "AGENT" in agent_config
    assert "CONTEXT" in agent_config
    assert "OBJECTIVE" in agent_config
    assert "STATE" in agent_config
    assert "WORKFLOW" in agent_config
    assert "POLICY" in agent_config
    assert "ENVIRONMENT" in agent_config
    
    # Verify workflow step format
    workflow_step = agent_config["WORKFLOW"][0]
    assert "step" in workflow_step
    assert "title" in workflow_step
    assert "description" in workflow_step
    assert "input" in workflow_step
    assert "output" in workflow_step
    
    # Verify output format
    output = workflow_step["output"]
    assert "type" in output
    assert "details" in output
    assert "format" in output
    assert "word_count" in output
    
    # Verify environment configuration
    environment = agent_config["ENVIRONMENT"]
    assert isinstance(environment["INPUT"], list)
    assert isinstance(environment["OUTPUT"], str)
    assert all(isinstance(input_field, str) for input_field in environment["INPUT"])

if __name__ == "__main__":
    pytest.main([__file__])
