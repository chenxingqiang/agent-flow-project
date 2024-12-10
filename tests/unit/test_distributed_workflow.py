import pytest
import ray
import asyncio
import time
import logging
import json
from typing import Dict, Any
from unittest.mock import AsyncMock, patch

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
                "step_id": "step_1",
                "type": "research",
                "title": "Extract Details from Student Inputs",
                "description": "Analyze the STUDENT_NEEDS, LANGUAGE, and TEMPLATE variables to understand the student's background, goals, and constraints.",
                "input": ["STUDENT_NEEDS", "LANGUAGE", "TEMPLATE"],
                "output": {
                    "type": "analysis",
                    "details": "Summarized student profile and requirements",
                    "format": "plain text",
                    "word_count": 200
                },
                "agent_config": {}
            },
            {
                "step": 2,
                "step_id": "step_2",
                "type": "research",
                "title": "Propose Innovative Ideas",
                "description": "Generate 3-5 innovative ideas tailored to the student's research topic, each with an evaluation of innovation, feasibility, and academic value.",
                "input": ["WORKFLOW.step_1"],
                "output": {
                    "type": "ideas",
                    "details": "Detailed list of innovative ideas with evaluations",
                    "format": "Markdown with LaTeX",
                    "word_count": 1000
                },
                "agent_config": {}
            },
            {
                "step": 3,
                "step_id": "step_3",
                "type": "research",
                "title": "Create Implementation Plans",
                "description": "Develop detailed implementation plans for the prioritized ideas, using the TEMPLATE for formatting and integrating LaTeX for technical content.",
                "input": ["WORKFLOW.step_1", "WORKFLOW.step_2"],
                "output": {
                    "type": "plan",
                    "details": "Step-by-step implementation for 1-2 prioritized ideas",
                    "format": "Markdown with LaTeX",
                    "word_count": 1200
                },
                "agent_config": {}
            }
        ]
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
            "step_id": "step_1",
            "max_retries": 3,
            "timeout": 30,
            "preprocessors": [],
            "postprocessors": []
        },
        "step_2_config": {
            "step_id": "step_2", 
            "max_retries": 3,
            "timeout": 30,
            "preprocessors": [],
            "postprocessors": []
        },
        "step_3_config": {
            "step_id": "step_3", 
            "max_retries": 3,
            "timeout": 30,
            "preprocessors": [],
            "postprocessors": []
        }
    }

@pytest.fixture
def mock_research_step():
    """Create a mock research step"""
    @ray.remote
    class MockResearchStep:
        def __init__(self, step_id: str, config: Dict[str, Any] = None):
            self.step_id = step_id
            self.config = config or {}
            self.step_number = 1
            self.input = self.config.get('input', [])
            self.output = self.config.get('output', {})
        
        def execute(self, input_data):
            return {
                'result': {
                    'student_analysis': "Detailed student needs analysis",
                    'research_topic': "Advanced Machine Learning Techniques",
                    'format': "plain text"
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
    return MockResearchStep

@pytest.fixture
def mock_document_step():
    """Create a mock document step"""
    @ray.remote
    class MockDocumentStep:
        def __init__(self, step_id: str, config: Dict[str, Any] = None):
            self.step_id = step_id
            self.config = config or {}
            self.step_number = 2
            self.input = self.config.get('input', [])
            self.output = self.config.get('output', {})
        
        def execute(self, input_data):
            return {
                'result': {
                    'research_ideas': [
                        {
                            'title': 'Innovative ML Approach',
                            'description': 'A novel method for improving model interpretability',
                            'innovation_score': 8.5,
                            'feasibility_score': 7.2
                        },
                        {
                            'title': 'Cross-Domain Learning',
                            'description': 'Transferring knowledge across different machine learning domains',
                            'innovation_score': 9.0,
                            'feasibility_score': 6.8
                        }
                    ],
                    'format': "Markdown with LaTeX"
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
    return MockDocumentStep

@pytest.fixture
def mock_implementation_step():
    """Create a mock implementation step"""
    @ray.remote
    class MockImplementationStep:
        def __init__(self, step_id: str, config: Dict[str, Any] = None):
            self.step_id = step_id
            self.config = config or {}
            self.step_number = 3
            self.input = self.config.get('input', [])
            self.output = self.config.get('output', {})
        
        def execute(self, input_data):
            return {
                'result': {
                    'implementation_plans': [
                        {
                            'idea': 'Innovative ML Approach',
                            'steps': [
                                'Literature review',
                                'Develop prototype',
                                'Experimental validation'
                            ],
                            'resources_needed': ['GPU cluster', 'Research dataset']
                        }
                    ],
                    'format': "Markdown with LaTeX"
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
    return MockImplementationStep

@pytest.fixture
def failing_step():
    """Create a failing step"""
    @ray.remote
    class FailingStep:
        def __init__(self, step_id: str, config: Dict[str, Any] = None):
            self.step_id = step_id
            self.config = config or {}
            self.step_number = 1
            self.input = self.config.get('input', [])
            self.output = self.config.get('output', {})
        
        def execute(self, input_data):
            raise StepExecutionError("Step execution failed")
    return FailingStep

@pytest.mark.asyncio
async def test_distributed_workflow_initialization(test_workflow, test_config):
    """Test distributed workflow initialization"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    
    # Check workflow configuration
    assert workflow.workflow_config == test_workflow
    assert workflow.config == test_config
    
    # Check required fields
    assert set(workflow.required_fields) == {"STUDENT_NEEDS", "LANGUAGE", "TEMPLATE"}

@pytest.mark.asyncio
async def test_distributed_workflow_execution(test_workflow, test_config, mock_research_step, mock_document_step, mock_implementation_step):
    """Test distributed workflow execution with agent.json aligned format"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    
    # Add mock steps
    workflow.distributed_steps["step_1"] = mock_research_step.remote(
        step_id="step_1",
        config=test_config['step_1_config']
    )
    workflow.distributed_steps["step_2"] = mock_document_step.remote(
        step_id="step_2",
        config=test_config['step_2_config']
    )
    workflow.distributed_steps["step_3"] = mock_implementation_step.remote(
        step_id="step_3",
        config=test_config['step_3_config']
    )
    
    # Prepare input data
    input_data = {
        "STUDENT_NEEDS": "API Testing",
        "LANGUAGE": "English",
        "TEMPLATE": "Academic Paper"
    }
    
    # Execute workflow
    result = await workflow.execute_async(input_data)
    
    # Validate result
    assert result is not None
    assert isinstance(result, dict)
    assert 'output' in result
    
    # Check step-specific outputs
    output = result.get('output', {})
    assert 'details' in output
    assert 'format' in output
    assert 'result' in output
    
    # Verify workflow state
    assert workflow.state_manager.get_step_status("step_1") == StepStatus.SUCCESS
    assert workflow.state_manager.get_step_status("step_2") == StepStatus.SUCCESS
    assert workflow.state_manager.get_step_status("step_3") == StepStatus.SUCCESS

@pytest.mark.asyncio
async def test_workflow_input_validation(test_workflow, test_config):
    """Test input validation for the distributed workflow"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    
    # Test with missing required inputs
    with pytest.raises(ValueError, match="Missing required inputs"):
        await workflow.execute({
            "LANGUAGE": "English",
            "TEMPLATE": "Research Proposal"
        })
    
    # Test with incomplete input data
    with pytest.raises(ValueError, match="Missing required inputs"):
        await workflow.execute({
            "STUDENT_NEEDS": "Research support",
            "LANGUAGE": "English"
        })

@pytest.mark.asyncio
async def test_workflow_retry_mechanism(test_workflow, test_config, failing_step):
    """Test workflow retry mechanism"""
    # Modify workflow to include retry configuration
    test_config.update({
        "max_retries": 2,
        "retry_delay": 0.1,
        "retry_backoff": 2.0
    })

    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    
    # Replace the first step with a failing step to test retry
    workflow.distributed_steps["step_1"] = failing_step.remote(
        step_id="step_1", 
        config=test_config['step_1_config']
    )
    
    # Prepare input data
    input_data = {
        "STUDENT_NEEDS": "Advanced ML Research",
        "LANGUAGE": "English", 
        "TEMPLATE": "Academic Research Proposal"
    }
    
    # Add logging to track retry attempts
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Expect retry mechanism to handle the failure
    with pytest.raises(WorkflowExecutionError, match="Persistent failure in step step_1"):
        logger.info("Starting workflow execution")
        await workflow.execute_async(input_data)
    
    # Verify that the step was retried and failed
    assert workflow.state_manager.get_step_status("step_1") == StepStatus.FAILED

@pytest.mark.asyncio
async def test_distributed_workflow_error_handling(test_workflow, test_config, failing_step):
    """Test error handling in distributed workflow"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)

    # Add failing step
    workflow.distributed_steps["step_1"] = failing_step.remote(
        step_id="step_1",
        config=test_config['step_1_config']
    )
    
    # Prepare input data
    input_data = {
        "STUDENT_NEEDS": "API Testing",
        "LANGUAGE": "English",
        "TEMPLATE": "Academic Paper"
    }
    
    # Expect workflow execution error
    with pytest.raises(WorkflowExecutionError, match="Persistent failure in step step_1"):
        await workflow.execute_async(input_data)

@pytest.mark.asyncio
async def test_workflow_step_dependencies(test_workflow, test_config, mock_research_step, mock_document_step, mock_implementation_step):
    """Test workflow step dependencies and data passing"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    
    # Set up workflow steps
    workflow.distributed_steps["step_1"] = mock_research_step.remote(
        step_id="step_1", 
        config=test_config['step_1_config']
    )
    workflow.distributed_steps["step_2"] = mock_document_step.remote(
        step_id="step_2", 
        config=test_config['step_2_config']
    )
    workflow.distributed_steps["step_3"] = mock_implementation_step.remote(
        step_id="step_3", 
        config=test_config['step_3_config']
    )
    
    # Prepare input data
    input_data = {
        "STUDENT_NEEDS": "Advanced Machine Learning Research",
        "LANGUAGE": "English", 
        "TEMPLATE": "Academic Research Proposal"
    }
    
    # Execute workflow
    result = await workflow.execute_async(input_data)
    
    # Verify result structure
    assert result is not None
    assert isinstance(result, dict)
    assert 'output' in result
    
    # Verify workflow state
    assert workflow.state_manager.get_step_status("step_1") == StepStatus.SUCCESS
    assert workflow.state_manager.get_step_status("step_2") == StepStatus.SUCCESS
    assert workflow.state_manager.get_step_status("step_3") == StepStatus.SUCCESS
