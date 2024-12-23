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

import ell

@pytest.fixture
def test_workflow():
    """Create a test workflow configuration"""
    return {
        "ENVIRONMENT": {
            "INPUT": {
                "STUDENT_NEEDS",
                "LANGUAGE",
                "TEMPLATE"
            }
        },
        "WORKFLOW": {
            "step_1": {
                "step": 1,
                "title": "Extract Details from Student Inputs",
                "description": "Analyze the STUDENT_NEEDS, LANGUAGE, and TEMPLATE variables to understand the student's background, goals, and constraints.",
                "input": ["STUDENT_NEEDS", "LANGUAGE", "TEMPLATE"],
                "output": {
                    "type": "analysis",
                    "details": "Summarized student profile and requirements",
                    "format": "plain text",
                    "word_count": 200
                }
            },
            "step_2": {
                "step": 2,
                "title": "Propose Innovative Ideas",
                "description": "Generate 3-5 innovative ideas tailored to the student's research topic, each with an evaluation of innovation, feasibility, and academic value.",
                "input": ["STUDENT_NEEDS.RESEARCH_TOPIC", "LANGUAGE.TYPE"],
                "output": {
                    "type": "ideas",
                    "details": "Detailed list of innovative ideas with evaluations",
                    "format": "Markdown with LaTeX",
                    "word_count": 1000
                }
            },
            "step_3": {
                "step": 3,
                "title": "Create Implementation Plans",
                "description": "Develop detailed implementation plans for the prioritized ideas, using the TEMPLATE for formatting and integrating LaTeX for technical content.",
                "input": ["TEMPLATE", "WORKFLOW.step_1.output"],
                "output": {
                    "type": "plan",
                    "details": "Step-by-step implementation for 1-2 prioritized ideas",
                    "format": "Markdown with LaTeX",
                    "word_count": 1200
                }
            },
            "step_4": {
                "step": 4,
                "title": "Develop Weekly Timeline",
                "description": "Construct a detailed weekly timeline for experiments, analysis, and writing, aligned with the student's DEADLINE.",
                "input": ["STUDENT_NEEDS.DEADLINE", "WORKFLOW.step_2.output"],
                "output": {
                    "type": "timeline",
                    "details": "Weekly schedule of tasks and milestones",
                    "format": "Markdown table",
                    "word_count": 300
                }
            },
            "step_5": {
                "step": 5,
                "title": "Provide Recommendations",
                "description": "Conclude with recommendations for tools, references, and resources to enhance the research and writing process.",
                "input": ["WORKFLOW.step_2.output", "WORKFLOW.step_3.output"],
                "output": {
                    "type": "recommendations",
                    "details": "List of tools, references, and optimization suggestions",
                    "format": "Markdown",
                    "word_count": 500
                }
            }
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
            "timeout": 30,
            "preprocessors": [],
            "postprocessors": []
        },
        "step_2_config": {
            "max_retries": 3,
            "timeout": 30,
            "preprocessors": [],
            "postprocessors": []
        },
        "step_3_config": {
            "max_retries": 3,
            "timeout": 30,
            "preprocessors": [],
            "postprocessors": []
        },
        "step_4_config": {
            "max_retries": 3,
            "timeout": 30,
            "preprocessors": [],
            "postprocessors": []
        },
        "step_5_config": {
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
    """Create a step that fails a specified number of times before succeeding"""
    @ray.remote
    class FailingStep:
        def __init__(self, step_id: str, config: Dict[str, Any]):
            self.step_id = step_id
            self.config = config
            self.attempts = 0
            self.max_retries = config.get('max_retries', 3)

        def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            # Increment attempt counter
            self.attempts += 1
            
            # Always fail with a specific error message
            error_msg = f"Step {self.step_id} execution failed (attempt {self.attempts}/{self.max_retries})"
            raise StepExecutionError(error_msg)

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

    # Check step configurations
    workflow_steps = test_workflow["WORKFLOW"]
    if isinstance(workflow_steps, dict):
        workflow_steps = [
            {"step_id": step_id, **step_config}
            for step_id, step_config in workflow_steps.items()
        ]

    for step in workflow_steps:
        step_id = step.get('step_id', f"step_{step.get('step', 0)}")
        step_config = workflow.config.get(f'{step_id}_config')
        assert step_config is not None
        assert step_config.get('timeout') == 30
        assert isinstance(step_config.get('preprocessors'), list)
        assert isinstance(step_config.get('postprocessors'), list)
        assert step_config.get('max_retries') == test_config.get('max_retries', 3)
        assert step_config.get('retry_delay') == test_config.get('retry_delay', 1.0)
        assert step_config.get('retry_backoff') == test_config.get('retry_backoff', 2.0)

@pytest.mark.asyncio
async def test_distributed_workflow_execution(test_workflow, test_config, mock_research_step, mock_document_step, mock_implementation_step):
    """Test distributed workflow execution with agent.json aligned format"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)

    # Add mock steps for all workflow steps
    workflow_steps = test_workflow["WORKFLOW"]
    if isinstance(workflow_steps, dict):
        workflow_steps = [
            {"step_id": step_id, **step_config}
            for step_id, step_config in workflow_steps.items()
        ]

    for step in workflow_steps:
        step_id = step.get('step_id', f"step_{step.get('step', 0)}")
        workflow.distributed_steps[step_id] = mock_research_step.remote(
            step_id=step_id,
            config=test_config[f"{step_id}_config"]
        )

    # Prepare input data
    input_data = {
        "STUDENT_NEEDS": {
            "RESEARCH_TOPIC": "Advanced ML Research",
            "DEADLINE": "2024-12-31"
        },
        "LANGUAGE": {
            "TYPE": "English"
        },
        "TEMPLATE": "Academic Research Proposal"
    }

    # Execute workflow
    result = await workflow.execute_async(input_data)

    # Validate result
    assert result is not None
    assert isinstance(result, dict)
    assert 'output' in result

    # Check step-specific outputs
    output = result.get('output', {})
    assert isinstance(output, dict)
    assert len(output) > 0

    # Verify workflow state
    status = workflow.state_manager.get_step_status("step_1")
    assert status.value == StepStatus.SUCCESS.value

    # Verify step results
    step_1_result = workflow.state_manager.get_step_result("step_1")
    assert step_1_result is not None
    assert isinstance(step_1_result, dict)

    # Verify workflow completion
    workflow_status = workflow.state_manager.get_workflow_status()
    assert workflow_status.value == WorkflowStatus.COMPLETED.value

@pytest.mark.asyncio
async def test_workflow_input_validation(test_workflow, test_config):
    """Test input validation for the distributed workflow"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)

    # Verify the required fields are set correctly
    assert workflow.required_fields == {"STUDENT_NEEDS", "LANGUAGE", "TEMPLATE"}

    # Test with missing required inputs
    with pytest.raises(WorkflowExecutionError) as excinfo:
        await workflow.execute_async({
            "LANGUAGE": {"TYPE": "English"},
            "TEMPLATE": "Research Proposal",
            # Missing STUDENT_NEEDS field
        })
    assert "Missing required input: STUDENT_NEEDS" in str(excinfo.value)

    # Successful case with all required inputs
    result = await workflow.execute_async({
        "LANGUAGE": {"TYPE": "English"},
        "TEMPLATE": "Research Proposal", 
        "STUDENT_NEEDS": {
            "DEADLINE": "2024-12-31", 
            "RESEARCH_TOPIC": "Advanced AI Research"
        }
    })
    assert result is not None

@pytest.mark.asyncio
async def test_workflow_retry_mechanism(test_workflow, test_config, failing_step):
    """Test retry mechanism in distributed workflow"""
    # Configure retry settings
    test_config['step_1_config']['max_retries'] = 3
    test_config['step_1_config']['retry_delay'] = 0.1
    test_config['step_1_config']['retry_backoff'] = 2.0

    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    workflow.distributed_steps["step_1"] = failing_step.remote(
        step_id="step_1",
        config=test_config['step_1_config']
    )

    input_data = {
        "STUDENT_NEEDS": {
            "RESEARCH_TOPIC": "Advanced ML Research",
            "DEADLINE": "2024-12-31"
        },
        "LANGUAGE": {
            "TYPE": "English"
        },
        "TEMPLATE": "Academic Research Proposal"
    }

    # Test that workflow raises WorkflowExecutionError after max retries
    with pytest.raises(WorkflowExecutionError) as excinfo:
        await workflow.execute_async(input_data)
    
    error_str = str(excinfo.value)
    assert "Step step_1 failed after 3 retries" in error_str

@pytest.mark.asyncio
async def test_workflow_error_handling(test_workflow, test_config, failing_step):
    """Test error handling in distributed workflow"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)
    workflow.distributed_steps["step_1"] = failing_step.remote(
        step_id="step_1",
        config=test_config['step_1_config']
    )

    input_data = {
        "STUDENT_NEEDS": {
            "RESEARCH_TOPIC": "API Testing",
            "DEADLINE": "2024-12-31"
        },
        "LANGUAGE": {
            "TYPE": "English"
        },
        "TEMPLATE": "Academic Paper"
    }

    # Test that workflow raises WorkflowExecutionError after max retries
    with pytest.raises(WorkflowExecutionError) as excinfo:
        await workflow.execute_async(input_data)
    
    expected_error = "Workflow execution failed: Step step_1 failed after 3 retries: Step step_1 execution failed (attempt 3/3)"
    assert str(excinfo.value) == expected_error

@pytest.mark.asyncio
async def test_workflow_step_dependencies(test_workflow, test_config, mock_research_step, mock_document_step, mock_implementation_step):
    """Test workflow step dependencies and data passing"""
    workflow = ResearchDistributedWorkflow(workflow_config=test_workflow, config=test_config)

    # Set up workflow steps
    workflow_steps = test_workflow["WORKFLOW"]
    if isinstance(workflow_steps, dict):
        workflow_steps = [
            {"step_id": step_id, **step_config}
            for step_id, step_config in workflow_steps.items()
        ]

    # Add mock steps for all workflow steps
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
    # Add a mock step for step_4 and step_5
    workflow.distributed_steps["step_4"] = mock_implementation_step.remote(
        step_id="step_4",
        config=test_config.get('step_4_config', {})
    )
    workflow.distributed_steps["step_5"] = mock_implementation_step.remote(
        step_id="step_5",
        config=test_config.get('step_5_config', {})
    )

    # Prepare input data
    input_data = {
        "STUDENT_NEEDS": "Advanced Machine Learning Research",
        "LANGUAGE": "English",
        "TEMPLATE": "Academic Research Proposal"
    }

    # Execute workflow
    results = await workflow.execute_async(input_data)

    # Verify results
    assert len(results.get('output', {})) > 0

@pytest.mark.asyncio
async def test_distributed_workflow_with_llm(test_workflow, test_config):
    """
    Test distributed workflow execution with LLM integration using ell.simple

    This test demonstrates how to incorporate an LLM model into the workflow
    execution process, simulating real-world AI-driven workflow scenarios.
    """
    try:
        import ell.store
        has_ell = True
    except ImportError:
        has_ell = False
        pytest.skip("ell.store not available - skipping LLM test")

    # Prepare input data with LLM-specific context
    input_data = {
        "STUDENT_NEEDS": {
            "RESEARCH_TOPIC": "AI Ethics in Machine Learning",
            "LANGUAGE": "English",
            "ACADEMIC_LEVEL": "PhD",
            "DEADLINE": "2024-12-31"
        },
        "LLM_CONFIG": {
            "provider": "ell.simple",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }

    # Create distributed workflow with LLM configuration
    workflow = ResearchDistributedWorkflow(
        workflow_config=test_workflow, 
        config={
            **test_config,
            "llm_model": input_data["LLM_CONFIG"]
        }
    )

    if has_ell:
        # Execute workflow with LLM tracing
        with ell.store.trace() as trace:
            try:
                result = await workflow.execute_async(input_data)
                assert result is not None
                assert isinstance(result, dict)
                assert 'output' in result

                # Verify trace contains LLM interactions
                assert len(trace.invocations) > 0
                for invocation in trace.invocations:
                    assert invocation.version is not None
                    assert invocation.inputs is not None
                    assert invocation.outputs is not None

            except Exception as e:
                pytest.fail(f"Workflow execution failed: {str(e)}")
