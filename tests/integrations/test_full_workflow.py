"""Test full workflow execution."""

import pytest
import logging
from typing import Dict, Any, List
from agentflow.agents.agent import Agent
from agentflow.agents.agent_types import AgentType, AgentConfig, AgentMode, ModelConfig
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, StepConfig
from agentflow.core.templates import WorkflowTemplate, TemplateParameter
from agentflow.core.workflow_executor import WorkflowExecutor
from agentflow.core.processors.transformers import FilterProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import yaml
from datetime import datetime
from agentflow.core.config import ConfigurationType

@pytest.fixture
def mock_openai(mocker):
    """Mock OpenAI client"""
    mock_response = mocker.Mock()
    mock_response.choices = [mocker.Mock(message=mocker.Mock(content="test_output"))]
    mock_response.usage = mocker.Mock(total_tokens=100)
    
    async def async_create(**kwargs):
        return mock_response
    
    mock_completions = mocker.Mock()
    mock_completions.create = async_create
    
    mock_chat = mocker.Mock()
    mock_chat.completions = mock_completions
    
    mock_client = mocker.Mock()
    mock_client.chat = mock_chat
    
    mock_openai = mocker.patch('openai.OpenAI', return_value=mock_client)
    return mock_openai

def create_test_workflow_file(path: str):
    """Create a test workflow file."""
    workflow_config = {
        'WORKFLOW': {
            'id': f"workflow-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'name': 'Test Workflow',
            'version': '1.0.0',
            'max_iterations': 10,
            'timeout': 3600,
            'distributed': False,
            'steps': [
                {
                    'id': 'step-1',
                    'type': 'research',
                    'name': 'Research Step',
                    'description': 'Perform research on the given topic',
                    'input': ['research_topic', 'deadline', 'academic_level'],
                    'output': {
                        'research_findings': 'str',
                        'research_summary': 'str'
                    },
                    'config': {
                        'strategy': 'research',
                        'params': {
                            'depth': 'comprehensive'
                        }
                    }
                },
                {
                    'id': 'step-2',
                    'type': 'document',
                    'name': 'Document Generation Step',
                    'description': 'Generate document from research findings',
                    'input': ['step_1.research_findings'],
                    'output': {
                        'document_content': 'str',
                        'document_metadata': 'dict'
                    },
                    'config': {
                        'strategy': 'document',
                        'params': {
                            'format': 'academic'
                        }
                    }
                }
            ]
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(workflow_config, f)

def create_test_steps():
    """Create test workflow steps."""
    return [
        WorkflowStep(
            id='step-1',
            name='Research Step',
            type='research',
            config=StepConfig(
                strategy='research',
                params={'depth': 'comprehensive'}
            )
        ),
        WorkflowStep(
            id='step-2',
            name='Document Generation Step',
            type='document',
            config=StepConfig(
                strategy='document',
                params={'format': 'academic'}
            )
        )
    ]

def create_test_context():
    """Create test workflow context."""
    return {
        "research_topic": "AI Ethics in Distributed Systems",
        "deadline": "2024-12-31",
        "academic_level": "PhD",
        "research_methodology": "Systematic Literature Review"
    }

@pytest.mark.asyncio
async def test_complete_workflow(test_data_dir, mock_openai):
    """Test complete workflow execution"""
    workflow_path = str(test_data_dir / "test_workflow.yaml")
    create_test_workflow_file(workflow_path)

    workflow_config = WorkflowConfig(
        id='test-workflow-1',
        name='Test Workflow',
        max_iterations=5,
        logging_level='INFO',
        distributed=False,
        steps=create_test_steps()
    )

    # Create workflow executor
    executor = WorkflowExecutor(workflow_config)
    
    # Execute workflow
    result = await executor.execute(create_test_context())
    
    # Verify results
    assert result is not None
    assert isinstance(result, dict)
    assert 'step-1' in result
    assert 'step-2' in result

@pytest.mark.asyncio
async def test_workflow_with_document_generation(test_data_dir, mock_openai):
    """Test workflow with document generation"""
    pytest.skip("Document generation test not implemented yet")

@pytest.mark.asyncio
async def test_async_workflow(test_data_dir, mock_openai):
    """Test asynchronous workflow"""
    workflow_path = str(test_data_dir / "test_workflow.yaml")
    create_test_workflow_file(workflow_path)

    workflow_config = WorkflowConfig(
        id='test-workflow-2',
        name='Test Workflow',
        max_iterations=5,
        logging_level='INFO',
        distributed=True,
        timeout=300,  # 5 minutes
        steps=create_test_steps()
    )

    # Create workflow executor
    executor = WorkflowExecutor(workflow_config)
    
    # Execute workflow
    result = await executor.execute(create_test_context())
    
    # Verify results
    assert result is not None
    assert isinstance(result, dict)
    assert 'step-1' in result
    assert 'step-2' in result

@pytest.mark.asyncio
async def test_full_workflow(tmp_path, mock_openai):
    """Test a complete workflow from start to finish"""
    workflow_path = str(tmp_path / "test_workflow.yaml")
    create_test_workflow_file(workflow_path)

    workflow_config = WorkflowConfig(
        id='test-workflow-3',
        name='Test Workflow',
        max_iterations=7,
        logging_level='INFO',
        distributed=False,
        steps=create_test_steps()
    )

    # Create workflow executor
    executor = WorkflowExecutor(workflow_config)
    
    # Execute workflow
    result = await executor.execute(create_test_context())
    
    # Verify results
    assert result is not None
    assert isinstance(result, dict)
    assert 'step-1' in result
    assert 'step-2' in result

@pytest.mark.asyncio
async def test_full_async_workflow(tmp_path, mock_openai):
    """Test a complete asynchronous workflow"""
    workflow_path = str(tmp_path / "test_workflow.yaml")
    create_test_workflow_file(workflow_path)

    workflow_config = WorkflowConfig(
        id='test-workflow-4',
        name='Test Workflow',
        max_iterations=7,
        logging_level='INFO',
        distributed=True,
        timeout=300,  # 5 minutes
        steps=create_test_steps()
    )

    # Create workflow executor
    executor = WorkflowExecutor(workflow_config)
    
    # Execute workflow
    result = await executor.execute(create_test_context())
    
    # Verify results
    assert result is not None
    assert isinstance(result, dict)
    assert 'step-1' in result
    assert 'step-2' in result

@pytest.mark.asyncio
async def test_error_handling(tmp_path, mock_openai):
    """Test error handling in workflow execution"""
    workflow_path = str(tmp_path / "test_workflow.yaml")
    create_test_workflow_file(workflow_path)

    workflow_config = WorkflowConfig(
        id='test-workflow-5',
        name='Test Workflow',
        max_iterations=7,
        logging_level='INFO',
        distributed=False,
        steps=create_test_steps()
    )

    # Create workflow executor
    executor = WorkflowExecutor(workflow_config)
    
    # Test with empty context
    with pytest.raises(Exception):
        await executor.execute({})

    # Test with invalid context
    with pytest.raises(Exception):
        await executor.execute(None)

    # Test with valid context
    result = await executor.execute(create_test_context())
    assert result is not None
    assert isinstance(result, dict)
    assert 'step-1' in result
    assert 'step-2' in result