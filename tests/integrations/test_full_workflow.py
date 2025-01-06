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
from agentflow.core.exceptions import WorkflowExecutionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import yaml
from datetime import datetime
from agentflow.core.config import ConfigurationType
from pathlib import Path
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_openai():
    """Mock OpenAI API responses."""
    with patch('openai.ChatCompletion.acreate') as mock:
        mock.return_value = AsyncMock(return_value={
            'choices': [{
                'message': {
                    'content': 'Test research output',
                    'role': 'assistant'
                }
            }],
            'usage': {
                'total_tokens': 100
            }
        })
        yield mock

def create_test_workflow_file(path: str) -> None:
    """Create a test workflow YAML file."""
    workflow_data = {
        'name': 'Test Workflow',
        'description': 'Test workflow for integration testing',
        'steps': [
            {
                'id': 'step-1',
                'name': 'Research Step',
                'type': 'research',
                'config': {
                    'strategy': 'research',
                    'params': {'depth': 'comprehensive'}
                }
            },
            {
                'id': 'step-2',
                'name': 'Document Generation Step',
                'type': 'document',
                'config': {
                    'strategy': 'document',
                    'params': {'format': 'academic'}
                }
            }
        ]
    }
    
    with open(path, 'w') as f:
        yaml.dump(workflow_data, f)

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

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path

@pytest.mark.asyncio
async def test_complete_workflow(test_data_dir, mock_openai):
    """Test complete workflow execution."""
    # Create test workflow file
    workflow_path = str(test_data_dir / "test_workflow.yaml")
    create_test_workflow_file(workflow_path)

    # Create workflow config
    workflow_config = WorkflowConfig(
        id='test-workflow-1',
        name='Test Workflow',
        max_iterations=5,
        timeout=30,
        logging_level='INFO',
        distributed=False,
        steps=create_test_steps()
    )

    try:
        # Create workflow executor
        executor = WorkflowExecutor(workflow_config)
        
        # Execute workflow
        result = await executor.execute(create_test_context())
        
        # Verify results
        assert result is not None
        assert isinstance(result, dict)
        assert 'step-1' in result
        assert 'step-2' in result
        
        # Verify research step output
        assert 'research_output' in result['step-1']
        assert isinstance(result['step-1']['research_output'], str)
        
        # Verify document generation step output
        assert 'document' in result['step-2']
        assert isinstance(result['step-2']['document'], str)
        
        # Verify workflow metadata
        assert 'metadata' in result
        assert 'execution_time' in result['metadata']
        assert 'total_tokens' in result['metadata']
        
    except WorkflowExecutionError as e:
        pytest.fail(f"Workflow execution failed: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {str(e)}")

def test_workflow_validation():
    """Test workflow configuration validation."""
    # Test with invalid step configuration
    invalid_steps = [
        WorkflowStep(
            id='invalid-step',
            name='Invalid Step',
            type='invalid_type',
            config=StepConfig(
                strategy='invalid',
                params={}
            )
        )
    ]
    
    with pytest.raises(ValueError, match="Invalid step type"):
        WorkflowConfig(
            id='invalid-workflow',
            name='Invalid Workflow',
            max_iterations=5,
            timeout=30,
            steps=invalid_steps
        )

def test_workflow_context_validation():
    """Test workflow context validation."""
    # Test with missing required fields
    invalid_context = {
        "academic_level": "PhD"  # Missing required research_topic
    }
    
    workflow_config = WorkflowConfig(
        id='test-workflow',
        name='Test Workflow',
        max_iterations=5,
        timeout=30,
        steps=create_test_steps()
    )
    
    executor = WorkflowExecutor(workflow_config)
    
    with pytest.raises(ValueError, match="Missing required field: research_topic"):
        executor.validate_context(invalid_context)