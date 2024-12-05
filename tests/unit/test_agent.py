import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from agentflow.core.config import AgentConfig, ModelConfig, WorkflowConfig
from agentflow.core.agent import Agent as AgentFlow

def test_agent_initialization(test_data_dir):
    """Test agent initialization"""
    config = AgentConfig(
        agent_type='research',
        model=ModelConfig(
            provider='openai',
            name='gpt-4',
            temperature=0.7
        ),
        workflow=WorkflowConfig(
            logging_level='INFO'
        )
    )
    agent = AgentFlow(config)
    
    assert agent is not None
    assert agent.config == config
    assert agent.config.agent_type == 'research'
    assert agent.config.model.provider == 'openai'

    agent = AgentFlow(
        str(test_data_dir / 'config.json'),
        str(test_data_dir / 'agent.json')
    )
    assert agent.config is not None
    assert agent.workflow_def is not None

@pytest.mark.parametrize("input_data,expected,mock_result", [
    (
        {"research_topic": "Test", "deadline": "2024-12-31", "academic_level": "PhD"},
        {"step_1": {"result": "Research findings for Test", "summary": "Analysis of Test", "recommendations": ["Recommendation 1", "Recommendation 2"]}},
        {"result": "Research findings for Test", "summary": "Analysis of Test", "recommendations": ["Recommendation 1", "Recommendation 2"]}
    ),
    (
        {"research_topic": "", "deadline": "", "academic_level": ""},
        {"step_1": {"result": "No research topic provided"}},
        {"result": "No research topic provided"}
    ),
])
def test_workflow_execution(test_data_dir, input_data, expected, mock_result):
    """Test workflow execution with different inputs"""
    config = AgentConfig(
        agent_type='research',
        model=ModelConfig(
            provider='openai',
            name='gpt-4',
            temperature=0.7
        ),
        workflow=WorkflowConfig(
            logging_level='INFO'
        )
    )
    agent = AgentFlow(config)
    
    with patch('agentflow.core.research_workflow.ResearchWorkflow.process_step', return_value=mock_result):
        result = agent.execute_workflow(input_data)
        assert result == expected

    agent = AgentFlow(
        str(test_data_dir / 'config.json'),
        str(test_data_dir / 'agent.json')
    )
    
    with patch('agentflow.core.research_workflow.ResearchWorkflow.process_step', return_value=mock_result):
        result = agent.execute_workflow(input_data)
        assert result == expected

@pytest.mark.asyncio
async def test_async_workflow_execution(test_data_dir):
    """Test asynchronous workflow execution"""
    config = AgentConfig(
        agent_type='research',
        model=ModelConfig(
            provider='openai',
            name='gpt-4',
            temperature=0.7
        ),
        workflow=WorkflowConfig(
            logging_level='INFO'
        )
    )
    agent = AgentFlow(config)
    
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    
    mock_result = {
        "result": "Research findings for Test Research",
        "summary": "Analysis of Test Research",
        "recommendations": ["Recommendation 1", "Recommendation 2"]
    }
    
    expected = {
        "step_1": mock_result
    }
    
    with patch('agentflow.core.research_workflow.ResearchWorkflow.process_step', return_value=mock_result):
        result = await agent.execute_workflow_async(input_data)
        assert result == expected

    agent = AgentFlow(
        str(test_data_dir / 'config.json'),
        str(test_data_dir / 'agent.json')
    )
    
    with patch('agentflow.core.research_workflow.ResearchWorkflow.process_step', return_value=mock_result):
        result = await agent.execute_workflow_async(input_data)
        assert result == expected

def test_error_handling(test_data_dir):
    """Test error handling"""
    config = AgentConfig(
        agent_type='research',
        model=ModelConfig(
            provider='openai',
            name='gpt-4',
            temperature=0.7
        ),
        workflow=WorkflowConfig(
            logging_level='INFO'
        )
    )
    agent = AgentFlow(config)
    
    # Test with non-research workflow
    agent.workflow_def = {
        'WORKFLOW': [
            {
                'step': 1,
                'name': 'Document Step',
                'input': ['document_id', 'content'],
                'output': {'type': 'document'}
            }
        ]
    }
    
    with pytest.raises(ValueError, match="Missing or empty inputs: document_id, content"):
        agent.execute_workflow({"invalid_input": "value"})

    agent = AgentFlow(
        str(test_data_dir / 'config.json'),
        str(test_data_dir / 'agent.json')
    )
    
    # Test with non-research workflow
    agent.workflow_def = {
        'WORKFLOW': [
            {
                'step': 1,
                'name': 'Document Step',
                'input': ['document_id', 'content'],
                'output': {'type': 'document'}
            }
        ]
    }
    
    with pytest.raises(ValueError, match="Missing or empty inputs: document_id, content"):
        agent.execute_workflow({"invalid_input": "value"})