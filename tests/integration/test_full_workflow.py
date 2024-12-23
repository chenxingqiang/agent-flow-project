import pytest
from unittest.mock import patch, MagicMock
from agentflow.core.agent import Agent
from agentflow.core.config import AgentConfig, ModelConfig, WorkflowConfig
from pathlib import Path
import os

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

def test_complete_workflow(test_data_dir, mock_openai):
    """Test complete workflow execution"""
    config = AgentConfig(
        agent_type='research',
        model=ModelConfig(
            provider='openai',
            name='gpt-4',
            temperature=0.7
        ),
        workflow=WorkflowConfig(
            max_iterations=5,
            logging_level='INFO',
            distributed=False
        )
    )
    agent = Agent(config)
    
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    
    results = agent.execute_workflow(input_data)
    
    assert results is not None
    assert "research_output" in results
    assert "step_1" in results
    assert len(results) >= 2

@pytest.mark.skip(reason="Document generation not implemented for Pydantic v2")
def test_workflow_with_document_generation(tmp_path, mock_openai):
    """Test workflow with document generation"""
    config = AgentConfig(
        agent_type='research',
        model=ModelConfig(
            provider='openai',
            name='gpt-4',
            temperature=0.7
        ),
        workflow=WorkflowConfig(
            max_iterations=5,
            logging_level='INFO',
            distributed=False
        )
    )
    agent = Agent(config)
    
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    
    results = agent.execute_workflow(input_data)
    
    # Generate documents in different formats
    formats = ["markdown", "pdf", "docx"]
    for fmt in formats:
        output_path = str(tmp_path / f"output.{fmt}")
        doc_path = agent.generate_output_document(results, fmt, output_path)
        assert Path(doc_path).exists()

@pytest.mark.asyncio
async def test_async_workflow(test_data_dir, mock_openai):
    """Test asynchronous workflow"""
    config = AgentConfig(
        agent_type='research',
        model=ModelConfig(
            provider='openai',
            name='gpt-4',
            temperature=0.7
        ),
        workflow=WorkflowConfig(
            max_iterations=5,
            logging_level='INFO',
            distributed=True,
            timeout=300  # 5 minutes
        )
    )
    agent = Agent(config)
    
    input_data = {
        "research_topic": "Test Research",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    
    results = await agent.execute_workflow_async(input_data)
    
    assert results is not None 

@pytest.mark.integration
def test_full_workflow(tmp_path, mock_openai):
    """Test a complete workflow from start to finish"""
    config = AgentConfig(
        agent_type='research',
        model=ModelConfig(
            provider='openai',
            name='gpt-4',
            temperature=0.7
        ),
        workflow=WorkflowConfig(
            max_iterations=7,
            logging_level='INFO',
            distributed=False
        )
    )
    agent = Agent(config)

    research_input = {
        "research_topic": "AI Ethics in Distributed Systems",
        "deadline": "2024-12-31",
        "academic_level": "PhD",
        "research_methodology": "Systematic Literature Review"
    }

    # Execute the full workflow
    workflow_result = agent.execute_workflow(research_input)
    
    # Validate workflow result structure
    assert workflow_result is not None
    assert "research_output" in workflow_result
    assert isinstance(workflow_result["research_output"], dict)
    
    # Validate research output content
    research_output = workflow_result["research_output"]
    assert "result" in research_output
    assert isinstance(research_output["result"], str)
    assert len(research_output["result"]) > 0
    
    # Verify OpenAI API was called
    mock_openai.assert_called()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_async_workflow(tmp_path, mock_openai):
    """Test a complete asynchronous workflow"""
    config = AgentConfig(
        agent_type='research',
        model=ModelConfig(
            provider='openai',
            name='gpt-4',
            temperature=0.7
        ),
        workflow=WorkflowConfig(
            max_iterations=7,
            logging_level='INFO',
            distributed=True,
            timeout=300  # 5 minutes
        )
    )
    agent = Agent(config)

    research_input = {
        "research_topic": "Machine Learning Fairness",
        "deadline": "2024-12-31",
        "academic_level": "PhD",
        "research_methodology": "Empirical Analysis"
    }

    # Execute the full async workflow
    workflow_result = await agent.execute_workflow_async(research_input)
    
    # Validate workflow result structure
    assert workflow_result is not None
    assert "research_output" in workflow_result
    assert isinstance(workflow_result["research_output"], dict)
    
    # Validate research output content
    research_output = workflow_result["research_output"]
    assert "result" in research_output
    assert isinstance(research_output["result"], str)
    assert len(research_output["result"]) > 0
    
    # Verify OpenAI API was called
    mock_openai.assert_called()

@pytest.mark.integration
def test_error_handling(tmp_path, mock_openai):
    """Test error handling in workflow execution"""
    config = AgentConfig(
        agent_type='research',
        model=ModelConfig(
            provider='openai',
            name='gpt-4',
            temperature=0.7
        ),
        workflow=WorkflowConfig(
            max_iterations=7,
            logging_level='INFO',
            distributed=False
        )
    )
    agent = Agent(config)

    # Test with missing required fields
    invalid_input = {
        "research_topic": "Test Research"
        # Missing deadline and academic_level
    }
    
    with pytest.raises(ValueError) as exc_info:
        agent.execute_workflow(invalid_input)
    assert "Missing required fields" in str(exc_info.value)

    # Test with empty input
    with pytest.raises(ValueError) as exc_info:
        agent.execute_workflow({})
    assert "Input data must be a non-empty dictionary" in str(exc_info.value)