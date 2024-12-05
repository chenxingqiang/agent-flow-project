import pytest
from typing import Dict, Any
from agentflow.core.workflow import BaseWorkflow
from agentflow.core.research_workflow import ResearchWorkflow

def test_workflow_input_validation(test_workflow, test_config):
    """Test workflow input validation"""
    workflow = ResearchWorkflow(workflow_def=test_workflow, config=test_config)
    
    # Test with valid input
    valid_input = {
        "research_topic": "Test Topic",
        "deadline": "2024-12-31",
        "academic_level": "PhD"
    }
    result = workflow.execute(valid_input)
    assert result is not None
    assert "step_1" in result
    assert "step_2" in result
    
    # Test with minimal input should raise an error
    minimal_input = {
        "research_topic": "Minimal Topic"
    }
    with pytest.raises(ValueError, match="Missing or empty inputs: deadline, academic_level"):
        workflow.execute(minimal_input)
    
    # Test with invalid input
    invalid_input = {}
    with pytest.raises(ValueError, match="Empty input data"):
        workflow.execute(invalid_input)