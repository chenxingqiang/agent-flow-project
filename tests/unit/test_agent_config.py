"""Test agent configuration."""

import pytest
from typing import Dict, Any
from pydantic import ValidationError
from agentflow.core.config import (
    AgentConfig,
    ModelConfig,
    WorkflowConfig,
    ConfigurationType
)
from agentflow.core.workflow_types import WorkflowStepType
from agentflow.agents.agent_types import AgentType, AgentMode

def test_agent_config_validation():
    """Test invalid configuration handling"""
    invalid_configs = [
        {
            "type": "invalid_type",
            "model": {
                "provider": "openai",
                "name": "gpt-4"
            }
        },
        {
            "type": "research",
            "model": {
                "provider": "invalid_provider",
                "name": "gpt-4"
            }
        }
    ]
    
    for config in invalid_configs:
        with pytest.raises(ValueError):
            AgentConfig(**config)

def test_agent_config_defaults():
    """Test default configuration values"""
    minimal_config = {
        "name": "Test Agent",
        "type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "id": "test-workflow-1",
            "name": "Research Workflow",
            "steps": [
                {
                    "id": "step1",
                    "name": "Research Step",
                    "type": WorkflowStepType.RESEARCH_EXECUTION.value,
                    "description": "Execute research step",
                    "config": {
                        "strategy": "standard",
                        "params": {"protocol": "federated"}
                    }
                }
            ]
        }
    }

    agent_config = AgentConfig(**minimal_config)
    
    # Check default values
    assert agent_config.name == "Test Agent"
    assert agent_config.type == "research"
    assert agent_config.workflow is not None
    assert agent_config.workflow.max_iterations == 10  # Default value from WorkflowConfig
    assert agent_config.workflow.timeout is None  # Default value from WorkflowConfig

def test_agent_config_serialization():
    """Test configuration serialization and deserialization"""
    config_data = {
        "name": "Test Agent",
        "type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.7
        },
        "workflow": {
            "id": "test-workflow-2",
            "name": "Research Workflow",
            "max_iterations": 5,
            "steps": [
                {
                    "id": "step1",
                    "name": "Research Step",
                    "type": WorkflowStepType.RESEARCH_EXECUTION.value,
                    "description": "Execute research step",
                    "config": {
                        "strategy": "standard",
                        "params": {"protocol": "federated"}
                    }
                }
            ]
        }
    }

    agent_config = AgentConfig(**config_data)
    
    # Convert to dictionary
    config_dict = agent_config.model_dump()
    
    assert config_dict['name'] == "Test Agent"
    assert config_dict['type'] == "research"
    assert config_dict['model']['provider'] == "openai"
    assert config_dict['workflow']['max_iterations'] == 5

def test_agent_config_complex_workflow():
    """Test complex workflow configuration"""
    config_data = {
        "name": "Test Agent",
        "type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "id": "test-workflow-3",
            "name": "Research Workflow",
            "max_iterations": 10,
            "steps": [
                {
                    "id": "step1",
                    "name": "Research Planning",
                    "type": WorkflowStepType.RESEARCH_EXECUTION.value,
                    "description": "Plan research execution",
                    "config": {
                        "strategy": "standard",
                        "params": {"protocol": "federated"}
                    }
                }
            ]
        }
    }

    agent_config = AgentConfig(**config_data)
    assert agent_config.name == "Test Agent"
    assert agent_config.type == "research"
    assert agent_config.workflow.max_iterations == 10
    assert len(agent_config.workflow.steps) == 1

if __name__ == "__main__":
    pytest.main([__file__])
