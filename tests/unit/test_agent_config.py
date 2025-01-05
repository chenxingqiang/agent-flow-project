import pytest
from typing import Dict, Any
from agentflow.core.config import AgentConfig, ModelConfig
from agentflow.core.workflow_types import WorkflowConfig
from agentflow.agents.agent import Agent
import pydantic_core

def test_agent_config_initialization():
    """Test comprehensive agent configuration initialization"""
    config_data = {
        "id": "research_agent_1",
        "name": "Research Agent",
        "type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4096
        },
        "workflow": {
            "max_iterations": 5,
            "logging_level": "INFO",
            "required_fields": ["research_topic"],
            "error_handling": {
                "missing_input_error": "No research topic provided"
            }
        }
    }

    agent_config = AgentConfig(**config_data)
    
    assert agent_config.id == "research_agent_1"
    assert agent_config.name == "Research Agent"
    assert agent_config.type == "research"
    assert agent_config.model.provider == "openai"
    assert agent_config.model.name == "gpt-4"
    assert agent_config.model.temperature == 0.7
    assert agent_config.model.max_tokens == 4096
    assert agent_config.workflow.max_iterations == 5
    assert agent_config.workflow.required_fields == ["research_topic"]

def test_agent_config_model_validation():
    """Test model configuration validation"""
    valid_models = [
        {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.5
        },
        {
            "provider": "anthropic",
            "name": "claude-2",
            "temperature": 0.7
        }
    ]

    for model_config in valid_models:
        config = {
            "type": "research",
            "model": model_config
        }
        agent_config = AgentConfig(**config)
        assert agent_config.model.provider == model_config["provider"]
        assert agent_config.model.name == model_config["name"]
        assert agent_config.model.temperature == model_config["temperature"]

def test_agent_config_workflow_policies():
    """Test workflow policies configuration"""
    config = {
        "type": "research",
        "workflow": {
            "max_iterations": 10,
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 1.0
            },
            "error_policy": {
                "ignore_warnings": True,
                "fail_fast": False
            }
        }
    }
    
    agent_config = AgentConfig(**config)
    assert agent_config.workflow.max_iterations == 10
    assert agent_config.workflow.retry_policy["max_retries"] == 3
    assert agent_config.workflow.retry_policy["retry_delay"] == 1.0

def test_agent_config_minimal_configuration():
    """Test minimal configuration requirements"""
    minimal_config = {
        "type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        }
    }
    
    agent_config = AgentConfig(**minimal_config)
    assert agent_config.type == "research"
    assert agent_config.model.provider == "openai"
    assert agent_config.model.name == "gpt-4"

def test_agent_config_invalid_configurations():
    """Test invalid configuration handling"""
    invalid_configs = [
        {
            "type": "invalid_type",
            "model": {"provider": "openai", "name": "gpt-4"}
        },
        {
            "type": "research",
            "model": {"provider": "invalid_provider", "name": "gpt-4"}
        }
    ]
    
    for config in invalid_configs:
        with pytest.raises(ValueError):
            AgentConfig(**config)

def test_agent_config_defaults():
    """Test default configuration values"""
    minimal_config = {
        "type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        }
    }

    agent_config = AgentConfig(**minimal_config)
    
    # Check default values
    assert agent_config.workflow is not None
    assert agent_config.workflow.max_iterations == 10  # Default value
    assert agent_config.workflow.timeout == 3600  # Default value
    assert agent_config.max_retries == 3  # Default value

def test_agent_config_serialization():
    """Test configuration serialization and deserialization"""
    config_data = {
        "type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.7
        },
        "workflow": {
            "max_iterations": 5,
            "logging_level": "DEBUG"
        }
    }

    agent_config = AgentConfig(**config_data)
    
    # Convert to dictionary
    config_dict = agent_config.model_dump()
    
    assert config_dict['type'] == "research"
    assert config_dict['model']['provider'] == "openai"
    assert config_dict['workflow']['max_iterations'] == 5

def test_agent_config_immutability():
    """Test configuration validation and immutability"""
    config_data = {
        "type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        }
    }

    agent_config = AgentConfig(**config_data)
    
    # Verify that the original configuration cannot be modified
    with pytest.raises(pydantic_core.ValidationError, match="Instance is frozen"):
        agent_config.name = "Modified Name"

def test_agent_config_complex_workflow_steps():
    """Test complex workflow steps configuration"""
    complex_workflow_config = {
        "id": "research_workflow_agent",
        "name": "Research Workflow Agent",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "max_iterations": 15,
            "steps": [
                {
                    "type": "research_planning",
                    "config": {
                        "depth": "comprehensive",
                        "sources": ["academic", "web"]
                    }
                },
                {
                    "type": "research_execution",
                    "config": {
                        "parallel_sources": True,
                        "max_source_depth": 3
                    }
                },
                {
                    "type": "document_generation",
                    "config": {
                        "format": "academic",
                        "citation_style": "APA"
                    }
                }
            ]
        }
    }

    agent_config = AgentConfig(**complex_workflow_config)
    
    assert len(agent_config.workflow.steps) == 3
    assert agent_config.workflow.max_iterations == 15
    assert agent_config.workflow.steps[0].type == "research_planning"
    assert agent_config.workflow.steps[1].type == "research_execution"
    assert agent_config.workflow.steps[2].type == "document_generation"

def test_agent_config_validation():
    """Test configuration validation and error handling"""
    # Test invalid configuration
    with pytest.raises(ValueError):
        AgentConfig(**{
            "agent_type": "invalid_type",
            "model": {
                "provider": "unsupported_provider"
            }
        })

def test_agent_config_complex_workflow():
    """Test complex workflow configuration"""
    config_data = {
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "max_iterations": 10,
            "logging_level": "INFO",
            "steps": [
                {
                    "type": "research_planning",
                    "config": {
                        "depth": "comprehensive"
                    }
                },
                {
                    "type": "document_generation",
                    "config": {
                        "format": "academic"
                    }
                }
            ]
        }
    }

    agent_config = AgentConfig(**config_data)
    
    assert len(agent_config.workflow.steps) == 2
    assert agent_config.workflow.steps[0].type == "research_planning"
    assert agent_config.workflow.steps[1].type == "document_generation"

if __name__ == "__main__":
    pytest.main([__file__])
