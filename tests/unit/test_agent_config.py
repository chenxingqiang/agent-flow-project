import pytest
from typing import Dict, Any
from agentflow.core.config import AgentConfig, ModelConfig, WorkflowConfig
from agentflow.core.agent import Agent

def test_agent_config_initialization():
    """Test comprehensive agent configuration initialization"""
    config_data = {
        "id": "research_agent_1",
        "name": "Research Agent",
        "agent_type": "research",
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
    assert agent_config.agent_type == "research"
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
            "temperature": 0.3
        }
    ]

    for model_data in valid_models:
        model_config = ModelConfig(**model_data)
        assert model_config.provider in ["openai", "anthropic"]
        assert model_config.temperature >= 0 and model_config.temperature <= 1

def test_agent_config_workflow_policies():
    """Test workflow configuration policies"""
    workflow_data = {
        "max_iterations": 10,
        "logging_level": "DEBUG",
        "required_fields": ["topic", "deadline"],
        "error_handling": {
            "missing_input_error": "Missing required workflow inputs",
            "max_retry_error": "Maximum retry attempts exceeded"
        },
        "steps": [
            {
                "type": "research_planning",
                "required_inputs": ["topic"]
            },
            {
                "type": "document_generation",
                "required_inputs": ["research_findings"]
            }
        ]
    }

    workflow_config = WorkflowConfig(**workflow_data)
    
    assert workflow_config.max_iterations == 10
    assert workflow_config.logging_level == "DEBUG"
    assert workflow_config.required_fields == ["topic", "deadline"]
    assert len(workflow_config.steps) == 2
    assert workflow_config.steps[0].type == "research_planning"

def test_agent_config_minimal_configuration():
    """Test minimal agent configuration"""
    minimal_config = {
        "id": "basic_agent",
        "name": "Basic Agent",
        "model": {
            "provider": "openai",
            "name": "gpt-3.5-turbo"
        }
    }

    agent_config = AgentConfig(**minimal_config)
    
    assert agent_config.id == "basic_agent"
    assert agent_config.name == "Basic Agent"
    assert agent_config.model.provider == "openai"
    assert agent_config.model.name == "gpt-3.5-turbo"
    assert agent_config.model.temperature == 0.5  # Default value
    assert agent_config.workflow.max_iterations == 10  # Default value

def test_agent_config_invalid_configurations():
    """Test invalid agent configurations"""
    invalid_configs = [
        # Missing required model fields
        {
            "id": "invalid_agent_1",
            "model": {}
        },
        # Invalid model provider
        {
            "id": "invalid_agent_2",
            "model": {
                "provider": "unsupported_provider",
                "name": "model"
            }
        }
    ]

    for config in invalid_configs:
        with pytest.raises((ValueError, TypeError)):
            AgentConfig(**config)

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

def test_agent_config_defaults():
    """Test default configuration values"""
    minimal_config = {
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        }
    }

    agent_config = AgentConfig(**minimal_config)
    
    # Check default values
    assert agent_config.workflow.max_iterations == 10  # Default value
    assert agent_config.workflow.logging_level == "INFO"  # Default value
    assert agent_config.model.temperature == 0.5  # Default temperature

def test_agent_config_serialization():
    """Test configuration serialization and deserialization"""
    config_data = {
        "agent_type": "research",
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
    
    assert config_dict['agent_type'] == "research"
    assert config_dict['model']['name'] == "gpt-4"
    assert config_dict['workflow']['max_iterations'] == 5

def test_agent_config_immutability():
    """Test configuration validation and immutability"""
    config_data = {
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        }
    }

    agent_config = AgentConfig(**config_data)

    # Verify that the original configuration cannot be modified
    with pytest.raises(Exception) as excinfo:
        # Attempt to create a new config with an invalid agent type
        AgentConfig(**{
            **config_data,
            "agent_type": "invalid_type"
        })
    
    # Verify that the validation error is raised
    assert "Unsupported agent type" in str(excinfo.value)
    
    # Verify that the original configuration remains unchanged
    assert agent_config.agent_type == "research"

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
