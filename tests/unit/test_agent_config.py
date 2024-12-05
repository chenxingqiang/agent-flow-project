import pytest
from agentflow.core.config import AgentConfig
from agentflow.core.agent import Agent

def test_agent_config_initialization():
    """Test basic agent configuration initialization"""
    config_data = {
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.7
        },
        "workflow": {
            "max_iterations": 5,
            "logging_level": "INFO"
        }
    }

    agent_config = AgentConfig(**config_data)
    
    assert agent_config.agent_type == "research"
    assert agent_config.model.provider == "openai"
    assert agent_config.model.name == "gpt-4"
    assert agent_config.model.temperature == 0.7
    assert agent_config.workflow.max_iterations == 5

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
    """Test configuration immutability"""
    config_data = {
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        }
    }

    agent_config = AgentConfig(**config_data)
    
    # Attempt to modify should raise an error
    with pytest.raises(TypeError):
        agent_config.agent_type = "different_type"

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
