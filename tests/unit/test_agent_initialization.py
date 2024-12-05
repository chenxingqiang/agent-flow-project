import pytest
import ray
from agentflow.core.config import AgentConfig
from agentflow.core.agent import Agent

@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing"""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()

def test_agent_initialization(ray_context):
    """Test basic agent initialization"""
    config_data = {
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.5
        },
        "workflow": {
            "max_iterations": 5,
            "logging_level": "INFO"
        }
    }

    agent_config = AgentConfig(**config_data)
    agent = Agent(agent_config)

    assert agent is not None
    assert agent.config == agent_config
    assert agent.agent_type == "research"
    assert agent.model.temperature == 0.5
    assert agent.workflow.max_iterations == 5

def test_agent_distributed_initialization(ray_context):
    """Test distributed agent initialization"""
    config_data = {
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "max_iterations": 10,
            "distributed": True
        }
    }

    agent_config = AgentConfig(**config_data)
    agent = Agent(agent_config)

    assert agent.is_distributed
    assert agent.ray_actor is not None
    assert isinstance(agent.ray_actor, ray.actor.ActorHandle)

def test_agent_invalid_initialization():
    """Test invalid agent initialization"""
    with pytest.raises(ValueError, match="Unsupported agent type"):
        config_data = {
            "agent_type": "unsupported_type",
            "model": {
                "provider": "openai",
                "name": "gpt-4"
            },
            "workflow": {
                "max_iterations": 5
            }
        }
        agent_config = AgentConfig(**config_data)
        Agent(agent_config)

def test_agent_model_configuration():
    """Test agent model configuration"""
    config_data = {
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 1000
        },
        "workflow": {
            "max_iterations": 5
        }
    }

    agent_config = AgentConfig(**config_data)
    agent = Agent(agent_config)

    assert agent.model.temperature == 0.5
    assert agent.model.max_tokens == 1000
    assert agent.model.provider == "openai"
    assert agent.model.name == "gpt-4"

def test_agent_workflow_configuration():
    """Test agent workflow configuration"""
    config_data = {
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "max_iterations": 7,
            "logging_level": "DEBUG",
            "timeout": 300,
            "distributed": False
        }
    }

    agent_config = AgentConfig(**config_data)
    agent = Agent(agent_config)

    assert agent.workflow.max_iterations == 7
    assert agent.workflow.logging_level == "DEBUG"
    assert agent.workflow.timeout == 300
    assert not agent.is_distributed

def test_agent_file_initialization(test_data_dir):
    """Test agent initialization from files"""
    agent = Agent(
        str(test_data_dir / 'config.json'),
        str(test_data_dir / 'workflow.json')
    )
    
    assert agent.config is not None
    assert agent.workflow_def is not None
    assert agent.config.agent_type == "research"
    assert "WORKFLOW" in agent.workflow_def

def test_agent_state_initialization():
    """Test agent state initialization"""
    config_data = {
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "max_iterations": 5
        }
    }

    agent_config = AgentConfig(**config_data)
    agent = Agent(agent_config)

    # Test initial state
    assert agent.state == {}
    assert agent.workflow_def is None
    assert agent.ray_actor is None

def test_agent_invalid_model_provider():
    """Test agent initialization with invalid model provider"""
    with pytest.raises(ValueError, match="Unsupported model provider"):
        config_data = {
            "agent_type": "research",
            "model": {
                "provider": "invalid_provider",
                "name": "gpt-4"
            },
            "workflow": {
                "max_iterations": 5
            }
        }
        agent_config = AgentConfig(**config_data)
        Agent(agent_config)

def test_agent_invalid_workflow_config():
    """Test agent initialization with invalid workflow config"""
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        config_data = {
            "agent_type": "research",
            "model": {
                "provider": "openai",
                "name": "gpt-4"
            },
            "workflow": {
                "max_iterations": 0  # Invalid: must be >= 1
            }
        }
        agent_config = AgentConfig(**config_data)
        Agent(agent_config)

def test_agent_invalid_file_path():
    """Test agent initialization with invalid file path"""
    with pytest.raises(FileNotFoundError):
        Agent("nonexistent_config.json")

if __name__ == "__main__":
    pytest.main([__file__])
