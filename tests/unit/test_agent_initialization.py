import pytest
import ray
from agentflow.core.config import AgentConfig
from agentflow.core.agent import Agent
from pathlib import Path
import json

@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing"""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()

def test_agent_initialization(ray_context):
    """Test basic agent initialization"""
    config_data = {
        "AGENT": {
            "name": "TestAgent",
            "type": "research",
            "version": "1.0.0"
        },
        "MODEL": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.5
        },
        "WORKFLOW": {
            "max_iterations": 5,
            "logging_level": "INFO"
        }
    }

    agent = Agent(config_data)
    assert agent.name == "TestAgent"
    assert agent.version == "1.0.0"
    assert agent.model is not None
    assert agent.workflow is not None

def test_agent_distributed_initialization(ray_context):
    """Test distributed agent initialization"""
    config_data = {
        "AGENT": {
            "name": "DistributedAgent",
            "type": "research",
            "version": "1.0.0"
        },
        "MODEL": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "WORKFLOW": {
            "max_iterations": 10,
            "distributed": True
        }
    }

    agent = Agent(config_data)
    assert agent.name == "DistributedAgent"
    assert agent.is_distributed == True

def test_agent_model_configuration():
    """Test agent model configuration"""
    config_data = {
        "AGENT": {
            "name": "ModelAgent",
            "type": "research",
            "version": "1.0.0"
        },
        "MODEL": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 1000
        },
        "WORKFLOW": {
            "max_iterations": 5
        }
    }

    agent = Agent(config_data)
    assert agent.model.provider == "openai"
    assert agent.model.name == "gpt-4"
    assert agent.model.temperature == 0.5

def test_agent_workflow_configuration():
    """Test agent workflow configuration"""
    config_data = {
        "AGENT": {
            "name": "WorkflowAgent",
            "type": "research",
            "version": "1.0.0"
        },
        "MODEL": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "WORKFLOW": {
            "max_iterations": 7,
            "logging_level": "DEBUG",
            "timeout": 300,
            "distributed": False
        }
    }

    agent = Agent(config_data)
    assert agent.workflow.max_iterations == 7
    assert agent.workflow.logging_level == "DEBUG"
    assert agent.workflow.timeout == 300

def test_agent_file_initialization(test_data_dir):
    """Test agent initialization from files"""
    config_path = test_data_dir / 'config.json'
    workflow_path = test_data_dir / 'workflow.json'

    # Create test config files if they don't exist
    if not config_path.exists():
        config_data = {
            "AGENT": {
                "name": "FileAgent",
                "type": "research",
                "version": "1.0.0"
            },
            "MODEL": {
                "provider": "openai",
                "name": "gpt-4"
            },
            "WORKFLOW": {
                "max_iterations": 5
            }
        }
        config_path.write_text(json.dumps(config_data))

    if not workflow_path.exists():
        workflow_data = {
            "WORKFLOW": {
                "max_iterations": 10,
                "distributed": False
            }
        }
        workflow_path.write_text(json.dumps(workflow_data))

    agent = Agent(str(config_path), str(workflow_path))
    assert agent.name == "FileAgent"
    assert agent.workflow.max_iterations == 10

def test_agent_state_initialization():
    """Test agent state initialization"""
    config_data = {
        "AGENT": {
            "name": "StateAgent",
            "type": "research",
            "version": "1.0.0"
        },
        "MODEL": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "WORKFLOW": {
            "max_iterations": 5
        }
    }

    agent = Agent(config_data)
    assert hasattr(agent, 'state')
    assert isinstance(agent.state, dict)

def test_agent_invalid_initialization():
    """Test agent initialization with invalid config"""
    with pytest.raises(ValueError):
        Agent({})

def test_agent_invalid_model_provider():
    """Test agent initialization with invalid model provider"""
    config_data = {
        "AGENT": {
            "name": "InvalidAgent",
            "type": "research"
        },
        "MODEL": {
            "provider": "invalid_provider",
            "name": "invalid_model"
        },
        "WORKFLOW": {
            "max_iterations": 5
        }
    }

    agent = Agent(config_data)
    assert agent.model.provider == "invalid_provider"

def test_agent_invalid_workflow_config():
    """Test agent initialization with invalid workflow config"""
    config_data = {
        "AGENT": {
            "name": "InvalidWorkflowAgent",
            "type": "research"
        },
        "MODEL": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "WORKFLOW": {
            "max_iterations": 0  # Invalid value
        }
    }

    with pytest.raises(ValueError):
        Agent(config_data)

def test_agent_invalid_file_path():
    """Test agent initialization with invalid file path"""
    with pytest.raises(FileNotFoundError):
        Agent("invalid_config.json")

if __name__ == "__main__":
    pytest.main([__file__])
