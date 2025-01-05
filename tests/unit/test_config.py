import pytest
import json
import os
from pathlib import Path
from agentflow.core.config_manager import ConfigManager, AgentConfig

def test_config_loading(test_data_dir):
    """Test configuration loading"""
    # Create a config manager with test data directory
    config_manager = ConfigManager(str(test_data_dir))
    
    # Load test agent config
    with open(test_data_dir / 'config.json') as f:
        config_data = json.load(f)
        
    # Create and save agent config
    agent_config = AgentConfig(
        id="test-agent",
        name=config_data["AGENT"]["name"],
        type=config_data["AGENT"]["type"],
        version=config_data["AGENT"]["version"],
        workflow_path=config_data["AGENT"]["workflow_path"]
    )
    config_manager.save_agent_config(agent_config)
    
    # Load and verify agent config
    loaded_config = config_manager.load_agent_config("test-agent")
    assert loaded_config is not None
    assert loaded_config.name == config_data["AGENT"]["name"]
    assert loaded_config.type == config_data["AGENT"]["type"]
    assert loaded_config.version == config_data["AGENT"]["version"]
    assert loaded_config.workflow_path == config_data["AGENT"]["workflow_path"]

def test_config_validation(test_data_dir):
    """Test configuration validation"""
    config_manager = ConfigManager(str(test_data_dir))
    
    # Create an agent config with invalid type
    with pytest.raises(ValueError):
        AgentConfig(
            id="invalid-agent",
            name="Invalid Agent",
            type="invalid_type"
        )

def test_variable_extraction(test_data_dir):
    """Test variable extraction"""
    config_manager = ConfigManager(str(test_data_dir))
    
    # Create and save agent config with variables
    agent_config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        domain_config={
            "test_var": "test_value"
        }
    )
    config_manager.save_agent_config(agent_config)
    
    # Load and verify variables
    loaded_config = config_manager.load_agent_config("test-agent")
    assert loaded_config is not None
    assert loaded_config.domain_config["test_var"] == "test_value"

@pytest.mark.parametrize("config_update,expected_error", [
    ({"invalid_key": "value"}, ValueError),
    ({"variables": {"invalid_type": {"type": "invalid"}}}, ValueError),
])
def test_config_update_validation(test_data_dir, config_update, expected_error):
    """Test configuration update validation"""
    config_manager = ConfigManager(str(test_data_dir))
    
    # Create base agent config
    agent_config = AgentConfig(
        id="test-agent",
        name="Test Agent"
    )
    config_manager.save_agent_config(agent_config)
    
    # Try to update with invalid config
    with pytest.raises(expected_error):
        loaded_config = config_manager.load_agent_config("test-agent")
        for key, value in config_update.items():
            setattr(loaded_config, key, value)