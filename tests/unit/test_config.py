"""Test configuration functionality."""

import pytest
import json
import os
from pathlib import Path
from typing import Dict, Any, cast
from agentflow.core.config_manager import ConfigManager, AgentConfig
from agentflow.core.config import AgentConfig
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path

def test_config_loading(test_data_dir):
    """Test configuration loading"""
    # Create test config data
    config_data = {
        "AGENT": {
            "name": "test_agent",
            "type": "research",
            "version": "1.0.0",
            "workflow": {
                "id": "test-workflow",
                "name": "Test Workflow",
                "steps": [
                    {
                        "id": "step1",
                        "name": "test_step",
                        "type": "transform",
                        "description": "Test transformation step",
                        "config": {
                            "strategy": "custom",
                            "params": {}
                        }
                    }
                ]
            }
        }
    }
    
    # Save config data
    with open(test_data_dir / 'config.json', 'w') as f:
        json.dump(config_data, f)
    
    # Create a config manager with test data directory
    config_manager = ConfigManager(str(test_data_dir))
    
    # Create and save agent config
    agent_config = AgentConfig(
        id="test-agent",
        name=config_data["AGENT"]["name"],
        type=config_data["AGENT"]["type"],
        version=config_data["AGENT"]["version"],
        workflow=config_data["AGENT"]["workflow"]
    )
    config_manager.save_agent_config(agent_config)
    
    # Load and verify agent config
    loaded_config = cast(AgentConfig, config_manager.load_agent_config("test-agent"))
    assert loaded_config.id == "test-agent"
    assert loaded_config.name == "test_agent"
    assert loaded_config.type == "research"
    assert loaded_config.version == "1.0.0"
    assert isinstance(loaded_config.workflow, WorkflowConfig)
    assert loaded_config.workflow.id == "test-workflow"
    assert loaded_config.workflow.name == "Test Workflow"
    assert len(loaded_config.workflow.steps) == 1

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
        },
        workflow={
            "id": "test-workflow",
            "name": "Test Workflow",
            "steps": [
                {
                    "id": "step1",
                    "name": "test_step",
                    "type": "transform",
                    "description": "Test transformation step",
                    "config": {
                        "strategy": "custom",
                        "params": {}
                    }
                }
            ]
        }
    )
    config_manager.save_agent_config(agent_config)
    
    # Load and verify variables
    loaded_config = cast(AgentConfig, config_manager.load_agent_config("test-agent"))
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