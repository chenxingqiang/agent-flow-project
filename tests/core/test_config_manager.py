"""
Tests for configuration management
"""

import os
import json
import pytest
import tempfile
from agentflow.core.config import AgentConfig, ModelConfig, ConfigurationType, WorkflowConfig, load_global_config
from agentflow.core.config_manager import ConfigManager
from agentflow.core.workflow_types import WorkflowStep, WorkflowStepType, StepConfig

@pytest.fixture
def config_manager():
    """Create a temporary config manager"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(tmpdir)
        yield manager

def test_agent_config_save_and_load(config_manager):
    """Test saving and loading agent configurations"""
    # Create test agent config
    agent_config = AgentConfig(
        id="test-agent-1",
        name="Test Agent",
        description="A test agent configuration",
        type=ConfigurationType.GENERIC,
        model=ModelConfig(
            name="test-model",
            provider="default"
        ),
        system_prompt="You are a test agent",
        workflow=WorkflowConfig(
            id="test-workflow",
            name="Test Workflow",
            steps=[
                WorkflowStep(
                    id="step-1",
                    name="step_1",
                    type=WorkflowStepType.TRANSFORM,
                    description="A test step",
                    config=StepConfig(
                        strategy="standard",
                        params={}
                    )
                )
            ]
        )
    )
    
    # Save configuration
    config_manager.save_agent_config(agent_config)
    
    # Load configuration
    loaded_config = config_manager.load_agent_config("test-agent-1")
    
    assert loaded_config is not None
    assert loaded_config.id == "test-agent-1"
    assert loaded_config.name == "Test Agent"
    assert loaded_config.model.name == "test-model"

def test_workflow_config_save_and_load(config_manager):
    """Test saving and loading workflow configurations"""
    # Create test workflow config
    workflow_config = WorkflowConfig(
        id="test-workflow-1",
        name="Test Workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="A test transform step",
                config=StepConfig(
                    strategy="standard",
                    params={}
                )
            )
        ]
    )
    
    # Save configuration
    config_manager.save_workflow_config(workflow_config)
    
    # Load configuration
    loaded_config = config_manager.load_workflow_config("test-workflow-1")
    
    assert loaded_config is not None
    assert loaded_config.id == workflow_config.id
    assert loaded_config.name == workflow_config.name
    assert len(loaded_config.steps) == len(workflow_config.steps)

def test_list_configurations(config_manager):
    """Test listing configurations"""
    configs = config_manager.list_configurations()
    assert isinstance(configs, dict)
    assert "agents" in configs
    assert "workflows" in configs

def test_delete_configurations(config_manager):
    """Test deleting configurations"""
    config_manager.delete_configuration("test-workflow-1", "workflow")
    assert config_manager.load_workflow_config("test-workflow-1") is None

def test_export_and_import_config(config_manager):
    """Test exporting and importing configurations"""
    # Create and save a test workflow configuration
    workflow_config = WorkflowConfig(
        id="test-workflow-1",
        name="Test Workflow 1",
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="A test transform step",
                config=StepConfig(
                    strategy="feature_engineering",
                    params={
                        "method": "standard",
                        "with_mean": True,
                        "with_std": True
                    }
                )
            )
        ]
    )
    config_manager.save_workflow_config(workflow_config)

    # Now try to export and import
    config = config_manager.export_configuration("test-workflow-1", "workflow")
    assert config["id"] == "test-workflow-1"
    assert config["name"] == "Test Workflow 1"
    assert len(config["steps"]) == 1

    # Import the configuration
    imported_config = config_manager.import_configuration(config, "workflow")
    assert imported_config.id == "test-workflow-1"
    assert imported_config.name == "Test Workflow 1"
    assert len(imported_config.steps) == 1

def test_agent_config_from_dict():
    """Test loading agent configuration from dictionary."""
    config_dict = {
        "id": "test-agent",
        "name": "test_agent",
        "type": "generic",
        "description": "A test agent",
        "model": {
            "name": "gpt-4",
            "provider": "openai"
        },
        "workflow": {
            "id": "test-workflow",
            "name": "Test Workflow",
            "steps": [
                {
                    "id": "step-1",
                    "name": "step_1",
                    "type": "transform",
                    "description": "A test transform step",
                    "config": {
                        "strategy": "standard",
                        "params": {}
                    }
                }
            ]
        }
    }
    config = AgentConfig.model_validate(config_dict)
    assert config.name == "test_agent"
    assert config.model.name == "gpt-4"

def test_workflow_config_from_dict():
    """Test loading workflow configuration from dictionary."""
    config_dict = {
        "id": "test-workflow",
        "name": "test_workflow",
        "max_iterations": 5,
        "timeout": 30,
        "steps": [
            {
                "id": "step-1",
                "name": "step_1",
                "type": "transform",
                "description": "A test transform step",
                "config": {
                    "strategy": "standard",
                    "params": {}
                }
            }
        ]
    }
    config = WorkflowConfig.model_validate(config_dict)
    assert config.name == "test_workflow"
    assert len(config.steps) == 1

def test_model_config_from_dict():
    """Test loading model configuration from dictionary."""
    config_dict = {
        "name": "gpt-4",
        "provider": "openai"
    }
    config = ModelConfig.model_validate(config_dict)
    assert config.name == "gpt-4"
    assert config.provider == "openai"

def test_global_config():
    """Test loading global configuration."""
    global_config = load_global_config()
    assert 'global_' in global_config
    assert 'logging_level' in global_config['global_']
