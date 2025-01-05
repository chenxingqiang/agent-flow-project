"""
Tests for configuration management
"""

import os
import json
import pytest
import tempfile

from agentflow.core.config import AgentConfig, ModelConfig, ConfigurationType, WorkflowConfig
from agentflow.core.config_manager import ConfigManager

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
        system_prompt="You are a test agent"
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
        description="A test workflow configuration",
        agents=[
            AgentConfig(
                id="agent-1",
                name="Agent 1",
                description="First test agent",
                type=ConfigurationType.GENERIC,
                model=ModelConfig(
                    name="test-model-1",
                    provider="default"
                ),
                system_prompt="You are agent 1"
            )
        ],
        processors=[],
        connections=[]
    )
    
    # Save configuration
    config_manager.save_workflow_config(workflow_config)
    
    # Load configuration
    loaded_config = config_manager.load_workflow_config("test-workflow-1")
    
    assert loaded_config is not None
    assert loaded_config.id == "test-workflow-1"
    assert loaded_config.name == "Test Workflow"
    assert len(loaded_config.agents) == 1
    assert loaded_config.agents[0].name == "Agent 1"

def test_list_configurations(config_manager):
    """Test listing configurations"""
    # Create multiple agent configs
    agents = [
        AgentConfig(
            id=f"agent-{i}",
            name=f"Agent {i}",
            description=f"Test agent {i}",
            type=ConfigurationType.GENERIC,
            model=ModelConfig(
                name=f"test-model-{i}",
                provider="default"
            ),
            system_prompt=f"You are agent {i}"
        ) for i in range(3)
    ]
    
    # Save configurations
    for agent in agents:
        config_manager.save_agent_config(agent)
    
    # List configurations
    listed_configs = config_manager.list_agent_configs()
    
    assert len(listed_configs) == 3
    assert all(isinstance(config, AgentConfig) for config in listed_configs)

def test_delete_configurations(config_manager):
    """Test deleting configurations"""
    # Create test agent config
    agent_config = AgentConfig(
        id="delete-test-agent",
        name="Delete Test Agent",
        description="An agent to be deleted",
        type=ConfigurationType.GENERIC,
        model=ModelConfig(
            name="delete-test-model",
            provider="default"
        ),
        system_prompt="You are a test agent to be deleted"
    )
    
    # Save configuration
    config_manager.save_agent_config(agent_config)
    
    # Verify configuration exists
    assert config_manager.load_agent_config("delete-test-agent") is not None
    
    # Delete configuration
    result = config_manager.delete_agent_config("delete-test-agent")
    
    assert result is True
    assert config_manager.load_agent_config("delete-test-agent") is None

def test_export_and_import_config(config_manager):
    """Test exporting and importing configurations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test agent config
        agent_config = AgentConfig(
            id="export-import-agent",
            name="Export Import Agent",
            description="Agent for export/import testing",
            type=ConfigurationType.GENERIC,
            model=ModelConfig(
                name="export-import-model",
                provider="default"
            ),
            system_prompt="You are an export/import test agent"
        )
        
        # Save original configuration
        config_manager.save_agent_config(agent_config)
        
        # Export configuration
        export_path = os.path.join(tmpdir, "exported_config.json")
        config_manager.export_config("export-import-agent", export_path)
        
        # Verify export file exists and is valid
        assert os.path.exists(export_path)
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data['id'] == "export-import-agent"
        
        # Import configuration to a new config manager
        new_config_manager = ConfigManager(tmpdir)
        new_config_manager.import_config(export_path)
        
        # Verify imported configuration
        imported_config = new_config_manager.load_agent_config("export-import-agent")
        assert imported_config is not None
        assert imported_config.id == "export-import-agent"
        assert imported_config.name == "Export Import Agent"
