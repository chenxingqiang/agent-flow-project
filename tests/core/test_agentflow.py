"""
Test suite for AgentFlow core management class
"""

import pytest
import asyncio
from typing import Dict, Any

from agentflow import AgentFlow
from agentflow.core.config import WorkflowConfig, AgentConfig, ModelConfig

@pytest.fixture
def agentflow_instance():
    """Create an AgentFlow instance for testing"""
    return AgentFlow()

@pytest.mark.asyncio
async def test_create_workflow(agentflow_instance):
    """Test creating a workflow"""
    workflow_config = WorkflowConfig(
        id="test-workflow",
        name="Test Workflow",
        agents=[
            AgentConfig(
                id="test-agent",
                name="Test Agent",
                type="test",
                model=ModelConfig(name="test-model", provider="default"),
                config={"test_key": "test_value"}
            )
        ]
    )
    
    # Verify workflow configuration
    assert workflow_config.id == "test-workflow"
    assert workflow_config.name == "Test Workflow"
    assert len(workflow_config.agents) == 1
    assert workflow_config.agents[0].id == "test-agent"
    assert workflow_config.agents[0].model.provider == "default"
    assert workflow_config.agents[0].config == {"test_key": "test_value"}

    workflow = agentflow_instance.create_workflow("test-workflow", workflow_config)
    
    assert workflow is not None
    assert "test-workflow" in agentflow_instance.workflows

@pytest.mark.asyncio
async def test_execute_workflow(agentflow_instance):
    """Test executing a workflow"""
    workflow_config = WorkflowConfig(
        id="test-workflow",
        name="Test Workflow",
        agents=[
            AgentConfig(
                id="test-agent",
                name="Test Agent",
                type="test",
                model=ModelConfig(name="test-model", provider="default"),
                config={"test_key": "test_value"}
            )
        ]
    )
    
    # Verify workflow configuration
    assert workflow_config.id == "test-workflow"
    assert workflow_config.name == "Test Workflow"
    assert len(workflow_config.agents) == 1
    assert workflow_config.agents[0].id == "test-agent"
    assert workflow_config.agents[0].model.provider == "default"
    assert workflow_config.agents[0].config == {"test_key": "test_value"}

    workflow = agentflow_instance.create_workflow("test-workflow", workflow_config)
    
    results = await agentflow_instance.execute_workflow("test-workflow")
    
    assert results is not None
    assert "test-workflow" in agentflow_instance.active_workflows

def test_list_workflows(agentflow_instance):
    """Test listing workflows"""
    workflow_config1 = WorkflowConfig(
        id="workflow-1",
        name="Workflow 1",
        agents=[]
    )
    
    workflow_config2 = WorkflowConfig(
        id="workflow-2",
        name="Workflow 2",
        agents=[]
    )
    
    agentflow_instance.create_workflow("workflow-1", workflow_config1)
    agentflow_instance.create_workflow("workflow-2", workflow_config2)
    
    workflows = agentflow_instance.list_workflows()
    
    assert len(workflows) == 2
    assert "workflow-1" in workflows
    assert "workflow-2" in workflows

def test_get_workflow_status(agentflow_instance):
    """Test getting workflow status"""
    workflow_config = WorkflowConfig(
        id="status-workflow",
        name="Status Workflow",
        agents=[]
    )
    
    workflow = agentflow_instance.create_workflow("status-workflow", workflow_config)
    
    with pytest.raises(ValueError):
        agentflow_instance.get_workflow_status("status-workflow")

@pytest.mark.asyncio
async def test_stop_workflow(agentflow_instance):
    """Test stopping a workflow"""
    workflow_config = WorkflowConfig(
        id="stop-workflow",
        name="Stop Workflow",
        agents=[
            AgentConfig(
                id="test-agent",
                name="Test Agent",
                type="test",
                model=ModelConfig(name="test-model", provider="default"),
                config={"test_key": "test_value"}
            )
        ]
    )
    
    # Verify workflow configuration
    assert workflow_config.id == "stop-workflow"
    assert workflow_config.name == "Stop Workflow"
    assert len(workflow_config.agents) == 1
    assert workflow_config.agents[0].id == "test-agent"
    assert workflow_config.agents[0].model.provider == "default"
    assert workflow_config.agents[0].config == {"test_key": "test_value"}

    workflow = agentflow_instance.create_workflow("stop-workflow", workflow_config)
    await agentflow_instance.execute_workflow("stop-workflow")
    
    await agentflow_instance.stop_workflow("stop-workflow")
    
    assert "stop-workflow" not in agentflow_instance.active_workflows

@pytest.mark.asyncio
async def test_workflow_not_found(agentflow_instance):
    """Test error handling for non-existent workflow"""
    with pytest.raises(ValueError):
        await agentflow_instance.execute_workflow("non-existent-workflow")
    
    with pytest.raises(ValueError):
        agentflow_instance.get_workflow_status("non-existent-workflow")
    
    with pytest.raises(ValueError):
        await agentflow_instance.stop_workflow("non-existent-workflow")
