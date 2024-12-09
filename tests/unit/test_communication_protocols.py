import pytest
import asyncio
from typing import Dict, Any
from agentflow.core.workflow import WorkflowEngine

@pytest.mark.asyncio
async def test_federated_learning_protocol():
    workflow_config = {
        "COLLABORATION": {
            "MODE": "PARALLEL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "FEDERATED"
            },
            "WORKFLOW": [
                {
                    "name": "local_model_1",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.1, "bias1": 0.2}
                },
                {
                    "name": "local_model_2",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.3, "bias1": 0.4}
                }
            ]
        }
    }
    
    workflow = WorkflowEngine(workflow_config)
    result = await workflow.execute({"task": "test_federated_learning"})
    
    assert "global_model" in result
    assert "weight1" in result["global_model"]
    assert "bias1" in result["global_model"]
    assert abs(result["global_model"]["weight1"] - 0.2) < 1e-6

@pytest.mark.asyncio
async def test_gossip_protocol():
    workflow_config = {
        "COLLABORATION": {
            "MODE": "PARALLEL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "GOSSIP"
            },
            "WORKFLOW": [
                {
                    "name": "node_1",
                    "knowledge": {"topic_a": "info_1", "topic_b": "data_1"}
                },
                {
                    "name": "node_2",
                    "knowledge": {"topic_a": "info_2", "topic_c": "data_2"}
                }
            ]
        }
    }
    
    workflow = WorkflowEngine(workflow_config)
    result = await workflow.execute({"task": "test_gossip_protocol"})
    
    assert len(result) > 0
    assert any("topic_a" in key for key in result.keys())
    assert any("topic_b" in key for key in result.keys())
    assert any("topic_c" in key for key in result.keys())

@pytest.mark.asyncio
async def test_hierarchical_merge_protocol():
    workflow_config = {
        "COLLABORATION": {
            "MODE": "DYNAMIC_ROUTING",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "HIERARCHICAL"
            },
            "WORKFLOW": {
                "low_level_agent_1": {
                    "hierarchy_level": 0,
                    "data": "raw_data_1"
                },
                "low_level_agent_2": {
                    "hierarchy_level": 0, 
                    "data": "raw_data_2"
                },
                "mid_level_agent": {
                    "hierarchy_level": 1,
                    "dependencies": ["low_level_agent_1_processed", "low_level_agent_2_processed"]
                }
            }
        }
    }
    
    workflow = WorkflowEngine(workflow_config)
    result = await workflow.execute({"task": "test_hierarchical_merge"})
    
    assert "level_0" in result
    assert "level_1" in result
    assert len(result) == 2

@pytest.mark.asyncio
async def test_invalid_communication_protocol():
    workflow_config = {
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "GOSSIP"
            },
            "WORKFLOW": [
                {
                    "name": "low_level_agent_1",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.1, "bias1": 0.2}
                },
                {
                    "name": "low_level_agent_2",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.3, "bias1": 0.4}
                }
            ]
        }
    }
    
    workflow = WorkflowEngine(workflow_config)
    result = await workflow.execute({"task": "test_invalid_protocol"})
    
    assert len(result) > 0  # 应该使用默认合并策略

@pytest.mark.asyncio
async def test_empty_workflow():
    workflow_config = {
        "COLLABORATION": {
            "MODE": "PARALLEL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "FEDERATED"
            },
            "WORKFLOW": []
        }
    }
    
    workflow = WorkflowEngine(workflow_config)
    result = await workflow.execute({"task": "test_empty_workflow"})
    
    assert result == {}

@pytest.mark.asyncio
async def test_hierarchical_merge_protocol_with_test_agent():
    workflow_config = {
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "HIERARCHICAL_MERGE"
            },
            "WORKFLOW": [
                {
                    "name": "high_level_agent",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.5, "bias1": 0.3}
                },
                {
                    "name": "low_level_agent_1",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.1, "bias1": 0.2}
                },
                {
                    "name": "low_level_agent_2",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.3, "bias1": 0.4}
                }
            ]
        }
    }
    
    workflow = WorkflowEngine(workflow_config)
    result = await workflow.execute({"task": "test_hierarchical_merge_protocol_with_test_agent"})
    
    assert len(result) > 0

@pytest.mark.asyncio
async def test_invalid_communication_protocol_with_test_agent():
    workflow_config = {
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "UNKNOWN_PROTOCOL"
            },
            "WORKFLOW": [
                {
                    "name": "agent_1",
                    "agent_type": "test"
                },
                {
                    "name": "agent_2",
                    "agent_type": "test"
                }
            ]
        }
    }
    
    workflow = WorkflowEngine(workflow_config)
    result = await workflow.execute({"task": "test_invalid_protocol_with_test_agent"})
    
    assert len(result) > 0  # 应该使用默认合并策略
