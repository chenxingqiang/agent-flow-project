import pytest
import asyncio
from typing import Dict, Any
from agentflow.core.workflow import WorkflowEngine
from agentflow.core.workflow_config import WorkflowConfig
from agentflow.core.metric_type import MetricType

@pytest.mark.asyncio
async def test_federated_learning_protocol():
    workflow_def = {
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
    
    workflow_config = WorkflowConfig(
        max_iterations=5,
        timeout=3600,
        logging_level="INFO",
        required_fields=[],
        error_handling={},
        retry_policy=None,
        error_policy=None,
        is_distributed=True,
        distributed=True,
        steps=[],
        metadata={},
        agents={}
    )

    workflow = WorkflowEngine(workflow_def, workflow_config)
    result = await workflow.execute({
        "research_topic": "AI Ethics",
        "metrics": {
            MetricType.LATENCY.value: [{"value": 100, "timestamp": 1234567890}]
        }
    })

    assert result is not None
    assert "metrics" in result

@pytest.mark.asyncio
async def test_gossip_protocol():
    workflow_def = {
        "COLLABORATION": {
            "MODE": "PARALLEL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "GOSSIP"
            },
            "WORKFLOW": [
                {
                    "name": "node_1",
                    "agent_type": "test",
                    "knowledge": {"topic_a": "info_1", "topic_b": "data_1"}
                },
                {
                    "name": "node_2",
                    "agent_type": "test",
                    "knowledge": {"topic_a": "info_2", "topic_c": "data_2"}
                }
            ]
        }
    }
    
    workflow_config = WorkflowConfig(
        max_iterations=5,
        timeout=3600,
        logging_level="INFO",
        required_fields=[],
        error_handling={},
        retry_policy=None,
        error_policy=None,
        is_distributed=True,
        distributed=True,
        steps=[],
        metadata={},
        agents={}
    )

    workflow = WorkflowEngine(workflow_def, workflow_config)
    result = await workflow.execute({
        "research_topic": "AI Ethics",
        "metrics": {
            MetricType.LATENCY.value: [{"value": 100, "timestamp": 1234567890}]
        }
    })

    assert result is not None
    assert "metrics" in result

@pytest.mark.asyncio
async def test_hierarchical_merge_protocol():
    workflow_def = {
        "COLLABORATION": {
            "MODE": "DYNAMIC_ROUTING",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "HIERARCHICAL"
            },
            "WORKFLOW": {
                "low_level_agent_1": {
                    "name": "low_level_agent_1",
                    "agent_type": "test",
                    "hierarchy_level": 0,
                    "data": "raw_data_1"
                },
                "low_level_agent_2": {
                    "name": "low_level_agent_2",
                    "agent_type": "test",
                    "hierarchy_level": 0,
                    "data": "raw_data_2"
                },
                "mid_level_agent": {
                    "name": "mid_level_agent",
                    "agent_type": "test",
                    "hierarchy_level": 1,
                    "dependencies": ["low_level_agent_1", "low_level_agent_2"]
                }
            }
        }
    }
    
    workflow_config = WorkflowConfig(
        max_iterations=5,
        timeout=3600,
        logging_level="INFO",
        required_fields=[],
        error_handling={},
        retry_policy=None,
        error_policy=None,
        is_distributed=True,
        distributed=True,
        steps=[],
        metadata={},
        agents={}
    )

    workflow = WorkflowEngine(workflow_def, workflow_config)
    result = await workflow.execute({
        "research_topic": "AI Ethics",
        "metrics": {
            MetricType.LATENCY.value: [{"value": 100, "timestamp": 1234567890}]
        }
    })

    assert result is not None
    assert "metrics" in result

@pytest.mark.asyncio
async def test_invalid_communication_protocol():
    workflow_def = {
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
    
    workflow_config = WorkflowConfig(
        max_iterations=5,
        timeout=3600,
        logging_level="INFO",
        required_fields=[],
        error_handling={},
        retry_policy=None,
        error_policy=None,
        is_distributed=False,
        distributed=False,
        steps=[],
        metadata={},
        agents={}
    )

    workflow = WorkflowEngine(workflow_def, workflow_config)
    with pytest.raises(ValueError):
        await workflow.execute({
            "research_topic": "AI Ethics",
            "metrics": {
                MetricType.LATENCY.value: [{"value": 100, "timestamp": 1234567890}]
            }
        })

@pytest.mark.asyncio
async def test_empty_workflow():
    workflow_def = {
        "COLLABORATION": {
            "MODE": "PARALLEL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "FEDERATED"
            },
            "WORKFLOW": []
        }
    }
    
    workflow_config = WorkflowConfig(
        max_iterations=5,
        timeout=3600,
        logging_level="INFO",
        required_fields=[],
        error_handling={},
        retry_policy=None,
        error_policy=None,
        is_distributed=True,
        distributed=True,
        steps=[],
        metadata={},
        agents={}
    )

    workflow = WorkflowEngine(workflow_def, workflow_config)
    result = await workflow.execute({
        "research_topic": "AI Ethics",
        "metrics": {
            MetricType.LATENCY.value: [{"value": 100, "timestamp": 1234567890}]
        }
    })

    assert result is not None
    assert "metrics" in result

@pytest.mark.asyncio
async def test_hierarchical_merge_protocol_with_test_agent():
    workflow_def = {
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
    
    workflow_config = WorkflowConfig(
        max_iterations=5,
        timeout=3600,
        logging_level="INFO",
        required_fields=[],
        error_handling={},
        retry_policy=None,
        error_policy=None,
        is_distributed=False,
        distributed=False,
        steps=[],
        metadata={},
        agents={}
    )

    workflow = WorkflowEngine(workflow_def, workflow_config)
    result = await workflow.execute({
        "research_topic": "AI Ethics",
        "metrics": {
            MetricType.LATENCY.value: [{"value": 100, "timestamp": 1234567890}]
        }
    })

    assert result is not None
    assert "metrics" in result

@pytest.mark.asyncio
async def test_invalid_communication_protocol_with_test_agent():
    workflow_def = {
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
    
    workflow_config = WorkflowConfig(
        max_iterations=5,
        timeout=3600,
        logging_level="INFO",
        required_fields=[],
        error_handling={},
        retry_policy=None,
        error_policy=None,
        is_distributed=False,
        distributed=False,
        steps=[],
        metadata={},
        agents={}
    )

    workflow = WorkflowEngine(workflow_def, workflow_config)
    with pytest.raises(ValueError):
        await workflow.execute({
            "research_topic": "AI Ethics",
            "metrics": {
                MetricType.LATENCY.value: [{"value": 100, "timestamp": 1234567890}]
            }
        })
