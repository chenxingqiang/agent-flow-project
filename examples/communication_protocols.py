from typing import Dict, Any
from agentflow.core.workflow import WorkflowEngine

def federated_learning_example():
    """
    联邦学习通信协议示例
    模拟多个本地Agent训练模型并聚合
    """
    workflow_config = {
        "COLLABORATION": {
            "MODE": "PARALLEL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "FEDERATED"
            },
            "WORKFLOW": [
                {
                    "name": "local_model_1",
                    "model_params": {"weight1": 0.1, "bias1": 0.2}
                },
                {
                    "name": "local_model_2", 
                    "model_params": {"weight1": 0.3, "bias1": 0.4}
                },
                {
                    "name": "local_model_3",
                    "model_params": {"weight1": 0.5, "bias1": 0.6}
                }
            ]
        }
    }
    
    workflow = WorkflowEngine(workflow_config)
    result = workflow.execute({"task": "federated_learning"})
    print("Federated Learning Result:", result)

def gossip_protocol_example():
    """
    Gossip 通信协议示例
    模拟分布式系统中的信息随机交换
    """
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
                },
                {
                    "name": "node_3", 
                    "knowledge": {"topic_b": "info_3", "topic_c": "data_3"}
                }
            ]
        }
    }
    
    workflow = WorkflowEngine(workflow_config)
    result = workflow.execute({"task": "gossip_information_exchange"})
    print("Gossip Protocol Result:", result)

def hierarchical_merge_example():
    """
    分层合并通信协议示例
    模拟多层级Agent的协作和信息聚合
    """
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
                },
                "high_level_agent": {
                    "hierarchy_level": 2,
                    "dependencies": ["mid_level_agent_processed"]
                }
            }
        }
    }
    
    workflow = WorkflowEngine(workflow_config)
    result = workflow.execute({"task": "hierarchical_collaboration"})
    print("Hierarchical Merge Result:", result)

def main():
    print("通信协议使用示例:")
    federated_learning_example()
    gossip_protocol_example()
    hierarchical_merge_example()

if __name__ == "__main__":
    main()
