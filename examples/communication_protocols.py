from typing import Dict, Any
import asyncio
from agentflow.core.workflow import WorkflowEngine
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig

async def federated_learning_example():
    """
    联邦学习通信协议示例
    模拟多个本地Agent训练模型并聚合
    """
    # Create workflow config
    workflow = WorkflowConfig(
        id="federated_learning",
        name="Federated Learning Example",
        steps=[
            WorkflowStep(
                id="local_model_1",
                name="Local Model 1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"weight1": 0.1, "bias1": 0.2}
                )
            ),
            WorkflowStep(
                id="local_model_2",
                name="Local Model 2",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"weight1": 0.3, "bias1": 0.4}
                )
            ),
            WorkflowStep(
                id="local_model_3",
                name="Local Model 3",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"weight1": 0.5, "bias1": 0.6}
                )
            )
        ]
    )
    
    # Create workflow engine
    engine = WorkflowEngine()
    await engine.initialize()
    
    # Create workflow instance
    instance = await engine.create_workflow("federated_learning", workflow)
    
    # Execute workflow
    result = await engine.execute_workflow_instance(instance)
    print("Federated Learning Result:", result)

async def gossip_protocol_example():
    """
    Gossip 通信协议示例
    模拟分布式系统中的信息随机交换
    """
    # Create workflow config
    workflow = WorkflowConfig(
        id="gossip_protocol",
        name="Gossip Protocol Example",
        steps=[
            WorkflowStep(
                id="node_1",
                name="Node 1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"topic_a": "info_1", "topic_b": "data_1"}
                )
            ),
            WorkflowStep(
                id="node_2",
                name="Node 2",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"topic_a": "info_2", "topic_c": "data_2"}
                )
            ),
            WorkflowStep(
                id="node_3",
                name="Node 3",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"topic_b": "info_3", "topic_c": "data_3"}
                )
            )
        ]
    )
    
    # Create workflow engine
    engine = WorkflowEngine()
    await engine.initialize()
    
    # Create workflow instance
    instance = await engine.create_workflow("gossip_protocol", workflow)
    
    # Execute workflow
    result = await engine.execute_workflow_instance(instance)
    print("Gossip Protocol Result:", result)

async def hierarchical_merge_example():
    """
    分层合并通信协议示例
    模拟多层级Agent的协作和信息聚合
    """
    # Create workflow config
    workflow = WorkflowConfig(
        id="hierarchical_merge",
        name="Hierarchical Merge Example",
        steps=[
            WorkflowStep(
                id="low_level_agent_1",
                name="Low Level Agent 1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"data": "raw_data_1"}
                )
            ),
            WorkflowStep(
                id="low_level_agent_2",
                name="Low Level Agent 2",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="custom",
                    params={"data": "raw_data_2"}
                )
            ),
            WorkflowStep(
                id="mid_level_agent",
                name="Mid Level Agent",
                type=WorkflowStepType.TRANSFORM,
                dependencies=["low_level_agent_1", "low_level_agent_2"],
                config=StepConfig(
                    strategy="custom",
                    params={}
                )
            ),
            WorkflowStep(
                id="high_level_agent",
                name="High Level Agent",
                type=WorkflowStepType.TRANSFORM,
                dependencies=["mid_level_agent"],
                config=StepConfig(
                    strategy="custom",
                    params={}
                )
            )
        ]
    )
    
    # Create workflow engine
    engine = WorkflowEngine()
    await engine.initialize()
    
    # Create workflow instance
    instance = await engine.create_workflow("hierarchical_merge", workflow)
    
    # Execute workflow
    result = await engine.execute_workflow_instance(instance)
    print("Hierarchical Merge Result:", result)

async def main():
    print("通信协议使用示例:")
    await federated_learning_example()
    await gossip_protocol_example()
    await hierarchical_merge_example()

if __name__ == "__main__":
    asyncio.run(main())
