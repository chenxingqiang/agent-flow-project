"""Visualization renderer for AgentFlow using ell.studio."""
from typing import Dict, Any, Optional, List
import json
from .components import (
    VisualGraph, VisualNode, VisualEdge,
    NodeType, EdgeType, DefaultStyles, VisualLayout
)

class AgentVisualizer:
    """Agent可视化器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.graph = VisualGraph()
        self.layout = VisualLayout()
        
    def visualize_agent(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """可视化Agent配置
        
        Args:
            agent_config: Agent配置
            
        Returns:
            ell.studio格式的可视化数据
        """
        # 创建Agent节点
        agent_node = VisualNode(
            id="agent_main",
            type=NodeType.AGENT,
            label=agent_config["agent"]["name"],
            data={
                "type": agent_config["agent"]["type"],
                "version": agent_config["agent"]["version"]
            },
            position={"x": 0, "y": 0},
            style=DefaultStyles.NODE_STYLES[NodeType.AGENT]
        )
        self.graph.add_node(agent_node)
        
        # 添加输入处理器
        input_node = self._add_input_processor(agent_config["input_specification"])
        self.graph.add_edge(VisualEdge(
            id=f"edge_input_{agent_node.id}",
            source=input_node.id,
            target=agent_node.id,
            type=EdgeType.DATA_FLOW,
            style=DefaultStyles.EDGE_STYLES[EdgeType.DATA_FLOW]
        ))
        
        # 添加输出处理器
        output_node = self._add_output_processor(agent_config["output_specification"])
        self.graph.add_edge(VisualEdge(
            id=f"edge_{agent_node.id}_output",
            source=agent_node.id,
            target=output_node.id,
            type=EdgeType.DATA_FLOW,
            style=DefaultStyles.EDGE_STYLES[EdgeType.DATA_FLOW]
        ))
        
        # 自动布局
        self.layout.auto_layout(self.graph)
        
        return self.graph.to_ell_format()
        
    def visualize_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """可视化工作流配置
        
        Args:
            workflow_config: 工作流配置
            
        Returns:
            ell.studio格式的可视化数据
        """
        # 创建工作流节点
        workflow_node = VisualNode(
            id="workflow_main",
            type=NodeType.WORKFLOW,
            label=workflow_config.get("name", "Workflow"),
            data=workflow_config,
            position={"x": 0, "y": 0},
            style=DefaultStyles.NODE_STYLES[NodeType.WORKFLOW]
        )
        self.graph.add_node(workflow_node)
        
        # 添加工作流步骤
        for i, step in enumerate(workflow_config.get("steps", [])):
            step_node = self._add_workflow_step(step, i)
            self.graph.add_edge(VisualEdge(
                id=f"edge_step_{i}",
                source=workflow_node.id,
                target=step_node.id,
                type=EdgeType.CONTROL_FLOW,
                style=DefaultStyles.EDGE_STYLES[EdgeType.CONTROL_FLOW]
            ))
            
        # 自动布局
        self.layout.auto_layout(self.graph)
        
        return self.graph.to_ell_format()
        
    def _add_input_processor(self, input_spec: Dict[str, Any]) -> VisualNode:
        """添加输入处理器节点"""
        node = VisualNode(
            id="input_processor",
            type=NodeType.INPUT,
            label="Input Processor",
            data={
                "modes": input_spec.get("MODES", []),
                "validation": input_spec.get("VALIDATION", {})
            },
            position={"x": -200, "y": 0},
            style=DefaultStyles.NODE_STYLES[NodeType.INPUT]
        )
        self.graph.add_node(node)
        return node
        
    def _add_output_processor(self, output_spec: Dict[str, Any]) -> VisualNode:
        """添加输出处理器节点"""
        node = VisualNode(
            id="output_processor",
            type=NodeType.OUTPUT,
            label="Output Processor",
            data={
                "modes": output_spec.get("MODES", []),
                "strategies": output_spec.get("STRATEGIES", {})
            },
            position={"x": 200, "y": 0},
            style=DefaultStyles.NODE_STYLES[NodeType.OUTPUT]
        )
        self.graph.add_node(node)
        return node
        
    def _add_workflow_step(self, step: Dict[str, Any], index: int) -> VisualNode:
        """添加工作流步骤节点"""
        node = VisualNode(
            id=f"step_{index}",
            type=NodeType.PROCESSOR,
            label=step.get("name", f"Step {index + 1}"),
            data=step,
            position={"x": 0, "y": (index + 1) * 100},
            style=DefaultStyles.NODE_STYLES[NodeType.PROCESSOR]
        )
        self.graph.add_node(node)
        return node
        
class LiveVisualizer(AgentVisualizer):
    """实时可视化器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.message_history: List[Dict[str, Any]] = []
        
    def add_message(self, message: Dict[str, Any]):
        """添加消息"""
        self.message_history.append(message)
        self._update_visualization()
        
    def _update_visualization(self):
        """更新可视化"""
        # 这里可以实现实时更新逻辑
        # 例如通过WebSocket推送更新
        pass
        
    def get_message_history(self) -> List[Dict[str, Any]]:
        """获取消息历史"""
        return self.message_history
