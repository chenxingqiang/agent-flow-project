"""Visualization components for AgentFlow using ell.studio."""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class NodeType(Enum):
    """节点类型"""
    AGENT = "agent"
    WORKFLOW = "workflow"
    INPUT = "input"
    OUTPUT = "output"
    PROCESSOR = "processor"
    CONNECTOR = "connector"

class EdgeType(Enum):
    """边类型"""
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"
    MESSAGE = "message"

@dataclass
class VisualNode:
    """可视化节点"""
    id: str
    type: NodeType
    label: str
    data: Dict[str, Any]
    position: Dict[str, float]
    style: Optional[Dict[str, Any]] = None

@dataclass
class VisualEdge:
    """可视化边"""
    id: str
    source: str
    target: str
    type: EdgeType
    data: Optional[Dict[str, Any]] = None
    style: Optional[Dict[str, Any]] = None

class VisualGraph:
    """可视化图"""
    def __init__(self):
        self.nodes: List[VisualNode] = []
        self.edges: List[VisualEdge] = []
        
    def add_node(self, node: VisualNode):
        """添加节点"""
        self.nodes.append(node)
        
    def add_edge(self, edge: VisualEdge):
        """添加边"""
        self.edges.append(edge)
        
    def to_ell_format(self) -> Dict[str, Any]:
        """转换为ell.studio格式"""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type.value,
                    "data": {
                        "label": node.label,
                        **node.data
                    },
                    "position": node.position,
                    "style": node.style or {}
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type.value,
                    "data": edge.data or {},
                    "style": edge.style or {}
                }
                for edge in self.edges
            ]
        }

class DefaultStyles:
    """默认样式"""
    NODE_STYLES = {
        NodeType.AGENT: {
            "backgroundColor": "#6366f1",
            "borderRadius": 8,
            "padding": 16,
            "color": "#ffffff"
        },
        NodeType.WORKFLOW: {
            "backgroundColor": "#10b981",
            "borderRadius": 8,
            "padding": 16,
            "color": "#ffffff"
        },
        NodeType.INPUT: {
            "backgroundColor": "#f59e0b",
            "borderRadius": 8,
            "padding": 12,
            "color": "#ffffff"
        },
        NodeType.OUTPUT: {
            "backgroundColor": "#ef4444",
            "borderRadius": 8,
            "padding": 12,
            "color": "#ffffff"
        },
        NodeType.PROCESSOR: {
            "backgroundColor": "#8b5cf6",
            "borderRadius": 8,
            "padding": 12,
            "color": "#ffffff"
        },
        NodeType.CONNECTOR: {
            "backgroundColor": "#64748b",
            "borderRadius": 8,
            "padding": 8,
            "color": "#ffffff"
        }
    }
    
    EDGE_STYLES = {
        EdgeType.DATA_FLOW: {
            "stroke": "#94a3b8",
            "strokeWidth": 2,
            "animated": True
        },
        EdgeType.CONTROL_FLOW: {
            "stroke": "#475569",
            "strokeWidth": 2,
            "strokeDasharray": "5,5"
        },
        EdgeType.MESSAGE: {
            "stroke": "#0ea5e9",
            "strokeWidth": 2,
            "animated": True
        }
    }

class VisualLayout:
    """布局管理器"""
    @staticmethod
    def auto_layout(graph: VisualGraph, config: Optional[Dict[str, Any]] = None) -> VisualGraph:
        """自动布局
        
        Args:
            graph: 可视化图
            config: 布局配置
            
        Returns:
            布局后的图
        """
        # 这里可以实现自动布局算法
        # 例如分层布局、力导向布局等
        return graph

class InteractionHandler:
    """交互处理器"""
    def __init__(self):
        self.handlers: Dict[str, Any] = {}
        
    def register_handler(self, event_type: str, handler: Any):
        """注册事件处理器"""
        self.handlers[event_type] = handler
        
    def handle_event(self, event_type: str, event_data: Dict[str, Any]):
        """处理事件"""
        if handler := self.handlers.get(event_type):
            return handler(event_data)
        return None
