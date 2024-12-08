"""Visualization service for AgentFlow using ell.studio."""
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket
import json
import asyncio
from .renderer import AgentVisualizer, LiveVisualizer
from ..core.config import AgentConfig

class VisualizationService:
    """可视化服务"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.app = FastAPI(
            title="AgentFlow Visualization Service",
            description="Visualization service for AgentFlow",
            version="1.0.0"
        )
        self.visualizer = AgentVisualizer(self.config)
        self.live_visualizer = LiveVisualizer(self.config)
        
        # 注册路由
        self._register_routes()
        
    def _register_routes(self):
        """注册API路由"""
        @self.app.post("/visualize/agent")
        async def visualize_agent(agent_config: Dict[str, Any]):
            """可视化Agent配置"""
            return self.visualizer.visualize_agent(agent_config)
            
        @self.app.post("/visualize/workflow")
        async def visualize_workflow(workflow_config: Dict[str, Any]):
            """可视化工作流配置"""
            return self.visualizer.visualize_workflow(workflow_config)
            
        @self.app.websocket("/live")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket实时更新端点"""
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_json()
                    self.live_visualizer.add_message(data)
                    await websocket.send_json(self.live_visualizer.get_message_history())
            except Exception as e:
                print(f"WebSocket error: {str(e)}")
                await websocket.close()
                
    def start(self, host: str = "0.0.0.0", port: int = 8001):
        """启动服务"""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

class EllStudioIntegration:
    """ell.studio集成"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.project_id = config.get("project_id")
        
    async def push_visualization(self, visual_data: Dict[str, Any]):
        """推送可视化数据到ell.studio
        
        Args:
            visual_data: 可视化数据
        """
        # 这里实现与ell.studio API的集成
        pass
        
    async def update_visualization(self, update_data: Dict[str, Any]):
        """更新ell.studio可视化
        
        Args:
            update_data: 更新数据
        """
        # 这里实现可视化更新逻辑
        pass
        
    async def get_visualization(self) -> Dict[str, Any]:
        """获取当前可视化状态
        
        Returns:
            可视化状态数据
        """
        # 这里实现获取可视化状态的逻辑
        pass
