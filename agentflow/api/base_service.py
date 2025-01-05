"""Base service class for exposing agents as API endpoints."""
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ray
import logging
from ..agents.agent import Agent
from ..core.config import AgentConfig

class AgentRequest(BaseModel):
    """Agent API请求模型"""
    input_data: Dict[str, Any]
    config_override: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    """Agent API响应模型"""
    result: Dict[str, Any]
    metadata: Dict[str, Any]

class BaseAgentService:
    """Agent服务基类"""
    
    def __init__(
        self,
        agent_config: Dict[str, Any],
        service_config: Optional[Dict[str, Any]] = None
    ):
        """初始化服务
        
        Args:
            agent_config: Agent配置
            service_config: 服务配置
        """
        self.agent_config = AgentConfig(**agent_config)
        self.service_config = service_config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init()
            
        # 创建FastAPI应用
        self.app = FastAPI(
            title=f"{self.agent_config.agent.name} Service",
            description=f"API service for {self.agent_config.agent.type} agent",
            version=self.agent_config.agent.version
        )
        
        # 注册路由
        self._register_routes()
        
    def _register_routes(self):
        """注册API路由"""
        @self.app.post("/process")
        async def process(request: AgentRequest) -> AgentResponse:
            try:
                # 合并配置
                if request.config_override:
                    config = self.agent_config.copy()
                    config.update(request.config_override)
                else:
                    config = self.agent_config
                    
                # 创建Agent实例
                agent = Agent(config=config.dict())
                
                # 异步处理请求
                result = await self._process_async(agent, request.input_data)
                
                return AgentResponse(
                    result=result,
                    metadata={
                        "agent_version": config.agent.version,
                        "agent_type": config.agent.type
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Error processing request: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
                
    async def _process_async(self, agent: Agent, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """异步处理请求
        
        Args:
            agent: Agent实例
            input_data: 输入数据
            
        Returns:
            处理结果
        """
        # 使用Ray进行异步处理
        @ray.remote
        def process_task(agent_config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
            agent = Agent(config=agent_config)
            return agent.process(data)
            
        # 提交任务
        task_ref = process_task.remote(agent.config, input_data)
        
        try:
            # 等待结果
            result = await ray.get(task_ref)
            return result
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            raise
            
    def start(self, host: str = "0.0.0.0", port: int = 8000):
        """启动服务
        
        Args:
            host: 服务主机
            port: 服务端口
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
        
    def stop(self):
        """停止服务"""
        # 清理Ray资源
        if ray.is_initialized():
            ray.shutdown()
            
class AgentServiceConfig(BaseModel):
    """Agent服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    max_retries: int = 3
