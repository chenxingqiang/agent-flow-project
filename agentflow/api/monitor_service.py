"""
Monitor service for tracking agent execution and performance metrics
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from fastapi import WebSocket
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class AgentMetrics(BaseModel):
    """Agent performance metrics"""
    tokens: int = 0
    latency: float = 0
    memory: int = 0
    
class AgentStatus(BaseModel):
    """Agent execution status"""
    id: str
    name: str
    type: str
    status: str
    progress: Optional[float]
    metrics: AgentMetrics
    
class LogEntry(BaseModel):
    """Execution log entry"""
    agent_name: str
    level: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
class MonitorService:
    """Service for monitoring agent execution"""
    
    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}
        self._agent_status: Dict[str, AgentStatus] = {}
        
    async def connect(self, agent_id: str, websocket: WebSocket):
        """Connect a new WebSocket client"""
        await websocket.accept()
        
        if agent_id not in self._connections:
            self._connections[agent_id] = []
        self._connections[agent_id].append(websocket)
        
        # Send current status if available
        if agent_id in self._agent_status:
            await websocket.send_json({
                'type': 'agent_status',
                'agent': self._agent_status[agent_id].dict()
            })
            
    async def disconnect(self, agent_id: str, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        if agent_id in self._connections:
            self._connections[agent_id].remove(websocket)
            
    async def update_status(self, agent_id: str, status: AgentStatus):
        """Update agent status and notify clients"""
        self._agent_status[agent_id] = status
        await self._broadcast(agent_id, {
            'type': 'agent_status',
            'agent': status.dict()
        })
        
    async def update_metrics(self, agent_id: str, metrics: AgentMetrics):
        """Update agent metrics and notify clients"""
        if agent_id in self._agent_status:
            self._agent_status[agent_id].metrics = metrics
            await self._broadcast(agent_id, {
                'type': 'metrics',
                'metrics': metrics.dict()
            })
            
    async def add_log(self, agent_id: str, log: LogEntry):
        """Add a log entry and notify clients"""
        await self._broadcast(agent_id, {
            'type': 'log',
            'log': log.dict()
        })
        
    async def _broadcast(self, agent_id: str, message: dict):
        """Broadcast a message to all connected clients for an agent"""
        if agent_id in self._connections:
            dead_connections = []
            for websocket in self._connections[agent_id]:
                try:
                    await websocket.send_json(message)
                except:
                    dead_connections.append(websocket)
                    logger.warning(f"Failed to send message to websocket")
                    
            # Clean up dead connections
            for websocket in dead_connections:
                await self.disconnect(agent_id, websocket)
                
monitor_service = MonitorService()
