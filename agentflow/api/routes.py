"""
API routes for the AgentFlow application
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .monitor_service import monitor_service

router = APIRouter()

@router.websocket("/monitor/{agent_id}")
async def monitor_agent(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for monitoring agent execution"""
    try:
        await monitor_service.connect(agent_id, websocket)
        while True:
            try:
                # Keep connection alive
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        await monitor_service.disconnect(agent_id, websocket)
