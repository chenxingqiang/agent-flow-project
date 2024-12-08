from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import ray

from agentflow import AgentFlow
from agentflow.core.config import AgentConfig

router = APIRouter()
logger = logging.getLogger(__name__)

class WorkflowRequest(BaseModel):
    """Workflow request model"""
    workflow: Dict[str, Any]
    input_data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None

@router.post("/execute")
async def execute_workflow(request: WorkflowRequest):
    """Execute workflow synchronously"""
    try:
        # Validate workflow configuration
        if not request.workflow or 'WORKFLOW' not in request.workflow:
            raise ValueError("Invalid workflow configuration")

        # Use default config if not provided
        config_dict = request.config or {}
        
        # Create agent config
        config = AgentConfig(
            agent_type=config_dict.get('agent_type', 'research'),
            model={
                'provider': config_dict.get('provider', 'openai'),
                'name': config_dict.get('model', 'gpt-4'),
                'temperature': config_dict.get('temperature', 0.5)
            },
            workflow={
                'max_iterations': config_dict.get('max_iterations', 10),
                'logging_level': config_dict.get('logging_level', 'INFO'),
                'distributed': False
            }
        )
        
        # Initialize agent
        agent = AgentFlow(config)
        agent.workflow_def = request.workflow
        
        # Execute workflow
        result = agent.execute_workflow(request.input_data)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/execute_async")
async def execute_workflow_async(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Execute workflow asynchronously"""
    try:
        # Validate workflow configuration
        if not request.workflow or 'WORKFLOW' not in request.workflow:
            raise ValueError("Invalid workflow configuration")

        # Use default config if not provided
        config_dict = request.config or {}
        
        # Create agent config
        config = AgentConfig(
            agent_type=config_dict.get('agent_type', 'research'),
            model={
                'provider': config_dict.get('provider', 'openai'),
                'name': config_dict.get('model', 'gpt-4'),
                'temperature': config_dict.get('temperature', 0.5)
            },
            workflow={
                'max_iterations': config_dict.get('max_iterations', 10),
                'logging_level': config_dict.get('logging_level', 'INFO'),
                'distributed': True
            }
        )
        
        # Initialize agent
        agent = AgentFlow(config)
        agent.workflow_def = request.workflow
        
        # Execute workflow asynchronously
        result_ref = agent.execute_workflow_async(request.input_data)
        
        # Return task ID for status checking
        return {"task_id": str(result_ref)}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Async workflow execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/status/{task_id}")
async def get_workflow_status(task_id: str):
    """Get workflow execution status"""
    try:
        # Get result if ready
        result = ray.get(ray.ObjectRef.from_hex(task_id))
        return {"status": "completed", "result": result}
    except ray.exceptions.GetTimeoutError:
        return {"status": "running"}
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Status check failed") 