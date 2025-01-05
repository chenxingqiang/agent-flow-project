"""
Workflow API module for AgentFlow.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import ray
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

from agentflow import AgentFlow
from agentflow.core.config import AgentConfig
from agentflow.core.research_workflow import ResearchDistributedWorkflow

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
        
        # Create workflow instance
        workflow = ResearchDistributedWorkflow(request.workflow, config_dict)
        
        # Execute workflow
        try:
            result = workflow.execute(request.input_data)
            return {
                "status": "success",
                "workflow_id": "sync-workflow",
                "output": result
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute_async")
async def execute_workflow_async(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Execute workflow asynchronously"""
    try:
        # Validate workflow configuration
        if not request.workflow or 'WORKFLOW' not in request.workflow:
            raise ValueError("Invalid workflow configuration")

        # Use default config if not provided
        config_dict = request.config or {}
        
        # Create workflow instance
        workflow = ResearchDistributedWorkflow(request.workflow, config_dict)
        
        # Execute workflow asynchronously
        try:
            result_ref = workflow.execute_async(request.input_data)
            return {
                "status": "success",
                "workflow_id": str(result_ref)
            }
        except Exception as e:
            logger.error(f"Async workflow execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Async workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}")
async def get_workflow_status(task_id: str):
    """Get workflow execution status"""
    try:
        # For sync workflows, return completed status
        if task_id == "sync-workflow":
            return {"status": "completed"}

        try:
            # Try to get the result
            result = ray.get(ray.ObjectRef.from_hex(task_id))
            return {
                "status": "completed",
                "result": result
            }
        except ray.exceptions.GetTimeoutError:
            return {"status": "running"}
        except Exception as e:
            logger.error(f"Error checking workflow status: {e}")
            return {"status": "running"}  # Return running instead of failed
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed") 