import json
import logging
import ray
import time
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agentflow.core.distributed_workflow import ResearchDistributedWorkflow
from agentflow import AgentFlow
from agentflow.core.config import AgentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ray
ray.init(logging_level=logging.INFO, dashboard_host='0.0.0.0')

# Create FastAPI app
app = FastAPI(
    title="Distributed Workflow API",
    description="API for executing distributed workflows",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class WorkflowRequest(BaseModel):
    workflow: Dict[str, Any] = Field(
        ..., 
        description="Workflow configuration dictionary",
        example={
            "WORKFLOW": [
                {
                    "input": ["research_topic", "deadline", "academic_level"],
                    "output": {"type": "research"},
                    "step": 1
                },
                {
                    "input": ["WORKFLOW.1"],
                    "output": {"type": "document"},
                    "step": 2
                }
            ]
        }
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Workflow configuration options"
    )
    input_data: Dict[str, Any] = Field(
        ..., 
        description="Input data for the workflow",
        example={
            "research_topic": "Distributed AI Systems",
            "deadline": "2024-12-31",
            "academic_level": "PhD"
        }
    )

@app.post("/workflow/execute")
async def execute_workflow(request: WorkflowRequest):
    """Execute workflow synchronously"""
    try:
        # Create agent config
        config = AgentConfig(
            agent_type=request.config.get('agent_type', 'research'),
            model={
                'provider': request.config.get('provider', 'openai'),
                'name': request.config.get('model', 'gpt-4'),
                'temperature': request.config.get('temperature', 0.5)
            },
            workflow={
                'max_iterations': request.config.get('max_iterations', 10),
                'logging_level': request.config.get('logging_level', 'INFO'),
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
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/workflow/execute_async")
async def execute_workflow_async(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Execute workflow asynchronously"""
    try:
        # Create agent config
        config = AgentConfig(
            agent_type=request.config.get('agent_type', 'research'),
            model={
                'provider': request.config.get('provider', 'openai'),
                'name': request.config.get('model', 'gpt-4'),
                'temperature': request.config.get('temperature', 0.5)
            },
            workflow={
                'max_iterations': request.config.get('max_iterations', 10),
                'logging_level': request.config.get('logging_level', 'INFO'),
                'distributed': True
            }
        )
        
        # Initialize agent
        agent = AgentFlow(config)
        agent.workflow_def = request.workflow
        
        # Execute workflow asynchronously
        result_ref = agent.execute_workflow_async(request.input_data)
        
        # Wait for result
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, ray.get, result_ref
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Async execution failed: {str(e)}")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Async workflow execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/workflow/validate")
async def validate_workflow(request: WorkflowRequest):
    """Validate workflow configuration"""
    try:
        if not request.workflow or 'WORKFLOW' not in request.workflow:
            raise ValueError("Invalid workflow configuration")
            
        # Validate workflow steps
        steps = request.workflow['WORKFLOW']
        if not steps or not isinstance(steps, list):
            raise ValueError("Workflow must contain steps")
            
        for step in steps:
            if not all(k in step for k in ['step', 'input', 'output']):
                raise ValueError("Invalid step configuration")
                
        return {"status": "valid"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/workflow/result/{result_ref}")
async def get_workflow_result(result_ref: str):
    """
    Retrieve the result of an asynchronously executed workflow.
    
    :param result_ref: Reference to the workflow result
    :return: Workflow execution results
    """
    try:
        # Convert string reference back to Ray ObjectRef
        import ray
        ref = ray.ObjectRef(result_ref.encode())
        
        # Retrieve result
        start_time = time.time()
        result = ray.get(ref)
        
        logger.info(f"Workflow result retrieved in {time.time() - start_time:.2f} seconds")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "result": result,
                "retrieval_time": time.time() - start_time
            }
        )
    
    except ray.exceptions.RayError as e:
        logger.error(f"Failed to retrieve workflow result: {str(e)}")
        raise HTTPException(
            status_code=404, 
            detail=f"Workflow result not found or no longer available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error retrieving workflow result: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error retrieving workflow result: {str(e)}"
        )

def start_server(host='0.0.0.0', port=8000):
    """
    Start the FastAPI server for workflow execution.
    
    :param host: Host to bind the server
    :param port: Port to listen on
    """
    uvicorn.run(
        "agentflow.api.workflow_server:app", 
        host=host, 
        port=port, 
        reload=True
    )

if __name__ == "__main__":
    start_server()
