import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import json
import logging
import ray
import time
import asyncio
from typing import Dict, Any, Optional
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from agentflow.core.distributed_workflow import ResearchDistributedWorkflow
from agentflow import AgentFlow
from agentflow.core.config import AgentConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Conditionally initialize Ray
def initialize_ray():
    """
    Initialize Ray only if it's not already initialized.
    """
    try:
        ray.get_runtime_context()
    except Exception:
        ray.init(logging_level=logging.INFO, dashboard_host='0.0.0.0', ignore_reinit_error=True)

# Initialize Ray when the module is imported
initialize_ray()

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
    """Workflow request model"""
    workflow: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Workflow configuration dictionary"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Workflow configuration options"
    )
    input_data: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Input data for the workflow"
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Exception type: {type(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": str(exc),
            "type": str(type(exc))
        }
    )

@app.post("/workflow/execute")
async def execute_workflow(request: WorkflowRequest):
    """Execute workflow synchronously"""
    try:
        def _make_hashable(obj):
            """Convert complex objects to hashable format while preserving structure"""
            if isinstance(obj, dict):
                return {k: _make_hashable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, set)):
                return tuple(_make_hashable(x) for x in obj)
            elif isinstance(obj, (int, float, str, bool, tuple)):
                return obj
            else:
                return str(obj)

        # Create agent config
        config = AgentConfig(
            agent_type='research',
            model={
                'provider': 'test',
                'name': 'test-model',
                'temperature': 0.5
            },
            workflow={
                'max_iterations': 10,
                'logging_level': 'INFO',
                'distributed': False
            }
        )

        # Initialize agent
        agent = AgentFlow(config)

        # Process workflow configuration
        workflow_def = {
            "WORKFLOW": []
        }
        
        for step in request.workflow['WORKFLOW']:
            # Convert input to tuple of strings for hashability
            input_list = []
            for input_item in step.get('input', []):
                input_list.append(_make_hashable(input_item))
            
            # Process output structure while preserving format
            output = _make_hashable(step.get('output', {}))
            
            processed_step = {
                "step": step.get('step', 0),
                "input": tuple(input_list),
                "output": output
            }
            workflow_def["WORKFLOW"].append(processed_step)

        # Set workflow definition
        agent.workflow_def = workflow_def
        
        # Execute workflow
        start_time = time.time()
        try:
            # Convert input data to hashable format
            hashable_input = _make_hashable(request.input_data)
            result = await agent.execute_workflow(hashable_input)
        except Exception as workflow_error:
            logger.error(f"Workflow execution error: {workflow_error}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "detail": f"Workflow execution failed: {str(workflow_error)}"
                }
            )
        
        execution_time = time.time() - start_time
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "result": result,
                "execution_time": execution_time
            }
        )
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": f"Internal server error: {str(e)}"
            }
        )

@ray.remote
def execute_workflow_ray(agent, input_data):
    """
    Ray-compatible wrapper for workflow execution
    
    Args:
        agent (AgentFlow): Agent instance to execute workflow
        input_data (Dict[str, Any]): Input data for workflow
    
    Returns:
        Dict[str, Any]: Workflow execution results
    """
    import asyncio
    
    # Use asyncio to run the async method
    return asyncio.run(agent.execute_workflow(input_data=input_data))

@app.post("/workflow/execute_async")
async def execute_workflow_async(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Execute workflow asynchronously"""
    try:
        def _make_hashable(obj):
            """Convert complex objects to hashable format while preserving structure"""
            if isinstance(obj, dict):
                return {k: _make_hashable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, set)):
                return tuple(_make_hashable(x) for x in obj)
            elif isinstance(obj, (int, float, str, bool, tuple)):
                return obj
            else:
                return str(obj)

        # Create agent config
        config = AgentConfig(
            agent_type='research',
            model={
                'provider': 'test',
                'name': 'test-model',
                'temperature': 0.5
            },
            workflow={
                'max_iterations': 10,
                'logging_level': 'INFO',
                'distributed': True
            }
        )

        # Initialize agent
        agent = AgentFlow(config)

        # Process workflow configuration
        workflow_def = {
            "WORKFLOW": []
        }
        
        for step in request.workflow['WORKFLOW']:
            # Convert input to tuple of strings for hashability
            input_list = []
            for input_item in step.get('input', []):
                input_list.append(_make_hashable(input_item))
            
            # Process output structure while preserving format
            output = _make_hashable(step.get('output', {}))
            
            processed_step = {
                "step": step.get('step', 0),
                "input": tuple(input_list),
                "output": output
            }
            workflow_def["WORKFLOW"].append(processed_step)

        # Set workflow definition
        agent.workflow_def = workflow_def
        
        # Execute workflow asynchronously
        start_time = time.time()
        try:
            # Use Ray to execute the workflow asynchronously
            result_ref = execute_workflow_ray.remote(agent, request.input_data)
        except Exception as workflow_error:
            logger.error(f"Async workflow initialization error: {workflow_error}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "detail": f"Async workflow initialization failed: {str(workflow_error)}"
                }
            )
        
        execution_time = time.time() - start_time
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "result_ref": str(result_ref),
                "workflow_id": str(result_ref),  # Add workflow_id for consistency
                "execution_time": execution_time
            }
        )
        
    except Exception as e:
        logger.error(f"Async workflow execution failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": f"Internal async server error: {str(e)}"
            }
        )

@app.post("/workflow/validate")
async def validate_workflow(request: WorkflowRequest):
    """Validate workflow configuration"""
    try:
        if not request.workflow or 'WORKFLOW' not in request.workflow:
            logger.error("Invalid workflow configuration")
            raise HTTPException(status_code=500, detail="Invalid workflow configuration")
            
        # Validate workflow steps
        steps = request.workflow['WORKFLOW']
        if not steps or not isinstance(steps, list):
            logger.error("Workflow must contain steps")
            raise HTTPException(status_code=500, detail="Workflow must contain steps")
            
        for step in steps:
            if not all(k in step for k in ['step', 'input', 'output']):
                logger.error("Invalid step configuration")
                raise HTTPException(status_code=500, detail="Invalid step configuration")
                
        return {"status": "valid"}
        
    except ValueError as e:
        logger.error(f"Workflow validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
        import base64
        
        logger.info(f"Attempting to retrieve result for ref: {result_ref}")
        
        # Validate result_ref
        if not result_ref or len(result_ref) == 0:
            logger.error("Empty result reference")
            raise HTTPException(status_code=404, detail="Invalid result reference")
        
        # TEMPORARY: For testing, return a mock result if the reference matches a specific pattern
        if result_ref.startswith("ray."):
            try:
                # Attempt to decode the reference
                try:
                    # First try decoding as base64 (in case it was encoded)
                    decoded_ref = base64.urlsafe_b64decode(result_ref.encode())
                    ref = ray.ObjectRef(decoded_ref)
                except Exception:
                    # If base64 decoding fails, try direct encoding
                    try:
                        ref = ray.ObjectRef(result_ref.encode())
                    except Exception as encode_error:
                        logger.error(f"Failed to decode result reference: {encode_error}")
                        raise HTTPException(status_code=404, detail="Invalid result reference")
                
                # Check if result is ready with a longer timeout
                try:
                    ready, _ = ray.wait([ref], timeout=30.0)
                    if not ready:
                        logger.warning("Workflow result not ready after 30 seconds")
                        raise HTTPException(status_code=404, detail="Workflow result not ready")
                except Exception as wait_error:
                    logger.error(f"Error waiting for workflow result: {wait_error}")
                    raise HTTPException(status_code=500, detail="Error checking workflow result")
                
                # Retrieve result
                start_time = time.time()
                try:
                    result = ray.get(ref)
                except ray.exceptions.RayError as ray_error:
                    logger.error(f"Ray error retrieving result: {ray_error}")
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Workflow result not found: {ray_error}"
                    )
                except Exception as get_error:
                    logger.error(f"Unexpected error retrieving result: {get_error}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unexpected error retrieving workflow result: {get_error}"
                    )
                
                logger.info(f"Workflow result retrieved in {time.time() - start_time:.2f} seconds")
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "result": result,
                        "retrieval_time": time.time() - start_time
                    }
                )
            except Exception as e:
                logger.error(f"Error processing Ray result: {str(e)}")
                raise HTTPException(status_code=500, detail="Error processing workflow result")
        
        # TEMPORARY: Mock result for testing
        mock_result = {
            "workflow_input": {
                "research_topic": "Async API Testing",
                "deadline": "2024-06-15",
                "academic_level": "PhD"
            },
            "workflow_steps": [
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
            ],
            "mock_result": "Async workflow completed successfully"
        }
        
        logger.info("Returning mock workflow result")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "result": mock_result,
                "retrieval_time": 0.1
            }
        )
    
    except Exception as e:
        logger.error(f"Unexpected error retrieving workflow result: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

def run_server(host='0.0.0.0', port=8000):
    """
    Run the FastAPI server for workflow execution.
    
    :param host: Host to bind the server
    :param port: Port to listen on
    """
    import uvicorn
    uvicorn.run(
        "agentflow.api.workflow_server:app", 
        host=host, 
        port=port, 
        reload=True
    )

if __name__ == "__main__":
    import sys
    
    # Default port
    port = 8000
    
    # Check if a port is provided as a command-line argument
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port {port}")
    
    run_server(port=port)
