import sys
import os
import uuid

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import json
import logging
import ray
import time
import asyncio
import traceback
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, ValidationError

from agentflow.core.research_workflow import ResearchDistributedWorkflow, DistributedConfig
from agentflow.core.config import AgentConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Dictionary to store workflow tasks and their status
task_refs: Dict[str, Dict[str, Any]] = {}

# Conditionally initialize Ray
def initialize_ray():
    """
    Initialize Ray only if it's not already initialized.
    """
    try:
        if not ray.is_initialized():
            ray.init(
                logging_level=logging.INFO,
                dashboard_host='0.0.0.0',
                ignore_reinit_error=True
            )
            logger.info("Ray initialized with basic configuration")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        raise

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

    @validator('workflow')
    def validate_workflow(cls, workflow):
        """Validate workflow configuration"""
        # Accept either 'workflow_steps' or 'WORKFLOW'
        workflow_steps = workflow.get('workflow_steps') or workflow.get('WORKFLOW')
        
        if not workflow_steps:
            raise ValueError("No workflow steps found")
        
        if not isinstance(workflow_steps, list):
            raise ValueError("Workflow steps must be a list")
        
        for step in workflow_steps:
            if not all(k in step for k in ['step', 'input', 'output']):
                raise ValueError(f"Invalid step configuration: {step}")
        
        # Normalize workflow configuration
        workflow['WORKFLOW'] = workflow_steps
        return workflow

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
        logger.debug(f"Received workflow request: {request}")
        
        # Validate input data
        input_data = request.input_data or {}
        
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Create distributed config from request config
        dist_config = DistributedConfig(
            num_workers=request.config.get('num_workers', 4),
            batch_size=request.config.get('batch_size', 10),
            timeout=request.config.get('timeout', 30.0),
            retry_limit=request.config.get('max_retries', 3),
            worker_init_timeout=request.config.get('worker_init_timeout', 5.0)
        )

        # Create workflow instance with proper name and config
        workflow = ResearchDistributedWorkflow(
            name=f"workflow_{workflow_id}",
            config=dist_config
        )
        
        # Store workflow info
        task_refs[workflow_id] = {
            'workflow_def': request.workflow,
            'config': request.config or {},
            'input_data': input_data,
            'status': 'pending',
            'result': None,
            'error': None
        }
        
        # Extract research parameters from input data
        research_input = {
            "RESEARCH_TOPIC": input_data.get("STUDENT_NEEDS", {}).get("RESEARCH_TOPIC", ""),
            "ACADEMIC_LEVEL": input_data.get("STUDENT_NEEDS", {}).get("ACADEMIC_LEVEL", ""),
            "LANGUAGE": input_data.get("LANGUAGE", {}),
            "TEMPLATE": input_data.get("TEMPLATE", "")
        }
        
        # Execute workflow with research input
        try:
            result = await workflow.execute(research_input)
            task_refs[workflow_id]['status'] = 'completed'
            task_refs[workflow_id]['result'] = result
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "workflow_id": workflow_id,
                    "result": result
                }
            )
        except Exception as e:
            task_refs[workflow_id]['status'] = 'failed'
            task_refs[workflow_id]['error'] = str(e)
            logger.error(f"Error executing workflow: {e}")
            raise
        
    except Exception as e:
        logger.error(f"Unexpected error in workflow execution: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/execute_async")
async def execute_workflow_async(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Execute workflow asynchronously"""
    logger.info("Received async workflow execution request")
    try:
        # Validate input data
        input_data = request.input_data or {}
        
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Create distributed config from request config
        dist_config = DistributedConfig(
            num_workers=request.config.get('num_workers', 4),
            batch_size=request.config.get('batch_size', 10),
            timeout=request.config.get('timeout', 30.0),
            retry_limit=request.config.get('max_retries', 3),
            worker_init_timeout=request.config.get('worker_init_timeout', 5.0)
        )

        # Create workflow instance with proper name and config
        workflow = ResearchDistributedWorkflow(
            name=f"workflow_{workflow_id}",
            config=dist_config
        )
        
        # Extract research parameters from input data
        research_input = {
            "RESEARCH_TOPIC": input_data.get("STUDENT_NEEDS", {}).get("RESEARCH_TOPIC", ""),
            "ACADEMIC_LEVEL": input_data.get("STUDENT_NEEDS", {}).get("ACADEMIC_LEVEL", ""),
            "LANGUAGE": input_data.get("LANGUAGE", {}),
            "TEMPLATE": input_data.get("TEMPLATE", "")
        }
        
        # Store workflow info
        task_refs[workflow_id] = {
            'workflow_def': request.workflow,
            'config': request.config or {},
            'input_data': research_input,
            'status': 'pending',
            'result': None,
            'error': None,
            'workflow': workflow
        }
        
        # Start background task for execution
        async def execute_workflow_task():
            try:
                result = await workflow.execute(research_input)
                task_refs[workflow_id]['status'] = 'completed'
                task_refs[workflow_id]['result'] = result
            except Exception as e:
                task_refs[workflow_id]['status'] = 'failed'
                task_refs[workflow_id]['error'] = str(e)
                logger.error(f"Error executing workflow: {e}")
        
        background_tasks.add_task(execute_workflow_task)
        task_refs[workflow_id]['status'] = 'running'
        
        # Return workflow ID
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "workflow_id": workflow_id
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in async workflow execution: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get workflow execution status"""
    try:
        # Check if workflow exists
        if workflow_id not in task_refs:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_info = task_refs[workflow_id]
        
        # If workflow is completed, return the result
        if workflow_info['status'] == 'completed':
            return {
                "status": "completed",
                "result": workflow_info['result']
            }
        
        # If workflow has a result reference, check its status
        if workflow_info.get('result_ref') is not None:
            try:
                # Try to get the result
                result = ray.get(workflow_info['result_ref'])
                workflow_info['status'] = 'completed'
                workflow_info['result'] = result
                return {
                    "status": "completed",
                    "result": result
                }
            except ray.exceptions.GetTimeoutError:
                return {"status": "running"}
            except Exception as e:
                logger.error(f"Error checking workflow status: {e}")
                return {"status": "running"}  # Return running instead of failed
        
        # Return current status
        return {"status": workflow_info['status']}
        
    except Exception as e:
        logger.error(f"Error checking workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _format_workflow_result(result: Any) -> Any:
    """Format workflow result to ensure consistent structure"""
    if result is None:
        return {'output': {'result': None}}
        
    if not isinstance(result, dict):
        return {'output': {'result': result}}
        
    # If result already has a properly structured output
    if 'output' in result and isinstance(result['output'], dict):
        return result
    
    # If result has step results, format them
    if any(key.startswith('step_') for key in result.keys()):
        formatted_steps = {}
        final_result = None
        for key, value in result.items():
            if key.startswith('step_'):
                step_num = int(key.split('_')[1])
                if isinstance(value, dict):
                    if 'output' in value:
                        formatted_steps[key] = value
                    else:
                        formatted_steps[key] = {'output': value}
                else:
                    formatted_steps[key] = {'output': {'result': value}}
                # Keep track of the last step's result
                final_result = formatted_steps[key]['output']
        # Return both the final result and all step results
        return {
            'output': final_result,
            'step_results': formatted_steps
        }
    
    # Otherwise, wrap the entire result
    return {'output': result}

# Create a Ray actor for handling async workflow execution
@ray.remote
class WorkflowActor:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def execute_workflow(self, workflow_ref, input_ref):
        """Execute workflow and handle coroutines"""
        # Check if workflow_ref is a Ray object reference
        if isinstance(workflow_ref, ray.ObjectRef):
            workflow = ray.get(workflow_ref)
        else:
            workflow = workflow_ref

        # Check if input_ref is a Ray object reference
        if isinstance(input_ref, ray.ObjectRef):
            input_data = ray.get(input_ref)
        else:
            input_data = input_ref

        # Execute workflow
        result = workflow.execute(input_data)
        
        # If result is a coroutine, run it in the event loop
        if asyncio.iscoroutine(result):
            result = self.loop.run_until_complete(result)
            
        return result

@app.post("/workflow/validate")
async def validate_workflow(request: WorkflowRequest):
    """Validate workflow configuration"""
    try:
        return {"status": "valid"}
        
    except ValueError as e:
        logger.error(f"Workflow validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def run_server(host='0.0.0.0', port=8000):
    """
    Run the FastAPI server for workflow execution.
    
    :param host: Host to bind the server
    :param port: Port to listen on
    """
    try:
        # Ensure Ray is initialized
        initialize_ray()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Output to console
                logging.FileHandler('workflow_server.log')  # Output to file
            ]
        )
        
        # Log server startup details
        logger.info(f"Starting workflow server on {host}:{port}")
        
        # Explicitly set the event loop policy for subprocess
        if sys.platform == 'darwin':  # macOS
            import asyncio
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        
        # Run the server using uvicorn
        uvicorn.run(
            "agentflow.api.workflow_server:app", 
            host=host, 
            port=port, 
            reload=False,
            workers=1,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Write error to a file for debugging
        with open('server_startup_error.log', 'w') as f:
            f.write(f"Server startup error: {e}\n")
            traceback.print_exc(file=f)
        raise

if __name__ == "__main__":
    import sys
    import os
    import uvicorn

    # Default port
    port = 8000

    # Check if a port is provided as a command-line argument
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port {port}")

    # Run the server
    run_server(host='0.0.0.0', port=port)
