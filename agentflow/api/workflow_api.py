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
                'distributed': False,
                'max_retries': config_dict.get('max_retries', 3),
                'retry_delay': config_dict.get('retry_delay', 1.0),
                'retry_backoff': config_dict.get('retry_backoff', 2.0),
                'timeout': config_dict.get('timeout', 300)
            }
        )

        # Initialize agent
        agent = AgentFlow(config)

        # Process workflow configuration with references
        processed_workflow = {
            "WORKFLOW": []
        }
        previous_step_outputs = {}

        for step in request.workflow['WORKFLOW']:
            # Process input references
            processed_input = []
            for input_item in step.get('input', []):
                if isinstance(input_item, str) and input_item.startswith('WORKFLOW.'):
                    # Extract step number from reference
                    try:
                        ref_parts = input_item.split('.')
                        if len(ref_parts) >= 2:
                            # Try parsing as integer first
                            try:
                                ref_step_num = int(ref_parts[1])
                            except ValueError:
                                # If not an integer, extract the step number
                                ref_step_num = int(ref_parts[1].replace('step_', ''))
                            
                            # Get output from previous step
                            if ref_step_num in previous_step_outputs:
                                step_output = previous_step_outputs[ref_step_num]
                                if isinstance(step_output, dict) and 'output' in step_output:
                                    processed_input.append(step_output['output'])
                                else:
                                    processed_input.append(step_output)
                            else:
                                processed_input.append(input_item)
                        else:
                            processed_input.append(input_item)
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Invalid workflow reference: {input_item}. Error: {str(e)}")
                        processed_input.append(input_item)
                else:
                    processed_input.append(input_item)

            # Prepare processed step
            processed_step = {
                "step": step.get('step', 0),
                "input": processed_input,
                "output": step.get('output', {}),
                "agent_config": step.get('agent_config', {}),
                "type": step.get('type', 'default'),
                "name": step.get('name', f'Step {step.get("step", 0)}'),
                "description": step.get('description', '')
            }
            processed_workflow["WORKFLOW"].append(processed_step)

            # Store step output for reference
            previous_step_outputs[step['step']] = processed_step.get('output', {})

        # Set processed workflow definition
        agent.workflow_def = processed_workflow
        
        # Execute workflow
        result = await agent.execute_workflow(request.input_data)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

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