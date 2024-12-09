"""
AgentFlow: Core Management Class for AI Workflow System
"""

from typing import Dict, Any, List
from .config import WorkflowConfig, AgentConfig, ModelConfig, ExecutionPolicies
from .workflow_executor import WorkflowExecutor
import asyncio

class AgentFlow:
    """
    Main management class for AgentFlow workflow system.
    
    Provides high-level management and execution of AI workflows.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize AgentFlow with optional global configuration.
        
        Args:
            config (Dict[str, Any], optional): Global configuration settings. Defaults to None.
        """
        self.config = config or {}
        self.workflows = {}
        self.active_workflows = {}
        self.workflow_def = None
    
    def _convert_input_for_hashability(self, input_data: Any) -> Any:
        """
        Convert input data to a hashable format
        
        Args:
            input_data: Input data to convert
        
        Returns:
            Hashable version of input data
        """
        def _convert(obj):
            if isinstance(obj, dict):
                # Convert dict to a tuple of sorted items, converting values to hashable types
                return tuple(
                    (str(k), _convert(v)) 
                    for k, v in sorted(obj.items())
                )
            elif isinstance(obj, list):
                # Convert list to tuple, converting elements to hashable types
                return tuple(_convert(x) for x in obj)
            elif isinstance(obj, set):
                # Convert set to tuple of sorted hashable elements
                return tuple(sorted(_convert(x) for x in obj))
            elif isinstance(obj, (dict, list, set)):
                # Recursively convert nested structures
                return str(obj)
            
            # If already hashable, return as is
            return obj

        return _convert(input_data)

    async def execute_workflow(
        self, 
        workflow_id: str = None,
        input_data: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a workflow by ID or using the default workflow definition.
        
        Args:
            workflow_id (str, optional): ID of the workflow to execute. Defaults to None.
            input_data (Dict[str, Any], optional): Input data for workflow. Defaults to None.
        
        Returns:
            List[Dict[str, Any]]: Results of workflow execution
        """
        # Check if workflow ID is provided and exists
        if workflow_id is not None:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            workflow_config = self.workflows[workflow_id]
        else:
            # If no workflow definition is set, raise an error
            if not self.workflow_def or 'WORKFLOW' not in self.workflow_def:
                raise ValueError("No workflow definition available")
            
            # Create a dynamic workflow configuration
            workflow_id = f"workflow_{len(self.workflows) + 1}"
            workflow_config = WorkflowConfig(
                id=workflow_id,
                name='Dynamic Workflow',
                description='Dynamically created workflow',
                agents=[],  # You might want to modify this based on your needs
                execution_policies=ExecutionPolicies(
                    steps=[],  # You might want to modify this based on your needs
                    distributed=False
                )
            )
        
        # Create workflow executor
        workflow_executor = WorkflowExecutor(workflow_config)
        
        # If a workflow definition exists, set it
        if hasattr(self, 'workflow_def') and self.workflow_def:
            # Preprocess workflow steps to ensure hashability
            processed_workflow = {
                "WORKFLOW": []
            }
            
            # Store previous step results to handle WORKFLOW.x references
            previous_step_results = {}
            
            for step in self.workflow_def.get('WORKFLOW', []):
                # Convert input to tuple of strings for hashability
                input_list = []
                for input_item in step.get('input', []):
                    if isinstance(input_item, str) and input_item.startswith('WORKFLOW.'):
                        # Extract step number from reference
                        try:
                            ref_step_num = int(input_item.split('.')[1])
                            if ref_step_num in previous_step_results:
                                input_list.append(previous_step_results[ref_step_num])
                            else:
                                # Keep the reference as is if the step hasn't been processed yet
                                input_list.append(input_item)
                        except (IndexError, ValueError):
                            # If reference is malformed, keep it as is
                            input_list.append(input_item)
                    else:
                        input_list.append(
                            str(x) if isinstance(x, (dict, list, set)) else x 
                            for x in [input_item]
                        )
                
                processed_step = {
                    "step": step.get('step', 0),
                    "input": tuple(input_list),
                    "output": step.get('output', {}),
                    # Preserve the original agent config for reference
                    "agent_config": step.get('agent_config', {})
                }
                processed_workflow["WORKFLOW"].append(processed_step)
            
            workflow_executor.workflow_def = processed_workflow
        
        # Convert input data to hashable format
        if input_data is not None:
            # Create a copy to avoid modifying the original input
            input_data = {
                k: self._convert_input_for_hashability(v) 
                for k, v in input_data.items()
            }
        
        # Store active workflow
        self.active_workflows[workflow_id] = workflow_executor
        
        # Execute workflow
        return await workflow_executor.execute(input_data)
    
    def execute_workflow_async(
        self, 
        workflow_def: Dict[str, Any] = None, 
        input_data: Dict[str, Any] = None
    ) -> Any:
        """
        Execute a workflow asynchronously.
        
        Args:
            workflow_def (Dict[str, Any], optional): Workflow definition. Defaults to None.
            input_data (Dict[str, Any], optional): Input data for workflow. Defaults to None.
        
        Returns:
            Any: Reference to the async workflow execution result
        """
        import ray
        import logging
        logger = logging.getLogger(__name__)
        
        @ray.remote
        def async_workflow_execution(workflow_def, input_data):
            """Wrapper for async workflow execution"""
            try:
                # Create a unique workflow ID
                workflow_id = f"workflow_{hash(str(workflow_def))}"
                logger.info(f"Starting async workflow execution: {workflow_id}")
                
                # Create workflow config
                workflow_config = WorkflowConfig(
                    id=workflow_id,
                    name='Dynamic Async Workflow',
                    description='Dynamically created async workflow',
                    agents=[],
                    execution_policies=ExecutionPolicies(
                        steps=[],
                        distributed=True
                    )
                )
                
                # Create workflow executor
                workflow_executor = WorkflowExecutor(workflow_config)
                workflow_executor.workflow_def = workflow_def
                
                # Execute workflow synchronously (this will be converted to async)
                result = ray.get(workflow_executor.execute(input_data))
                
                logger.info(f"Async workflow {workflow_id} completed successfully")
                return result
            
            except Exception as e:
                logger.error(f"Error in async workflow execution: {str(e)}")
                logger.error(f"Workflow definition: {workflow_def}")
                logger.error(f"Input data: {input_data}")
                raise
        
        # Use workflow definition from the method or from the class
        if workflow_def is None:
            workflow_def = self.workflow_def
        
        # If no workflow definition is available, raise an error
        if not workflow_def or 'WORKFLOW' not in workflow_def:
            logger.error("No workflow definition available")
            raise ValueError("No workflow definition available")
        
        # Submit async task to Ray
        try:
            result_ref = async_workflow_execution.remote(workflow_def, input_data)
            logger.info(f"Async workflow task submitted: {result_ref}")
            return result_ref
        except Exception as e:
            logger.error(f"Failed to submit async workflow task: {str(e)}")
            raise
    
    def create_workflow(self, workflow_id: str, workflow_config: WorkflowConfig) -> Any:
        """
        Create a workflow with the given ID and configuration.
        
        Args:
            workflow_id (str): Unique identifier for the workflow
            workflow_config (WorkflowConfig): Configuration for the workflow
        
        Returns:
            Any: Created workflow executor
        """
        # Check if workflow ID already exists
        if workflow_id in self.workflows:
            raise ValueError(f"Workflow {workflow_id} already exists")
        
        # Create workflow executor
        workflow_executor = WorkflowExecutor(workflow_config)
        
        # Store workflow configuration and executor
        self.workflows[workflow_id] = workflow_config
        
        return workflow_executor
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get status of a specific workflow.
        
        Args:
            workflow_id (str): ID of the workflow
        
        Returns:
            Dict[str, Any]: Current workflow status
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not active")
        
        workflow_executor = self.active_workflows[workflow_id]
        return workflow_executor.get_status()
    
    async def stop_workflow(self, workflow_id: str) -> None:
        """
        Stop an active workflow asynchronously.
        
        Args:
            workflow_id (str): ID of the workflow to stop
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not active")
        
        workflow_executor = self.active_workflows[workflow_id]
        
        # Check if stop method is a coroutine
        if asyncio.iscoroutinefunction(workflow_executor.stop):
            await workflow_executor.stop()
        else:
            workflow_executor.stop()
        
        del self.active_workflows[workflow_id]
    
    def list_workflows(self) -> List[str]:
        """
        List all created workflow IDs.
        
        Returns:
            List[str]: List of workflow IDs
        """
        return list(self.workflows.keys())
