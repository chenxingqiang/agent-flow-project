import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from .base import BaseWorkflow
from .config import WorkflowConfig, NodeState
from .exceptions import WorkflowExecutionError, WorkflowTimeoutError
from .workflow import WorkflowNode

logger = logging.getLogger(__name__)

class WorkflowEngine(BaseWorkflow):
    """Workflow execution engine."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get("timeout", 60)  # Default 60 seconds
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1)  # Default 1 second
        
    async def execute_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow node with timeout and retry logic."""
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                # Create task with timeout
                async with asyncio.timeout(self.timeout):
                    result = await node.execute(inputs)
                return result
                
            except asyncio.TimeoutError:
                if retries == self.max_retries:
                    raise WorkflowTimeoutError(
                        f"Node {node.id} execution timed out after {self.timeout} seconds"
                    )
                    
            except Exception as e:
                if retries == self.max_retries:
                    if last_error:
                        raise WorkflowExecutionError(
                            f"Node {node.id} failed after {retries} retries. "
                            f"Last error: {str(last_error)}"
                        ) from last_error
                    raise
                last_error = e
                
            retries += 1
            if retries <= self.max_retries:
                await asyncio.sleep(self.retry_delay * retries)  # Exponential backoff
                
        raise WorkflowExecutionError(f"Node {node.id} failed after {retries} retries")
        
    async def execute_workflow(self, workflow: WorkflowConfig) -> Dict[str, Any]:
        """Execute workflow with error handling."""
        try:
            # Initialize workflow state
            self.workflow = workflow
            self.state = {}
            
            # Build execution graph
            graph = self._build_execution_graph()
            
            # Execute nodes in dependency order
            results = {}
            for node in graph:
                # Get node inputs
                node_inputs = self._get_node_inputs(node, results)
                
                try:
                    # Execute node with timeout and retry
                    result = await self.execute_node(node, node_inputs)
                    results[node.id] = result
                    
                except (WorkflowTimeoutError, WorkflowExecutionError) as e:
                    # Handle node execution errors
                    self._handle_node_error(node, e)
                    raise
                    
            return results
            
        except Exception as e:
            # Handle workflow-level errors
            self._handle_workflow_error(e)
            raise
            
    def _handle_node_error(self, node: WorkflowNode, error: Exception) -> None:
        """Handle node execution error."""
        # Log error
        logger.error(f"Error executing node {node.id}: {str(error)}")
        
        # Update node state
        node.state = NodeState.FAILURE
        
        # Store error in workflow state
        self.state[node.id] = {
            "status": "error",
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
    def _handle_workflow_error(self, error: Exception) -> None:
        """Handle workflow-level error."""
        # Log error
        logger.error(f"Workflow execution error: {str(error)}")
        
        # Update workflow state
        self.state["workflow"] = {
            "status": "error",
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
    def _build_execution_graph(self) -> List[WorkflowNode]:
        """Build execution graph from workflow config."""
        # Create nodes
        nodes = {}
        for step in self.workflow.steps:
            node = WorkflowNode(
                id=step.id,
                name=step.name,
                type=step.type,
                config=step.config,
                inputs=step.inputs,
                outputs=step.outputs,
                dependencies=step.dependencies
            )
            nodes[step.id] = node
            
        # Sort nodes by dependencies
        sorted_nodes = []
        visited = set()
        
        def visit(node_id):
            if node_id in visited:
                return
            node = nodes[node_id]
            for dep in node.dependencies:
                visit(dep)
            visited.add(node_id)
            sorted_nodes.append(nodes[node_id])
            
        for node_id in nodes:
            visit(node_id)
            
        return sorted_nodes
        
    def _get_node_inputs(self, node: WorkflowNode, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get inputs for a node from previous results."""
        inputs = {}
        for input_id in node.inputs:
            if input_id in results:
                inputs[input_id] = results[input_id]
        return inputs 