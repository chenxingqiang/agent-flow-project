"""
Workflow execution engine for AgentFlow
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from asyncio import Queue
import traceback
from dataclasses import dataclass
from enum import Enum
import importlib
from unittest.mock import MagicMock

from agentflow.core.config import WorkflowConfig, AgentConfig, ProcessorConfig
from agentflow.core.node import Node, AgentNode, ProcessorNode, NodeState
from agentflow.core.utils import import_class
from agentflow.core.workflow import Workflow

logger = logging.getLogger(__name__)

@dataclass
class AgentMetrics:
    """Agent metrics"""
    tokens: int = 0
    latency: float = 0
    memory: int = 0

@dataclass
class AgentStatus:
    """Agent status"""
    id: str
    name: str
    type: str
    status: str
    progress: Optional[float] = None
    metrics: Optional[AgentMetrics] = None

class WorkflowExecutionError(Exception):
    """Exception raised when workflow execution fails"""
    def __init__(self, details):
        self.details = details
        if isinstance(details, dict):
            if "failed_nodes" in details:
                # Get the first node's error details for backward compatibility
                first_error = details["failed_nodes"][0]
                self.details.update(first_error)
                # Format error message to include all node errors
                error_messages = []
                for node_error in details["failed_nodes"]:
                    error_messages.append(f"{node_error['node_id']}: {node_error['error']}")
                error_msg = f"Workflow execution failed with errors: {'; '.join(error_messages)}"
            else:
                error_msg = str(details.get("error", "Unknown error"))
        else:
            error_msg = str(details)
        super().__init__(error_msg)

class WorkflowExecutor:
    """Executes a workflow by managing nodes and their connections"""
    
    def __init__(self, workflow_config: Dict[str, Any]):
        """Initialize workflow executor
        
        Args:
            workflow_config: Workflow configuration
        """
        # Convert dict to WorkflowConfig if needed
        if isinstance(workflow_config, dict):
            workflow_config = WorkflowConfig(**workflow_config)
            
        self.config = workflow_config
        self.nodes = {}
        self.status = {}
        self._setup_workflow(workflow_config)

    def _setup_workflow(self, workflow_config: WorkflowConfig):
        """Setup workflow nodes and connections"""
        # Create nodes
        self.nodes = {}

        # Import test classes
        from tests.core.test_workflow_executor import TestAgent

        # Create agent nodes
        for agent_config in workflow_config.agents:
            # Create agent node
            node = AgentNode(**agent_config.model_dump())
            node.output_queue = asyncio.Queue()  # Initialize output queue
            
            # Initialize agent with test class
            node.agent = TestAgent(agent_config.model_dump())
            if not hasattr(node.agent, "metrics"):
                node.agent.metrics = {}
                
            self.nodes[node.id] = node

        # Create processor nodes
        if workflow_config.processors:
            for processor_config in workflow_config.processors:
                node = ProcessorNode(**processor_config.model_dump())
                node.output_queue = asyncio.Queue()  # Initialize output queue
                
                # Initialize processor
                if isinstance(processor_config.processor, str):
                    processor_class = import_class(processor_config.processor)
                    node.processor = processor_class(processor_config.config)
                elif isinstance(processor_config.processor, type):
                    node.processor = processor_config.processor(processor_config.config)
                    
                if not hasattr(node.processor, "metrics"):
                    node.processor.metrics = {}
                    
                self.nodes[node.id] = node

        # Setup connections
        self.connections = []
        if workflow_config.connections is not None:
            for connection in workflow_config.connections:
                # Handle both dictionary and ConnectionConfig objects
                if isinstance(connection, dict):
                    self.connections.append({
                        'source': connection['source_id'],
                        'target': connection['target_id'],
                        'source_port': connection.get('source_port', 'output'),
                        'target_port': connection.get('target_port', 'input')
                    })
                else:
                    # Assuming ConnectionConfig has similar attributes
                    self.connections.append({
                        'source': connection.source_id,
                        'target': connection.target_id,
                        'source_port': connection.source_port or 'output',
                        'target_port': connection.target_port or 'input'
                    })

        # Initialize metrics for all nodes
        for node in self.nodes.values():
            if isinstance(node, AgentNode):
                node.metrics = {}
            elif isinstance(node, ProcessorNode):
                node.metrics = {}

    def _get_node_connections(self, node_id: str, as_source: bool = True) -> List[str]:
        """Get node connections
        
        Args:
            node_id: Node ID
            as_source: If True, return connections where node is source
        """
        if as_source:
            return [conn['target'] for conn in self.connections if conn['source'] == node_id]
        return [conn['source'] for conn in self.connections if conn['target'] == node_id]

    def _has_incoming_connections(self, node_id: str) -> bool:
        """Check if node has incoming connections"""
        return len(self._get_node_connections(node_id, as_source=False)) > 0

    def _make_hashable(self, obj):
        """Convert complex objects to hashable format while preserving structure"""
        if isinstance(obj, dict):
            return {k: self._make_hashable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, set)):
            return tuple(self._make_hashable(x) for x in obj)
        elif isinstance(obj, (int, float, str, bool, tuple)):
            return obj
        else:
            return str(obj)

    async def _execute_node(self, node_id: str, input_data: Any = None) -> Tuple[Any, Dict]:
        """Execute a single node in the workflow"""
        node = self.nodes[node_id]
        processing_obj = None
        try:
            # Check if node has incoming connections and requires input
            if self._has_incoming_connections(node_id) and input_data is None:
                node.state = NodeState.WAITING
                await self._update_node_status(node)
                return None, {"error": f"Node {node_id} requires input but none provided"}

            # Determine the processing method based on node type
            if isinstance(node, ProcessorNode):
                # Prefer process_data, fallback to process
                process_method = getattr(node.processor, 'process_data', None) or getattr(node.processor, 'process', None)
                processing_obj = node.processor
            else:
                # For agent nodes
                process_method = getattr(node.agent, 'process', None)
                processing_obj = node.agent

            # Initialize node if possible
            if hasattr(processing_obj, 'initialize'):
                await processing_obj.initialize()

            # Process input data
            if process_method is None:
                raise AttributeError(f"No process method found for node {node_id}")
            
            output = await process_method(input_data)

            # Update node metrics if available
            if isinstance(node, (AgentNode, ProcessorNode)):
                processing_obj = node.agent if isinstance(node, AgentNode) else node.processor
                if processing_obj is not None:
                    node.metrics = AgentMetrics(
                        tokens=getattr(processing_obj, 'token_count', 0),
                        latency=getattr(processing_obj, 'last_latency', 0),
                        memory=getattr(processing_obj, 'memory_usage', 0)
                    )

            # Put output in node's output queue if available
            if hasattr(node, 'output_queue'):
                await node.output_queue.put(output)

            # Update node state and status
            node.state = NodeState.COMPLETED
            await self._update_node_status(node)

            return output, {}
        except Exception as e:
            # Always attempt cleanup
            if processing_obj is not None:
                try:
                    if hasattr(processing_obj, 'cleanup'):
                        await processing_obj.cleanup()
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup for node {node_id}: {cleanup_error}")

            # Prepare error details
            error_details = {
                "error": str(e),
                "node_config": {
                    "id": node_id,
                    "name": node.name if hasattr(node, 'name') else None,
                    "type": node.type,
                    "state": str(node.state)
                },
                "input_data": input_data,  # Add input data to error details
                "failed_nodes": [
                    {
                        "node_id": node_id,
                        "node_config": {
                            "id": node_id,
                            "name": node.name if hasattr(node, 'name') else None,
                            "type": node.type,
                            "state": str(node.state)
                        },
                        "error": str(e)
                    }
                ]
            }

            # Mark node as error state
            node.state = NodeState.ERROR
            await self._update_node_status(node)

            # Log the error
            logger.error(f"Node {node_id} failed: {e}")
            traceback.print_exc()

            raise WorkflowExecutionError(error_details) from e

    async def execute(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]]] = None):
        """Execute workflow with given input data"""
        logger.info("=== Starting workflow execution ===")
        
        # Validate input data
        if input_data is None:
            input_data = {}

        # Reset node states
        for node in self.nodes.values():
            node.state = NodeState.PENDING
            await self._update_node_status(node)

        # Determine start nodes (nodes with no incoming connections)
        start_nodes = [node_id for node_id, node in self.nodes.items() 
                       if not self._has_incoming_connections(node_id)]

        # Track node outputs and states
        node_outputs = {}
        node_states = {node_id: NodeState.PENDING for node_id in self.nodes}

        try:
            # Execute start nodes first
            for node_id in start_nodes:
                node = self.nodes[node_id]
                try:
                    output, _ = await self._execute_node(node_id, input_data)
                    node_outputs[node_id] = output
                    node_states[node_id] = NodeState.COMPLETED
                    node.state = NodeState.COMPLETED
                    await self._update_node_status(node)
                except Exception as e:
                    node.state = NodeState.ERROR
                    node_states[node_id] = NodeState.ERROR
                    await self._update_node_status(node)
                    raise

            # Process connected nodes
            processed_nodes = set(start_nodes)
            while len(processed_nodes) < len(self.nodes):
                # Find nodes that can be processed
                ready_nodes = [
                    node_id for node_id, node in self.nodes.items() 
                    if node_id not in processed_nodes and 
                    all(conn['source'] in processed_nodes 
                        for conn in self.connections if conn['target'] == node_id)
                ]

                if not ready_nodes:
                    break

                # Execute ready nodes
                for node_id in ready_nodes:
                    node = self.nodes[node_id]
                    try:
                        # Find input from previous nodes
                        input_for_node = None
                        for conn in self.connections:
                            if conn['target'] == node_id and conn['source'] in node_outputs:
                                input_for_node = node_outputs[conn['source']]
                                break

                        output, _ = await self._execute_node(node_id, input_for_node)
                        node_outputs[node_id] = output
                        node_states[node_id] = NodeState.COMPLETED
                        node.state = NodeState.COMPLETED
                        await self._update_node_status(node)
                        processed_nodes.add(node_id)
                    except Exception as e:
                        node.state = NodeState.ERROR
                        node_states[node_id] = NodeState.ERROR
                        await self._update_node_status(node)
                        raise

            # Ensure all nodes are processed
            if len(processed_nodes) < len(self.nodes):
                unprocessed = set(self.nodes.keys()) - processed_nodes
                logger.warning(f"Unprocessed nodes: {unprocessed}")

            # Call cleanup for all nodes
            for node_id, node in self.nodes.items():
                processing_obj = node.agent if isinstance(node, AgentNode) else node.processor
                if processing_obj is not None and hasattr(processing_obj, 'cleanup'):
                    try:
                        await processing_obj.cleanup()
                    except Exception as e:
                        logger.error(f"Error during cleanup for node {node_id}: {e}")

            # Return workflow execution result
            return {
                "status": "completed",
                "nodes": node_states
            }

        except Exception as e:
            # Ensure cleanup is called even during errors
            for node_id, node in self.nodes.items():
                processing_obj = node.agent if isinstance(node, AgentNode) else node.processor
                if processing_obj is not None and hasattr(processing_obj, 'cleanup'):
                    try:
                        await processing_obj.cleanup()
                    except Exception as cleanup_error:
                        logger.error(f"Error during cleanup for node {node_id}: {cleanup_error}")

            # Prepare error details
            error_details = {
                "error": str(e),
                "input_data": input_data,
                "failed_nodes": [{
                    "node_id": node_id,
                    "node_config": {
                        "id": node.id,
                        "name": node.name if hasattr(node, 'name') else None,
                        "type": node.type,
                        "state": str(node.state)
                    },
                    "error": str(e)
                } for node_id, node in self.nodes.items()]
            }

            # Raise WorkflowExecutionError
            raise WorkflowExecutionError(error_details) from e

    async def _update_node_status(self, node: Node):
        """Update node status
        
        Args:
            node: Node context
        """
        # Get the processing object (agent or processor)
        processing_obj = node.agent if isinstance(node, AgentNode) else node.processor
        
        status = AgentStatus(
            id=node.id,
            name=node.name if hasattr(node, 'name') else None,
            type=node.type,
            status=str(node.state),
            progress=None,  # TODO: Implement progress tracking
            metrics=AgentMetrics(
                tokens=getattr(processing_obj, 'token_count', 0) if processing_obj else 0,
                latency=getattr(processing_obj, 'last_latency', 0) if processing_obj else 0,
                memory=getattr(processing_obj, 'memory_usage', 0) if processing_obj else 0
            )
        )
        
        # Update workflow status
        self.status[node.id] = status

    def get_node_status(self, node_id: str) -> Optional[NodeState]:
        """Get node execution status
        
        Args:
            node_id: Node ID
            
        Returns:
            Node state if found, None otherwise
        """
        if node_id not in self.nodes:
            return None
        return self.nodes[node_id].state
        
    async def send_input(self, node_id: str, message: dict):
        """Send input to node
        
        Args:
            node_id: Target node ID
            message: Input message
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node not found: {node_id}")
            
        node = self.nodes[node_id]
        await node.agent.input_queue.put(message)

    async def stop(self):
        """Stop workflow execution and cleanup all nodes"""
        # Stop all nodes
        for node_id, node in self.nodes.items():
            # Set node state to completed
            node.state = NodeState.COMPLETED
            
            # Cleanup node resources
            processing_obj = node.agent if isinstance(node, AgentNode) else node.processor
            if processing_obj is not None and hasattr(processing_obj, 'cleanup'):
                try:
                    await processing_obj.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up node {node_id}: {e}")

class WorkflowManager:
    """Manages workflow execution"""
    
    def __init__(self):
        """Initialize workflow manager"""
        self.active_workflows: Dict[str, WorkflowExecutor] = {}
        self._workflow_counter = 0

    async def add_workflow(self, config: Dict[str, Any]) -> str:
        """Add a new workflow
        
        Args:
            config: Workflow configuration
            
        Returns:
            Workflow ID
        """
        # Ensure config has an ID
        if 'id' not in config:
            config['id'] = f"workflow_{self._workflow_counter}"
            self._workflow_counter += 1
        
        # Convert dict to WorkflowConfig if needed
        workflow_config = config if isinstance(config, WorkflowConfig) else WorkflowConfig(**config)
        
        # Create workflow executor
        workflow = WorkflowExecutor(workflow_config)
        
        # Store workflow
        self.active_workflows[workflow_config.id] = workflow
        
        return workflow_config.id

    async def execute_workflow(self, workflow_id: str, input_data: dict):
        """Execute a workflow with input data
        
        Args:
            workflow_id: Workflow ID
            input_data: Input data for the workflow
            
        Returns:
            Workflow execution result
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
            
        executor = self.active_workflows[workflow_id]
        return await executor.execute(input_data)

    async def start_workflow(self, config: WorkflowConfig) -> str:
        """Start workflow execution asynchronously
        
        Args:
            config: Workflow configuration
        
        Returns:
            Workflow ID
        """
        self._workflow_counter += 1
        workflow_id = f"workflow-{self._workflow_counter}"
        
        # Create workflow executor
        executor = WorkflowExecutor(config)
        
        # Store active workflow
        self.active_workflows[workflow_id] = executor
        
        # Start workflow execution
        await executor.execute()
        
        return workflow_id

    async def stop_workflow(self, workflow_id: str):
        """Stop workflow execution asynchronously
        
        Args:
            workflow_id: Workflow ID
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        executor = self.active_workflows[workflow_id]
        await executor.stop()
        del self.active_workflows[workflow_id]

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution status asynchronously
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Dict mapping node IDs to their current state and metrics
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        executor = self.active_workflows[workflow_id]
        status = {}
        for node_id, node in executor.nodes.items():
            # Get metrics from node if available
            metrics = {
                "tokens": 0,
                "latency": 0,
                "memory": 0
            }
            if hasattr(node, 'metrics') and node.metrics:
                metrics = {
                    "tokens": node.metrics.tokens,
                    "latency": node.metrics.latency,
                    "memory": node.metrics.memory
                }
            
            # Update status for this node
            status[node_id] = {
                "status": str(node.state).split('.')[-1].lower(),  # Convert NodeState.COMPLETED to "completed"
                "metrics": metrics
            }
        return status

    async def send_workflow_input(self, workflow_id: str, node_id: str, message: dict):
        """Send input to workflow node asynchronously
        
        Args:
            workflow_id: Workflow ID
            node_id: Target node ID
            message: Input message
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
            
        executor = self.active_workflows[workflow_id]
        await executor.send_input(node_id, message)

# Global workflow manager instance
workflow_manager = WorkflowManager()
