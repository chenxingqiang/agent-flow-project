from enum import Enum
from typing import Any, Dict, Optional, Union, List
from pydantic import BaseModel, Field, ConfigDict
import asyncio
import logging
from abc import ABC, abstractmethod
from ..ell2a.integration import ELL2AIntegration

logger = logging.getLogger(__name__)

# Initialize ELL2A integration
ell2a_integration = ELL2AIntegration()

class NodeType(str, Enum):
    """Node type"""
    AGENT = "agent"
    PROCESSOR = "processor"
    WORKFLOW = "workflow"

class NodeState(str, Enum):
    """Node state"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

class Node(ABC):
    """Base class for all nodes in the workflow."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.metadata: Dict[str, Any] = {}
        self.status: str = "initialized"
    
    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input context and return output."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the node."""
        self.status = "ready"
    
    async def cleanup(self) -> None:
        """Clean up node resources."""
        self.status = "cleaned"
    
    def get_status(self) -> str:
        """Get the current node status."""
        return self.status
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the node."""
        self.metadata[key] = value
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get node metadata."""
        return self.metadata.copy()
    
    def validate(self) -> bool:
        """Validate the node configuration."""
        return True

class AgentNode(Node):
    """Node that executes an agent"""
    type: str = "agent"
    model: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Union[int, float]] = Field(default_factory=lambda: {
        'tokens': 0,
        'latency': 0.0,
        'memory': 0
    })
    status: Optional[Dict[str, Any]] = None  # For storing status information
    use_ell2a: bool = Field(default=False)  # Flag to use ELL2A
    ell2a_mode: str = Field(default="simple")  # ELL2A mode: "simple" or "complex"

    def __init__(self, **data):
        super().__init__(**data)
        if self.use_ell2a:
            from .ell2a_integration import ell2a_integration
            self.ell2a = ell2a_integration

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with input data"""
        if self.use_ell2a:
            return await self._execute_with_ell2a(input_data)
        return await self._execute_standard(input_data)

    async def _execute_with_ell2a(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent using ELL2A"""
        # Create ELL2A message
        message = self.ell2a.create_message(
            role="user",
            content=input_data.get("content", ""),
            metadata={
                "type": LMPType.AGENT,
                "config": self.config
            }
        )
        self.ell2a.add_message(message)

        # Execute with ELL2A based on mode
        if self.ell2a_mode == "complex":
            result = await self._execute_complex_ell2a(message)
        else:
            result = await self._execute_simple_ell2a(message)
        
        # Update metrics
        if "usage" in result:
            self.metrics.update({
                "tokens": result["usage"].get("total_tokens", 0),
                "latency": result["usage"].get("latency", 0.0)
            })

        return result

    @ell2a_integration.track_function()
    async def _execute_simple_ell2a(self, message: Any) -> Dict[str, Any]:
        """Execute using ELL2A simple mode"""
        try:
            # Get agent implementation
            agent = self._get_agent_instance()

            # Execute with global ELL2A simple mode
            @self.ell2a.with_ell2a(mode="simple")
            async def execute_agent():
                return await agent.execute({
                    "messages": [message],
                    "config": self.config
                })

            return await execute_agent()
        except Exception as e:
            self.logger.error(f"Error executing agent with ELL2A simple: {str(e)}")
            raise

    @ell2a_integration.track_function()
    async def _execute_complex_ell2a(self, message: Any) -> Dict[str, Any]:
        """Execute using ELL2A complex mode"""
        try:
            # Get agent implementation
            agent = self._get_agent_instance()

            # Execute with global ELL2A complex mode
            @self.ell2a.with_ell2a(mode="complex")
            async def execute_agent():
                return await agent.execute({
                    "messages": self.ell2a.get_messages(),
                    "config": self.config
                })

            return await execute_agent()
        except Exception as e:
            self.logger.error(f"Error executing agent with ELL2A complex: {str(e)}")
            raise

    def _get_agent_instance(self):
        """Get agent instance"""
        if isinstance(self.agent, str):
            module_path, class_name = self.agent.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            return agent_class(**self.config)
        return self.agent(**self.config)

    async def _execute_standard(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent using standard method"""
        agent = self._get_agent_instance()
        return await agent.execute(input_data)

class ProcessorNode(Node):
    """Node that executes a processor"""
    type: str = "processor"
    processor: Any  # Can be string (import path) or class
    config: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Union[int, float]] = Field(default_factory=lambda: {
        'tokens': 0,
        'latency': 0.0,
        'memory': 0
    })
    use_ell2a: bool = Field(default=False)  # Flag to use ELL2A
    ell2a_mode: str = Field(default="simple")  # ELL2A mode: "simple" or "complex"

    def __init__(self, **data):
        super().__init__(**data)
        if self.use_ell2a:
            from .ell2a_integration import ell2a_integration
            self.ell2a = ell2a_integration

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute processor with input data"""
        if self.use_ell2a:
            return await self._execute_with_ell2a(input_data)
        return await self._execute_standard(input_data)

    async def _execute_with_ell2a(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute processor using ELL2A"""
        # Create ELL2A message
        message = self.ell2a.create_message(
            role="user",
            content=str(input_data),
            metadata={
                "type": LMPType.TOOL,
                "config": self.config,
                "processor": self.processor.__name__ if hasattr(self.processor, "__name__") else "processor"
            }
        )
        self.ell2a.add_message(message)

        # Execute with ELL2A based on mode
        if self.ell2a_mode == "complex":
            result = await self._execute_complex_ell2a(message)
        else:
            result = await self._execute_simple_ell2a(message)
        
        # Update metrics
        if "usage" in result:
            self.metrics.update({
                "tokens": result["usage"].get("total_tokens", 0),
                "latency": result["usage"].get("latency", 0.0)
            })

        return result

    @ell2a_integration.track_function()
    async def _execute_simple_ell2a(self, message: Any) -> Dict[str, Any]:
        """Execute using ELL2A simple mode"""
        try:
            # Get processor implementation
            processor = self._get_processor_instance()

            # Execute with global ELL2A simple mode
            @self.ell2a.with_ell2a(mode="simple")
            async def execute_processor():
                if hasattr(processor, 'process_data'):
                    return await processor.process_data({
                        "messages": [message],
                        "config": self.config
                    })
                return await processor.process({
                    "messages": [message],
                    "config": self.config
                })

            return await execute_processor()
        except Exception as e:
            self.logger.error(f"Error executing processor with ELL2A simple: {str(e)}")
            raise

    @ell2a_integration.track_function()
    async def _execute_complex_ell2a(self, message: Any) -> Dict[str, Any]:
        """Execute using ELL2A complex mode"""
        try:
            # Get processor implementation
            processor = self._get_processor_instance()

            # Execute with global ELL2A complex mode
            @self.ell2a.with_ell2a(mode="complex")
            async def execute_processor():
                if hasattr(processor, 'process_data'):
                    return await processor.process_data({
                        "messages": self.ell2a.get_messages(),
                        "config": self.config
                    })
                return await processor.process({
                    "messages": self.ell2a.get_messages(),
                    "config": self.config
                })

            return await execute_processor()
        except Exception as e:
            self.logger.error(f"Error executing processor with ELL2A complex: {str(e)}")
            raise

    def _get_processor_instance(self):
        """Get processor instance"""
        if isinstance(self.processor, str):
            module_path, class_name = self.processor.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            processor_class = getattr(module, class_name)
            return processor_class(**self.config)
        return self.processor(**self.config)

    async def _execute_standard(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute processor using standard method"""
        processor = self._get_processor_instance()
        if hasattr(processor, 'process_data'):
            return await processor.process_data(input_data)
        return await processor.process(input_data)

class WorkflowNode(BaseModel):
    """Workflow node class."""
    model_config = ConfigDict(frozen=False, validate_assignment=True)
    
    id: str
    name: Optional[str] = None
    node_type: NodeType = Field(default=NodeType.WORKFLOW)
    state: NodeState = Field(default_factory=lambda: NodeState.PENDING)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    agent_instance: Optional[Any] = None
    processor: Optional[Any] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow node with input data"""
        try:
            self.state = NodeState.RUNNING
            if self.node_type == NodeType.AGENT and self.agent_instance:
                # Add delay if specified in config
                if self.config.get('delay'):
                    await asyncio.sleep(self.config['delay'])
                result = await self.agent_instance.execute(input_data)
                self.state = NodeState.COMPLETED
                return result
            self.state = NodeState.COMPLETED
            return {}
        except Exception as e:
            self.state = NodeState.ERROR
            raise e
