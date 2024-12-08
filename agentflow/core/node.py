from enum import Enum
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field

class NodeState(str, Enum):
    """Node state"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

class Node(BaseModel):
    """Base node class"""
    id: str
    name: Optional[str] = None
    type: str
    state: NodeState = NodeState.PENDING
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    output_queue: Optional[Any] = None  # Queue for output data
    metrics: Dict[str, Union[int, float]] = Field(default_factory=lambda: {
        'tokens': 0,
        'latency': 0.0,
        'memory': 0
    })

    def __init__(self, **data):
        super().__init__(**data)
        self._agent = None

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, value):
        self._agent = value

    def reset(self):
        """Reset node state"""
        self.state = NodeState.PENDING
        self.input_data = None
        self.output_data = None
        self.error = None
        self.metrics = {
            'tokens': 0,
            'latency': 0.0,
            'memory': 0
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node with input data"""
        raise NotImplementedError

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

    def __init__(self, **data):
        super().__init__(**data)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with input data"""
        if isinstance(self.agent, str):
            # Import agent class
            module_path, class_name = self.agent.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            agent = agent_class(**self.config)
        else:
            agent = self.agent(**self.config)

        # Execute agent
        result = await agent.execute(input_data)
        return result

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

    def __init__(self, **data):
        super().__init__(**data)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute processor with input data"""
        if isinstance(self.processor, str):
            # Import processor class
            module_path, class_name = self.processor.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            processor_class = getattr(module, class_name)
            processor = processor_class(**self.config)
        else:
            processor = self.processor(**self.config)

        # Execute processor
        if hasattr(processor, 'process_data'):
            result = await processor.process_data(input_data)
        else:
            result = await processor.process(input_data)
        return result
