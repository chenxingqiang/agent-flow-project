"""Agent module."""

from typing import Dict, Any, Optional, Union, List, Type, Callable
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
import uuid
import logging

from ..core.config import AgentConfig, ModelConfig, WorkflowConfig
from ..core.isa.isa_manager import Instruction, ISAManager
from ..core.isa.types import InstructionType
from ..core.instruction_selector import InstructionSelector
from ..ell2a.integration import ELL2AIntegration
from ..ell2a.types.message import Message, MessageRole
from .agent_types import AgentType, AgentStatus
from ..transformations.advanced_strategies import AdvancedTransformationStrategy
from ..core.isa.instruction import Instruction

logger = logging.getLogger(__name__)

class TransformationPipeline:
    """Pipeline for applying multiple transformation strategies in sequence."""
    
    def __init__(self):
        """Initialize transformation pipeline."""
        self.strategies = []
        
    def add_strategy(self, strategy: AdvancedTransformationStrategy) -> None:
        """Add a transformation strategy to the pipeline.
        
        Args:
            strategy: Strategy to add
        """
        self.strategies.append(strategy)
        
    def fit_transform(self, data: Any) -> Any:
        """Apply all transformation strategies in sequence.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        result = data
        for strategy in self.strategies:
            result = strategy.transform(result)
        return result

class AgentState(BaseModel):
    """Agent state class."""
    
    status: AgentStatus = Field(default=AgentStatus.IDLE)
    iteration: int = Field(default=0)
    last_error: Optional[str] = None
    messages_processed: int = Field(default=0)

class Agent(BaseModel):
    """Agent class."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    description: Optional[str] = None
    type: AgentType = Field(default=AgentType.GENERIC)
    mode: str = Field(default="simple")
    version: str = Field(default="1.0.0")
    config: Optional[AgentConfig] = None
    state: AgentState = Field(default_factory=AgentState)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, str]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    max_errors: int = Field(default=10)
    domain_config: Dict[str, Any] = Field(default_factory=dict)
    is_distributed: bool = Field(default=False)
    process_message: Optional[Callable[..., Any]] = Field(default=None)
    
    # Private attributes
    _ell2a: Optional[ELL2AIntegration] = PrivateAttr(default=None)
    _isa_manager: Optional[ISAManager] = PrivateAttr(default=None)
    _instruction_selector: Optional[InstructionSelector] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    
    def __init__(self, config: Union[Dict[str, Any], str, AgentConfig], **data):
        """Initialize agent."""
        if isinstance(config, str):
            # Load config from file
            import json
            from pathlib import Path
    
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config}")
    
            with open(config_path) as f:
                config = json.load(f)
    
            # Load workflow from separate file if specified
            if "AGENT" in config and "workflow_path" in config["AGENT"]:
                workflow_path = Path(config["AGENT"]["workflow_path"])
                if not workflow_path.exists():
                    raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
                    
                with open(workflow_path) as f:
                    workflow = json.load(f)
                    config["AGENT"]["workflow"] = workflow
                    
        # Convert config to AgentConfig if needed
        if isinstance(config, dict):
            config = AgentConfig(**config)
        elif not isinstance(config, AgentConfig):
            raise TypeError(f"Expected AgentConfig, got {type(config)}")
            
        # Initialize base class
        super().__init__(config=config, **data)
        
        # Initialize ISA manager
        self._isa_manager = ISAManager()
        self._instruction_selector = InstructionSelector()
        
    def add_error(self, error_msg: str):
        """Add error to the list, maintaining max size."""
        self.errors.append(error_msg)
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
        self.state.last_error = error_msg
        self.state.status = AgentStatus.FAILED
    
    @property
    def client(self) -> Any:
        """Get OpenAI client."""
        return self._ell2a
        
    @client.setter
    def client(self, value: Any) -> None:
        """Set OpenAI client."""
        self._ell2a = value
        
    @property
    def isa_manager(self):
        """Expose ISA manager as a property."""
        return self._isa_manager
    
    @property
    def instruction_selector(self):
        """Expose instruction selector as a property."""
        return self._instruction_selector
    
    async def initialize(self) -> None:
        """Initialize agent."""
        if self._initialized:
            return
        
        # Allow a default system prompt if not specified
        if not self._ell2a:
            system_prompt = self.config.system_prompt or "You are a helpful AI assistant."
            self._ell2a = ELL2AIntegration()
            self._ell2a.configure({
                "system_prompt": system_prompt
            })
        
        # Initialize ISA manager with default instructions
        if not self._isa_manager:
            self._isa_manager = ISAManager()
        
        # Ensure default instructions are added
        default_instructions = [
            Instruction(
                id="init",
                name="initialize",
                type="control",
                params={"init_param": "value"},
                description="Initialize system and prepare environment"
            ),
            Instruction(
                id="process",
                name="process_data",
                type="computation",
                params={"data_param": "value"},
                description="Process and analyze input data efficiently"
            ),
            Instruction(
                id="validate",
                name="validate_result",
                type="validation",
                params={"threshold": 0.9},
                description="Validate and verify the results for accuracy"
            )
        ]
        
        # Add instructions if not already present
        for instruction in default_instructions:
            try:
                self._isa_manager.get_instruction(instruction.id)
            except ValueError:
                self._isa_manager.register_instruction(instruction)
        
        # Initialize ISA manager
        await self._isa_manager.initialize()
            
        if not self._instruction_selector:
            self._instruction_selector = InstructionSelector()
            
            # Train the instruction selector with default instructions
            self._instruction_selector.train(self._isa_manager.instructions)
            await self._instruction_selector.initialize()
        
        # Set the process_message method
        self.process_message = self._process_message
        
        self._initialized = True
        self.state.status = AgentStatus.IDLE
        
    async def _process_message(self, input_data: Union[Dict[str, Any], str, Message]) -> Any:
        """Process an input message through the agent's workflow.
        
        Args:
            input_data: Input data to process, can be a dictionary, string, or Message object
        
        Returns:
            Processed result from the agent's workflow
        
        Raises:
            ValueError: If input type is unsupported
            Exception: If message processing fails
        """
        # Ensure agent is initialized
        if not self._initialized:
            await self.initialize()
        
        # Convert input to Message if needed
        if isinstance(input_data, str):
            message = Message(role=MessageRole.USER, content=input_data)
        elif isinstance(input_data, dict):
            message = Message(role=MessageRole.USER, content=input_data.get('message', ''))
        elif isinstance(input_data, Message):
            message = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Update agent state
        self.state.iteration += 1
        self.state.messages_processed += 1
        self.state.status = AgentStatus.PROCESSING
        
        try:
            # Use ELL2A integration to process the message
            if not self._ell2a:
                raise ValueError("ELL2A integration not configured")
            
            # Process message through ELL2A
            response = await self._ell2a.process_message(message)
            
            # Update history
            self.history.append({
                'input': message.content,
                'output': response.content,
                'timestamp': str(response.timestamp)
            })
            
            # Update agent state
            self.state.status = AgentStatus.IDLE
            
            return response.content
        
        except Exception as e:
            # Handle and log errors
            error_msg = str(e)
            self.add_error(error_msg)
            
            # Ensure status is set to FAILED
            self.state.status = AgentStatus.FAILED
            
            # Re-raise the exception to allow test error handling
            raise

    async def process_message(self, input_data: Union[Dict[str, Any], str, Message]) -> Any:
        """Proxy method for process_message."""
        if callable(self._process_message):
            return await self._process_message(input_data)
        raise NotImplementedError("process_message not configured")

    async def cleanup(self) -> None:
        """Clean up agent resources."""
        if not self._initialized:
            return
        
        # Reset state
        self.state.status = AgentStatus.STOPPED
        self.state.iteration = 0
        self.state.messages_processed = 0
        
        # Cleanup ELL2A integration
        if self._ell2a and hasattr(self._ell2a, 'cleanup'):
            await self._ell2a.cleanup()
        
        # Cleanup ISA manager
        if self._isa_manager and hasattr(self._isa_manager, 'cleanup'):
            await self._isa_manager.cleanup()
        
        # Cleanup instruction selector
        if self._instruction_selector and hasattr(self._instruction_selector, 'cleanup'):
            await self._instruction_selector.cleanup()
        
        # Reset initialization flag
        self._initialized = False
        
    def __aiter__(self):
        """Return async iterator."""
        return self
        
    async def __anext__(self):
        """Return next value from async iterator."""
        if not self._initialized:
            await self.initialize()
        return self
