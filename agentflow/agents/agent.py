"""Agent module."""

from typing import Dict, Any, Optional, Union, List, Type, Callable, TYPE_CHECKING
import uuid
from datetime import datetime
import ray
import logging
import json
import asyncio
from enum import Enum

from pydantic import BaseModel, Field, PrivateAttr, ConfigDict, ValidationError

if TYPE_CHECKING:
    from ..core.workflow import WorkflowEngine
    from ..core.isa.isa_manager import ISAManager
    from ..core.isa.selector import InstructionSelector
    from ..core.ell2a_integration import ELL2AIntegration

from ..core.model_config import ModelConfig
from ..core.isa.isa_manager import Instruction, InstructionType, InstructionStatus
from ..core.isa.selector import InstructionSelector
from ..ell2a.integration import ELL2AIntegration
from ..ell2a.types.message import Message, MessageRole, MessageType
from ..core.types import AgentType, AgentMode, AgentStatus, AgentConfig, ModelConfig
from ..transformations.advanced_strategies import AdvancedTransformationStrategy
from agentflow.errors import WorkflowExecutionError, ConfigurationError
from unittest.mock import AsyncMock
from ..core.workflow_types import WorkflowConfig

logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    """Agent state class."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    status: AgentStatus = Field(default=AgentStatus.INITIALIZED)
    iteration: int = Field(default=0)
    last_error: Optional[str] = None
    messages_processed: int = Field(default=0)

class AgentBase(BaseModel):
    """Base agent class."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        validate_assignment=True,
        from_attributes=True,
        use_enum_values=True,
        populate_by_name=True,
        strict=False
    )
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    description: Optional[str] = None
    type: AgentType = Field(default=AgentType.GENERIC)
    mode: str = Field(default="simple")
    version: str = Field(default="1.0.0")
    system_prompt: Optional[str] = Field(default=None)
    config: Optional['AgentConfig'] = None
    state: AgentState = Field(default_factory=AgentState)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, str]] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(default_factory=list)
    max_errors: int = Field(default=10)
    domain_config: Dict[str, Any] = Field(default_factory=dict)
    is_distributed: bool = Field(default=False)
    
    def __init__(self, **data):
        """Initialize agent with configuration."""
        super().__init__(**data)
        self._initialized = False
        
    async def initialize(self):
        """Initialize agent state and resources."""
        if self._initialized:
            return
        
        self.state = AgentState()
        
        await self._initialize_ell2a()
        await self._initialize_isa()
        await self._initialize_instruction_selector()
        
        self._initialized = True
        
    async def _initialize_ell2a(self):
        """Initialize ELL2A integration."""
        if not self._ell2a:
            # Create ELL2A integration instance
            self._ell2a = ELL2AIntegration()
            
            # Configure with model and domain settings
            if self.config:
                config = {
                    "model": self.config.model if hasattr(self.config, "model") else None,
                    "domain": self.config.domain_config if hasattr(self.config, "domain_config") else {},
                    "enabled": True,
                    "tracking_enabled": True
                }
                self._ell2a.configure(config)
        
    async def _initialize_isa(self):
        """Initialize ISA manager."""
        if not self.isa_manager:
            config_path = None
            if self.config and hasattr(self.config, "domain_config"):
                config_path = self.config.domain_config.get("isa_config_path")
            self.isa_manager = ISAManager(config_path=config_path)
            await self.isa_manager.initialize()
        
    async def _initialize_instruction_selector(self):
        """Initialize instruction selector."""
        if not self.instruction_selector:
            if self.config is None:
                # Import at runtime to avoid circular import
                from ..core.config import AgentConfig
                self.config = AgentConfig(name=self.name or str(uuid.uuid4()))
            self.instruction_selector = InstructionSelector(self.config)
            await self.instruction_selector.initialize()
        
    def add_error(self, error: str):
        """Add error to error list."""
        max_errors = self.max_errors if self.max_errors is not None else 10
        if len(self.errors) >= max_errors:
            self.errors.pop(0)
        self.errors.append({"error": error, "timestamp": str(datetime.now())})

class Agent:
    """Base agent class."""
    
    def __init__(self, config: Optional[Union['AgentConfig', Dict[str, Any]]] = None, name: Optional[str] = None, **kwargs):
        """Initialize agent.
        
        Args:
            config: Agent configuration or dictionary
            name: Agent name (optional, will use config name if not provided)
            **kwargs: Additional parameters to override config values
            
        Raises:
            ValueError: If the configuration is invalid
        """
        # Import at runtime to avoid circular import
        from ..core.config import AgentConfig, ModelConfig, WorkflowConfig

        if config is None:
            config = AgentConfig(name=name or str(uuid.uuid4()))
        elif isinstance(config, dict):
            if not config:  # Empty dictionary
                raise ValueError("Agent must have a configuration")
            try:
                config = AgentConfig(**config)
            except Exception as e:
                raise ValueError(f"Invalid agent configuration: {str(e)}")

        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.name = name or config.name
        self.type = kwargs.get('type', config.type)
        self.mode = kwargs.get('mode', getattr(config, 'mode', 'sequential'))
        self.config = config
        self.metadata: Dict[str, Any] = {}
        self._initialized = False
        self._status: AgentStatus = AgentStatus.IDLE
        
        # Ensure max_errors is always an int
        config_max_errors = getattr(config, 'max_errors', None)
        self.max_errors: int = int(config_max_errors) if isinstance(config_max_errors, (int, float, str)) else 10
        
        self.errors: List[Dict[str, str]] = []
        self.history: List[Dict[str, Any]] = []  # Add history list
        
        # Initialize state
        self.state = AgentState(status=AgentStatus.IDLE)
        
        # Initialize components
        self._ell2a = kwargs.get('ell2a', None)
        self._isa_manager = kwargs.get('isa_manager', None)
        self._instruction_selector = kwargs.get('instruction_selector', None)
        
        # If workflow is provided in kwargs, create a new config with updated workflow
        if 'workflow' in kwargs:
            workflow_data = kwargs['workflow']
            if workflow_data is None:
                # If workflow is explicitly set to None, create an empty workflow
                workflow = WorkflowConfig()
            elif isinstance(workflow_data, dict):
                workflow = WorkflowConfig(**workflow_data)
            elif isinstance(workflow_data, WorkflowConfig):
                workflow = workflow_data
            else:
                raise ValueError("workflow must be a dictionary or WorkflowConfig instance")
            self.config = AgentConfig(
                **{**self.config.model_dump(), "workflow": workflow}
            )
        
        # If model is provided in kwargs, create a new config with updated model
        if 'model' in kwargs:
            model_data = kwargs['model']
            if isinstance(model_data, dict):
                model = ModelConfig(**model_data)
            elif isinstance(model_data, ModelConfig):
                model = model_data
            else:
                raise ValueError("model must be a dictionary or ModelConfig instance")
            self.config = AgentConfig(
                **{**self.config.model_dump(), "model": model}
            )

    @property
    def isa_manager(self):
        """Get ISA manager."""
        return self._isa_manager

    @property
    def instruction_selector(self):
        """Get instruction selector."""
        return self._instruction_selector
        
    @property
    def status(self) -> AgentStatus:
        """Get agent status."""
        return self._status
        
    @status.setter
    def status(self, value: AgentStatus) -> None:
        """Set agent status."""
        self._status = value
        
    async def initialize(self) -> None:
        """Initialize agent resources."""
        if not self._initialized:
            try:
                # Initialize components
                if self.config:
                    # Initialize ELL2A singleton with agent-specific settings
                    from ..core.ell2a_integration import ELL2AIntegration
                    self._ell2a = ELL2AIntegration()
                    ell2a_config = {
                        "model": self.config.model if hasattr(self.config, "model") else None,
                        "domain": self.config.domain_config if hasattr(self.config, "domain_config") else {},
                        "enabled": True,
                        "tracking_enabled": True
                    }
                    self._ell2a.configure(ell2a_config)

                # Initialize ISA manager
                from ..core.isa.isa_manager import ISAManager
                config_path = self.config.domain_config.get("isa_config_path") if self.config else None
                self._isa_manager = ISAManager(config_path=config_path)
                await self._isa_manager.initialize()

                # Initialize instruction selector
                from ..core.isa.selector import InstructionSelector
                self._instruction_selector = InstructionSelector(self.config)
                await self._instruction_selector.initialize()
                
                self._initialized = True
                self._status = AgentStatus.IDLE
            except Exception as e:
                self._status = AgentStatus.FAILED
                raise
    
    def add_error(self, error: str) -> None:
        """Add error to error list."""
        if len(self.errors) >= self.max_errors:
            self.errors.pop(0)
        self.errors.append({"error": error, "timestamp": str(datetime.now())})
    
    async def process_message(self, message: Union[str, Dict[str, Any], Message]) -> str:
        """Process a message.
        
        Args:
            message: Message to process
            
        Returns:
            str: Processing result
        """
        if isinstance(message, str):
            message = Message(content=message, role=MessageRole.USER)
        elif isinstance(message, dict):
            message = Message(**message)
            
        # Add message to history
        self.history.append({
            "role": message.role,
            "content": message.content,
            "timestamp": str(datetime.now())
        })
            
        # Update state
        self.state.messages_processed += 1
        self.state.status = AgentStatus.PROCESSING
        
        try:
            # Process message using ELL2A
            result = await self._ell2a.process_message(message)
            self.state.status = AgentStatus.IDLE
            
            # Add response to history
            response_content = result.content if isinstance(result, Message) else str(result)
            if isinstance(response_content, list):
                response_content = " ".join(str(block.content) for block in response_content)
            self.history.append({
                "role": MessageRole.ASSISTANT,
                "content": response_content,
                "timestamp": str(datetime.now())
            })
            
            # Return the response content
            if isinstance(result, Message):
                content = result.content
                if isinstance(content, list):
                    return " ".join(str(block.content) for block in content)
                return str(content)
            if isinstance(result, (list, tuple)):
                return str(result[0] if result else "")
            return str(result)
        except Exception as e:
            self.state.status = AgentStatus.FAILED
            self.state.last_error = str(e)
            self.add_error(str(e))
            raise
    
    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Execute agent with input data.
        
        Args:
            input_data: Input data for execution
            
        Returns:
            str: Execution result
        """
        if not isinstance(input_data, dict):
            input_data = {"data": input_data}
            
        # Create message from input data
        message = Message(
            content=str(input_data.get("data", "")),
            role=MessageRole.USER,
            metadata=input_data.get("metadata", {})
        )
        
        return await self.process_message(message)
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        try:
            if self._ell2a:
                if hasattr(self._ell2a, 'cleanup'):
                    cleanup_method = getattr(self._ell2a, 'cleanup')
                    if asyncio.iscoroutinefunction(cleanup_method):
                        await cleanup_method()
                    else:
                        cleanup_method()
                        
            if self._isa_manager:
                if hasattr(self._isa_manager, 'cleanup'):
                    cleanup_method = getattr(self._isa_manager, 'cleanup')
                    if asyncio.iscoroutinefunction(cleanup_method):
                        await cleanup_method()
                    else:
                        cleanup_method()
                        
            if self._instruction_selector:
                # Only try to cleanup if the method exists
                if hasattr(self._instruction_selector, 'cleanup'):
                    cleanup_method = getattr(self._instruction_selector, 'cleanup')
                    if asyncio.iscoroutinefunction(cleanup_method):
                        await cleanup_method()
                    else:
                        cleanup_method()
                        
            self._initialized = False
            self._status = AgentStatus.INITIALIZED
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise

@ray.remote
class RemoteAgent:
    """Remote agent class for distributed operations."""
    def __init__(self, config=None):
        """Initialize remote agent."""
        self.config = config
        self.id = str(uuid.uuid4())
        self.name = 'remote_agent'
        self.type = AgentType.GENERIC
        self.version = '1.0.0'

    async def initialize(self):
        """Initialize remote agent."""
        pass

    async def cleanup(self):
        """Clean up remote agent resources."""
        pass
