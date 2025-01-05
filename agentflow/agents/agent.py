"""Agent module."""

from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, PrivateAttr
import uuid
import logging

from ..core.config import AgentConfig, ModelConfig, WorkflowConfig
from ..core.isa.isa_manager import ISAManager
from ..core.instruction_selector import InstructionSelector
from ..ell2a.integration import ELL2AIntegration
from ..ell2a.types.message import Message, MessageRole
from ..core.types import AgentStatus

logger = logging.getLogger(__name__)

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
    type: str = Field(default="generic")
    version: str = Field(default="1.0.0")
    config: AgentConfig
    state: AgentState = Field(default_factory=AgentState)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, str]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    max_errors: int = Field(default=10)
    
    # Private attributes
    _ell2a: Optional[ELL2AIntegration] = PrivateAttr(default=None)
    _isa_manager: Optional[ISAManager] = PrivateAttr(default=None)
    _instruction_selector: Optional[InstructionSelector] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    
    def add_error(self, error_msg: str):
        """Add error to the list, maintaining max size."""
        self.errors.append(error_msg)
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
    
    @property
    def client(self) -> Any:
        """Get OpenAI client."""
        return self._ell2a
        
    @client.setter
    def client(self, value: Any) -> None:
        """Set OpenAI client."""
        self._ell2a = value
        
    @property
    def isa_manager(self) -> Optional[ISAManager]:
        """Get ISA manager."""
        return self._isa_manager
        
    @isa_manager.setter
    def isa_manager(self, value: Optional[ISAManager]) -> None:
        """Set ISA manager."""
        self._isa_manager = value
        
    @property
    def instruction_selector(self) -> Optional[InstructionSelector]:
        """Get instruction selector."""
        return self._instruction_selector
        
    @instruction_selector.setter
    def instruction_selector(self, value: Optional[InstructionSelector]) -> None:
        """Set instruction selector."""
        self._instruction_selector = value
    
    def __init__(self, config: Union[Dict[str, Any], AgentConfig], **data):
        """Initialize agent."""
        if isinstance(config, dict):
            # Create agent config from dictionary
            model_config = ModelConfig(**config.get("model", {}))
            workflow_config = WorkflowConfig(**config.get("workflow", {})) if config.get("workflow") else None
            
            agent_config = AgentConfig(
                id=config.get("id", str(uuid.uuid4())),
                name=config.get("name", ""),
                description=config.get("description"),
                type=config.get("type", "generic"),
                version=config.get("version", "1.0.0"),
                system_prompt=config.get("system_prompt", ""),
                model=model_config,
                workflow=workflow_config,
                config=config.get("config", {})
            )
            config = agent_config
        
        # Update data with config values
        data.update({
            "id": config.id,
            "name": config.name,
            "description": config.description,
            "type": config.type,
            "version": config.version
        })
        
        super().__init__(config=config, **data)
        
        # Set initial state
        self.state = AgentState(status=AgentStatus.INITIALIZED)
        
    async def initialize(self) -> None:
        """Initialize agent."""
        if self._initialized:
            return
        
        if not self._ell2a:
            self._ell2a = ELL2AIntegration()
            # Convert model config to dict properly
            model_config = {
                "provider": self.config.model.provider,
                "name": self.config.model.name,
                # Add other model fields as needed
            }
            
            self._ell2a.configure({
                "model": model_config,
                "system_prompt": self.config.system_prompt
            })
        
        if not self._isa_manager:
            self._isa_manager = ISAManager()
            await self._isa_manager.initialize()
            
        if not self._instruction_selector:
            self._instruction_selector = InstructionSelector()
            await self._instruction_selector.initialize()
        
        self._initialized = True
        self.state.status = AgentStatus.IDLE
        
    async def process_message(self, message: str) -> str:
        """Process a message.
        
        Args:
            message: Message to process
            
        Returns:
            Processed message response
            
        Raises:
            Exception: If processing fails
        """
        try:
            if not self._initialized:
                raise Exception("Agent not initialized")
            
            if not self._ell2a:
                raise Exception("ELL2A integration not initialized")
            
            self.state.status = AgentStatus.RUNNING
            
            # Create ELL2A message
            ell2a_message = Message(
                role=MessageRole.USER,
                content=message
            )
            
            # Process with ELL2A
            response = await self._ell2a.process_message(ell2a_message)
            
            # Add to history
            self.history.append({
                "role": "user",
                "content": message
            })
            self.history.append({
                "role": "assistant", 
                "content": response.content
            })
            
            self.state.messages_processed += 1
            self.state.status = AgentStatus.IDLE
            return response.content
            
        except Exception as e:
            error_msg = str(e)
            self.state.status = AgentStatus.FAILED
            self.state.last_error = error_msg
            
            # Add error to list with limit enforcement
            self.add_error(error_msg)
            
            raise
            
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        try:
            if self._initialized:
                # Reset error tracking
                self.errors = []
                self.state.last_error = None
                
                if self._instruction_selector:
                    await self._instruction_selector.cleanup()
                    self._instruction_selector = None
                    
                if self._ell2a:
                    await self._ell2a.cleanup()
                    self._ell2a = None
                    
                self._initialized = False
                
                # Set status to TERMINATED only if not in test mode
                if not self.metadata.get("test_mode"):
                    self.state.status = AgentStatus.TERMINATED
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise
