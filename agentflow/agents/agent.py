"""Agent module."""

from typing import Dict, Any, Optional, Union, List, Type
from pydantic import BaseModel, Field, PrivateAttr
import uuid
import logging

from ..core.config import AgentConfig, ModelConfig, WorkflowConfig
from ..core.isa.isa_manager import ISAManager
from ..core.instruction_selector import InstructionSelector
from ..ell2a.integration import ELL2AIntegration
from ..ell2a.types.message import Message, MessageRole
from .agent_types import AgentType, AgentStatus
from ..transformations.advanced_strategies import AdvancedTransformationStrategy

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
    type: str = Field(default="generic")
    mode: str = Field(default="simple")
    version: str = Field(default="1.0.0")
    config: AgentConfig
    state: AgentState = Field(default_factory=AgentState)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, str]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    max_errors: int = Field(default=10)
    domain_config: Dict[str, Any] = Field(default_factory=dict)
    is_distributed: bool = Field(default=False)
    
    # Private attributes
    _ell2a: Optional[ELL2AIntegration] = PrivateAttr(default=None)
    _isa_manager: Optional[ISAManager] = PrivateAttr(default=None)
    _instruction_selector: Optional[InstructionSelector] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields
    
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
    
    @property
    def model(self) -> Optional[ModelConfig]:
        """Get model configuration."""
        return self.config.model if self.config else None
        
    @property
    def workflow(self) -> Optional[WorkflowConfig]:
        """Get workflow configuration."""
        return self.config.workflow if self.config else None
    
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
                    workflow_data = json.load(f)
                    config["WORKFLOW"] = workflow_data
        
        if isinstance(config, dict):
            # Get agent-specific config
            agent_data = config.get("AGENT", {})
            
            # Create agent config from dictionary
            model_data = config.get("MODEL", {})
            if model_data.get("provider") not in ["openai", "anthropic"]:
                raise ValueError(f"Invalid model provider: {model_data.get('provider')}")
            model_config = ModelConfig(**model_data)
            
            # Create workflow config
            workflow_data = config.get("WORKFLOW", {})
            if workflow_data:
                if workflow_data.get("max_iterations", 1) <= 0:
                    raise ValueError("max_iterations must be greater than 0")
                workflow_data["id"] = workflow_data.get("id", str(uuid.uuid4()))
                workflow_config = WorkflowConfig(**workflow_data)
            else:
                workflow_config = WorkflowConfig()
            
            # Get domain config
            domain_config = config.get("DOMAIN_CONFIG", {})
            
            agent_config = AgentConfig(
                id=agent_data.get("id", str(uuid.uuid4())),
                name=agent_data.get("name", ""),
                description=agent_data.get("description"),
                type=agent_data.get("type", "generic"),
                version=agent_data.get("version", "1.0.0"),
                model=model_config,
                workflow=workflow_config
            )
            config = agent_config
            
            # Get mode from agent data
            mode = agent_data.get("mode", "simple")
            
            # Get agent-specific attributes
            agent_type = agent_data.get("type", "generic")
            if agent_type == AgentType.DATA_SCIENCE.value:
                metrics = domain_config.get("metrics", [])
            elif agent_type == AgentType.RESEARCH.value:
                research_domains = domain_config.get("research_domains", [])
                
            # Set distributed flag
            is_distributed = workflow_data.get("distributed", False)
        else:
            mode = getattr(config, "mode", "simple")
            agent_type = config.type
            if agent_type == AgentType.DATA_SCIENCE.value:
                metrics = config.config.get("metrics", [])
            elif agent_type == AgentType.RESEARCH.value:
                research_domains = config.config.get("research_domains", [])
            is_distributed = getattr(config.workflow, "distributed", False)
        
        # Update data with config values
        data.update({
            "id": config.id,
            "name": config.name,
            "description": config.description,
            "type": agent_type,
            "mode": mode,
            "version": config.version,
            "domain_config": getattr(config, "config", {}),
            "is_distributed": is_distributed
        })
        
        # Add agent-specific attributes
        if agent_type == AgentType.DATA_SCIENCE.value:
            data["metrics"] = metrics
        elif agent_type == AgentType.RESEARCH.value:
            data["research_domains"] = research_domains
        
        super().__init__(config=config, **data)
        
        # Set initial state
        self.state = AgentState(status=AgentStatus.IDLE)
        
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
            
            # Add message to history
            self.history.append({
                "role": "user",
                "content": message
            })
            
            # Process with ELL2A
            response = await self._ell2a.process_message(ell2a_message)
            
            # Add response to history
            self.history.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Update state
            self.state.messages_processed += 1
            self.state.status = AgentStatus.IDLE
            
            return response.content
            
        except Exception as e:
            error_msg = str(e)
            self.add_error(error_msg)
            raise
            
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self._ell2a:
            await self._ell2a.cleanup()
        
        if self._isa_manager:
            await self._isa_manager.cleanup()
            
        if self._instruction_selector:
            await self._instruction_selector.cleanup()
            
        self._initialized = False
        self.state.status = AgentStatus.TERMINATED
        
    def __aiter__(self):
        """Return async iterator."""
        return self
        
    async def __anext__(self):
        """Return next value from async iterator."""
        if not self._initialized:
            await self.initialize()
        return self
