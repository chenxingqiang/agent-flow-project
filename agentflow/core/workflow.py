"""Base workflow module."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
import logging
import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime

from .types import AgentStatus
from ..ell2a.integration import ELL2AIntegration
from ..ell2a.types.message import Message, MessageRole
from .isa.isa_manager import ISAManager
from .instruction_selector import InstructionSelector

if TYPE_CHECKING:
    from ..agents.agent import Agent

logger = logging.getLogger(__name__)

@dataclass
class WorkflowInstance:
    """Workflow instance class."""
    id: str
    agent: 'Agent'
    status: str = "initialized"
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None

class WorkflowEngine:
    """Workflow engine class."""
    
    def __init__(self):
        """Initialize workflow engine."""
        self._initialized = False
        self.workflows: Dict[str, WorkflowInstance] = {}
        self._ell2a: Optional[ELL2AIntegration] = None
        self._isa_manager: Optional[ISAManager] = None
        self._instruction_selector: Optional[InstructionSelector] = None
    
    async def initialize(self) -> None:
        """Initialize workflow engine."""
        if self._initialized:
            return
            
        if not self._ell2a:
            self._ell2a = ELL2AIntegration()
            
        if not self._isa_manager:
            self._isa_manager = ISAManager()
            await self._isa_manager.initialize()
            
        if not self._instruction_selector:
            self._instruction_selector = InstructionSelector()
            await self._instruction_selector.initialize()
            
        self._initialized = True
    
    async def register_workflow(self, agent: 'Agent') -> str:
        """Register a workflow with an agent.
        
        Args:
            agent: Agent to register
            
        Returns:
            Workflow ID
        """
        if not self._initialized:
            await self.initialize()
            
        workflow_id = str(uuid.uuid4())
        self.workflows[workflow_id] = WorkflowInstance(
            id=workflow_id,
            agent=agent
        )
        return workflow_id
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a workflow.
        
        Args:
            workflow_id: Workflow ID
            input_data: Input data for workflow
            
        Returns:
            Execution result
            
        Raises:
            Exception: If workflow execution fails
        """
        if not self._initialized:
            await self.initialize()
            
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        workflow.start_time = datetime.now()
        workflow.status = "running"
        
        try:
            # Create message from input data
            message = input_data.get("message", "")
            if not isinstance(message, str):
                message = str(message)
                
            # Process message with agent
            result = await workflow.agent.process_message(message)
            
            # Create response
            response = {
                "workflow_id": workflow_id,
                "content": result,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
            workflow.status = "completed"
            workflow.end_time = datetime.now()
            workflow.result = response
            return response
            
        except Exception as e:
            error_msg = str(e)
            workflow.status = "failed"
            workflow.error = error_msg
            workflow.end_time = datetime.now()
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Status information
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        return {
            "id": workflow.id,
            "status": workflow.status,
            "error": workflow.error,
            "start_time": workflow.start_time.isoformat() if workflow.start_time else None,
            "end_time": workflow.end_time.isoformat() if workflow.end_time else None,
            "agent_status": workflow.agent.state.status.value
        }
    
    async def cleanup(self) -> None:
        """Clean up workflow engine resources."""
        if not self._initialized:
            return
            
        # Clean up workflows
        for workflow in self.workflows.values():
            await workflow.agent.cleanup()
        self.workflows.clear()
        
        # Clean up components
        if self._ell2a:
            await self._ell2a.cleanup()
            
        if self._isa_manager:
            await self._isa_manager.cleanup()
            
        if self._instruction_selector:
            await self._instruction_selector.cleanup()
            
        self._initialized = False
    
    def __aiter__(self):
        """Return async iterator."""
        return self
        
    async def __anext__(self):
        """Return next value from async iterator."""
        if not self._initialized:
            await self.initialize()
        return self