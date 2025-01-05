"""Workflow engine module."""

from typing import Dict, Any, Optional
import uuid
import logging
import asyncio

from .workflow import Workflow
from ..ell2a.integration import ELL2AIntegration
from ..core.instruction_selector import InstructionSelector
from ..core.isa.isa_manager import ISAManager
from .types import AgentStatus

logger = logging.getLogger(__name__)

class WorkflowEngine:
    """Workflow engine class."""
    
    def __init__(self):
        """Initialize workflow engine."""
        self.workflows: Dict[str, Workflow] = {}
        self._ell2a: Optional[ELL2AIntegration] = None
        self._isa_manager: Optional[ISAManager] = None
        self._instruction_selector: Optional[InstructionSelector] = None
        self._initialized: bool = False
    
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
    
    async def register_workflow(self, agent: Any) -> str:
        """Register a workflow with an agent.
        
        Args:
            agent: Agent to register
            
        Returns:
            Workflow ID
        """
        if not self._initialized:
            await self.initialize()
            
        workflow = Workflow(
            name=f"workflow-{len(self.workflows) + 1}"
        )
        workflow_id = str(uuid.uuid4())
        self.workflows[workflow_id] = workflow
        
        # Initialize agent if needed
        if not agent._initialized:
            await agent.initialize()
            
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow.
        
        Args:
            workflow_id: Workflow ID
            input_data: Input data for workflow
            
        Returns:
            Workflow execution result
            
        Raises:
            ValueError: If workflow not found
            Exception: If workflow execution fails
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
            
        workflow = self.workflows[workflow_id]
        
        try:
            # Execute workflow with input data
            result = await workflow.execute(input_data)
            
            return {
                "workflow_id": workflow_id,
                "result": result,
                "status": workflow.status
            }
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
            workflow.status = "failed"
            workflow.error = str(e)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup workflow engine resources."""
        try:
            # Store components for cleanup
            instruction_selector = self._instruction_selector
            isa_manager = self._isa_manager
            ell2a = self._ell2a
            
            # Cleanup all workflows
            for workflow in self.workflows.values():
                await workflow.cleanup()
                
            self.workflows.clear()
            
            # Cleanup components
            if instruction_selector:
                await instruction_selector.cleanup()
                
            if isa_manager:
                await isa_manager.cleanup()
                
            if ell2a:
                await ell2a.cleanup()
                
            # Clear component references
            self._instruction_selector = None
            self._isa_manager = None
            self._ell2a = None
            self._initialized = False
            
        except Exception as e:
            logger.error(f"Error during workflow engine cleanup: {str(e)}")
            raise 