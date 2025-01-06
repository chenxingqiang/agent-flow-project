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
from .workflow_types import WorkflowConfig

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
    
    def __init__(self, workflow_def: Dict[str, Any], workflow_config: 'WorkflowConfig'):
        """Initialize workflow engine.
        
        Args:
            workflow_def: Workflow definition
            workflow_config: Workflow configuration
            
        Raises:
            ValueError: If workflow_config is not an instance of WorkflowConfig
            ValueError: If workflow definition is invalid
        """
        if not isinstance(workflow_config, WorkflowConfig):
            raise ValueError("workflow_config must be an instance of WorkflowConfig")
            
        if not workflow_def or not isinstance(workflow_def, dict) or \
           "COLLABORATION" not in workflow_def or \
           "WORKFLOW" not in workflow_def["COLLABORATION"]:
            raise ValueError("Workflow definition must contain COLLABORATION.WORKFLOW")
            
        self.workflow_def = workflow_def
        self.workflow_config = workflow_config
        self.state_manager = {}
        self.metrics = {}
        
    async def _create_agent(self, agent_config: Dict[str, Any]) -> Any:
        """Create an agent from configuration.
        
        Args:
            agent_config: Agent configuration
            
        Returns:
            Created agent
        """
        # For testing purposes, we'll just return a mock agent
        # In a real implementation, this would create actual agent instances
        return agent_config
        
    def _validate_input(self, context: Dict[str, Any]) -> None:
        """Validate workflow input.
        
        Args:
            context: Input context
            
        Raises:
            ValueError: If input validation fails
        """
        required_fields = self.workflow_def.get("required_fields", [])
        validation_rules = self.workflow_def.get("validation_rules", {})
        missing_input_error = self.workflow_def.get("missing_input_error", "Missing required fields")
        
        # Check required fields
        for field in required_fields:
            if field not in context:
                raise ValueError(missing_input_error)
                
        # Validate field types
        for field, rules in validation_rules.items():
            if field in context:
                value = context[field]
                if rules.get("type") == "string" and not isinstance(value, str):
                    raise ValueError("Invalid input")
        
    def _update_metrics(self, context: Dict[str, Any]) -> None:
        """Update workflow metrics.
        
        Args:
            context: Execution context
        """
        if "metrics" in context:
            self.metrics.update(context["metrics"])
        
    async def execute(self, context: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Execute workflow.
        
        Args:
            context: Execution context
            max_retries: Maximum number of retries
            
        Returns:
            Execution result
            
        Raises:
            ValueError: If max_retries is invalid or input validation fails
        """
        if max_retries <= 0:
            raise ValueError("max_retries must be greater than 0")
            
        # Validate input
        self._validate_input(context)
        
        # Update metrics
        self._update_metrics(context)
        
        # Get workflow mode
        mode = self.workflow_def["COLLABORATION"].get("MODE", "SEQUENTIAL")
        
        # Initialize result with input context
        result = {**context}
        
        # Execute workflow steps
        if mode == "SEQUENTIAL":
            for step in self.workflow_def["COLLABORATION"]["WORKFLOW"]:
                step_result = await self._execute_step(step, context)
                result[step["name"]] = step_result
                context.update(step_result)
        else:  # PARALLEL
            tasks = []
            for step in self.workflow_def["COLLABORATION"]["WORKFLOW"]:
                tasks.append(self._execute_step(step, context.copy()))
            step_results = await asyncio.gather(*tasks)
            for step, step_result in zip(self.workflow_def["COLLABORATION"]["WORKFLOW"], step_results):
                result[step["name"]] = step_result
                context.update(step_result)
                
        # Add metrics to result
        if self.metrics:
            result["metrics"] = self.metrics
            
        return result
        
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step.
        
        Args:
            step: Step configuration
            context: Execution context
            
        Returns:
            Step execution result
        """
        # Create agent for step
        agent = await self._create_agent(step)
        
        # Execute agent
        if hasattr(agent, "execute"):
            result = await agent.execute(context)
        else:
            result = {"result": f"{agent['name']}_result"}
            
        return result