"""ELL2A-enabled workflow executor module."""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
from datetime import datetime

from .workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType
from .exceptions import WorkflowExecutionError, StepExecutionError
from .ell2a_integration import ell2a_integration
from ..ell2a.lmp import LMPType
from ..ell2a.workflow import ELL2AWorkflow

logger = logging.getLogger(__name__)

class ELL2AWorkflowExecutor:
    """ELL2A-enabled workflow executor."""
    
    def __init__(self, config: Union[Dict[str, Any], WorkflowConfig]):
        """Initialize workflow executor.
        
        Args:
            config: Workflow configuration
        """
        self.config = config if isinstance(config, WorkflowConfig) else WorkflowConfig(**config)
        self.ell2a = ell2a_integration
        self.workflow: Optional[ELL2AWorkflow] = None
        self.metrics: Dict[str, Any] = {
            'tokens': 0,
            'latency': 0.0,
            'memory': 0,
            'steps_completed': 0,
            'steps_failed': 0
        }
        
    async def initialize(self) -> None:
        """Initialize workflow executor."""
        if self.config.use_ell2a:
            self.workflow = ELL2AWorkflow(
                name=self.config.name,
                description=self.config.description
            )
            # Add workflow steps
            for step in self.config.steps:
                self.workflow.add_step({
                    "name": step.get("name", "step"),
                    "type": step.get("type", WorkflowStepType.DEFAULT),
                    "config": step.get("config", {})
                })
            await self.workflow.initialize()
            self.ell2a.register_workflow(self.workflow.id, self.workflow)
            
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow.
        
        Args:
            context: Workflow context
            
        Returns:
            Dict[str, Any]: Workflow results
        """
        try:
            if not self.workflow and self.config.use_ell2a:
                await self.initialize()
                
            results = {}
            for step in self.config.steps:
                step_result = await self._execute_step(step, context)
                results[step.get("name", "step")] = step_result
                context.update(step_result)
                
            return results
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise WorkflowExecutionError(str(e))
        finally:
            if self.workflow:
                self.ell2a.unregister_workflow(self.workflow.id)
                
    @ell2a_integration.track_function()
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow step.
        
        Args:
            step: Step configuration
            context: Workflow context
            
        Returns:
            Dict[str, Any]: Step results
        """
        try:
            # Create step message
            message = self.ell2a.create_message(
                role="user",
                content=str(context),
                metadata={
                    "type": LMPType.TOOL,
                    "step": step.get("name", "step"),
                    "step_type": step.get("type", WorkflowStepType.DEFAULT),
                    "config": step.get("config", {})
                }
            )
            self.ell2a.add_message(message)
            
            # Execute step with appropriate mode
            if self.config.ell2a_mode == "complex":
                result = await self._execute_complex_step(step, message)
            else:
                result = await self._execute_simple_step(step, message)
                
            # Update metrics
            self._update_metrics(result)
            self.metrics['steps_completed'] += 1
            
            return result
        except Exception as e:
            self.metrics['steps_failed'] += 1
            logger.error(f"Step execution failed: {str(e)}")
            raise StepExecutionError(str(e))
            
    @ell2a_integration.track_function()
    async def _execute_simple_step(self, step: Dict[str, Any], message: Any) -> Dict[str, Any]:
        """Execute step using ELL2A simple mode."""
        @self.ell2a.with_ell2a(mode="simple")
        async def execute_step():
            return await self.workflow.run({
                "messages": [message],
                "config": self.ell2a.get_mode_config("simple"),
                "step": step
            })
            
        return await execute_step()
        
    @ell2a_integration.track_function()
    async def _execute_complex_step(self, step: Dict[str, Any], message: Any) -> Dict[str, Any]:
        """Execute step using ELL2A complex mode."""
        @self.ell2a.with_ell2a(mode="complex")
        async def execute_step():
            return await self.workflow.run({
                "messages": self.ell2a.get_messages(),
                "config": self.ell2a.get_mode_config("complex"),
                "step": step
            })
            
        return await execute_step()
        
    def _update_metrics(self, result: Dict[str, Any]) -> None:
        """Update executor metrics."""
        if "usage" in result:
            self.metrics['tokens'] += result["usage"].get("total_tokens", 0)
            self.metrics['latency'] += result["usage"].get("latency", 0.0)
            self.metrics['memory'] += result["usage"].get("memory", 0)
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get executor metrics."""
        return self.metrics.copy()
        
    def reset_metrics(self) -> None:
        """Reset executor metrics."""
        self.metrics = {
            'tokens': 0,
            'latency': 0.0,
            'memory': 0,
            'steps_completed': 0,
            'steps_failed': 0
        } 