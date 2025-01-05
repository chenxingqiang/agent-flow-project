"""Workflow executor module."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from .workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig, WorkflowStatus, StepStatus
from .workflow_state import WorkflowStateManager
from .metrics import MetricsManager, MetricType
from .exceptions import WorkflowExecutionError
from .processors.transformers import TransformProcessor, ProcessorResult

logger = logging.getLogger(__name__)

class WorkflowExecutor:
    """Workflow executor class."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize workflow executor."""
        self.config = config
        self.state_manager = WorkflowStateManager()
        self.metrics = MetricsManager()
        self.status = "initialized"
        self.start_time = None
        self.end_time = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow."""
        workflow_id = self.config.id
        try:
            # Initialize workflow state
            self.state_manager.initialize_workflow(workflow_id)
            self.status = "running"
            self.start_time = datetime.now()

            # Record start metrics
            self.metrics.record_metric(
                MetricType.LATENCY,
                0.0,
                {"workflow_id": workflow_id}
            )

            # Execute workflow steps
            result = await self._execute_steps(context)

            # Record completion metrics
            execution_time = (datetime.now() - self.start_time).total_seconds()
            self.metrics.record_metric(
                MetricType.LATENCY,
                execution_time,
                {"workflow_id": workflow_id}
            )

            # Update workflow state
            self.state_manager.update_workflow_status(workflow_id, WorkflowStatus.COMPLETED)
            self.status = "completed"
            self.end_time = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            self.state_manager.update_workflow_status(workflow_id, WorkflowStatus.FAILED)
            self.status = "failed"
            self.end_time = datetime.now()
            raise WorkflowExecutionError(str(e), workflow_id)

    async def _execute_steps(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps."""
        workflow_id = self.config.id
        result = {}
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            raise WorkflowExecutionError("Circular dependencies detected in workflow", workflow_id)
        
        # Check for missing dependencies
        if self._has_missing_dependencies():
            raise WorkflowExecutionError("Missing dependencies detected in workflow", workflow_id)
        
        # Execute steps in order
        for step in self.config.steps:
            try:
                # Check if dependencies are satisfied
                if not self._are_dependencies_satisfied(step, result):
                    raise WorkflowExecutionError(f"Dependencies not satisfied for step {step.id}", workflow_id)
                
                # Check timeout before executing step
                if self._is_timeout():
                    raise WorkflowExecutionError("Workflow execution timeout", workflow_id)
                
                # Update step status
                self.state_manager.update_step_status(workflow_id, step.id, StepStatus.RUNNING)
                step.status = StepStatus.RUNNING
                step.start_time = datetime.now()

                # Create step context with results from previous steps
                step_context = {**context, **result}

                # Execute step with timeout
                try:
                    step_result = await asyncio.wait_for(
                        self._execute_step(step, step_context),
                        timeout=self.config.timeout
                    )
                    result[step.id] = step_result

                    # Update step status
                    self.state_manager.update_step_status(workflow_id, step.id, StepStatus.COMPLETED)
                    step.status = StepStatus.COMPLETED
                    step.end_time = datetime.now()
                except asyncio.TimeoutError:
                    error_msg = f"Step {step.id} timed out after {self.config.timeout} seconds"
                    logger.error(error_msg)
                    self.state_manager.update_step_status(workflow_id, step.id, StepStatus.FAILED)
                    step.status = StepStatus.FAILED
                    step.error = error_msg
                    raise WorkflowExecutionError(error_msg, workflow_id)

            except Exception as e:
                error_msg = f"Step {step.id} failed: {str(e)}"
                logger.error(error_msg)
                self.state_manager.update_step_status(workflow_id, step.id, StepStatus.FAILED)
                step.status = StepStatus.FAILED
                step.error = str(e)
                raise WorkflowExecutionError(error_msg, workflow_id)

        return result

    async def _execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step."""
        if step.type == WorkflowStepType.TRANSFORM:
            processor = TransformProcessor(step.config)
            try:
                # Check timeout before executing
                if self._is_timeout():
                    raise WorkflowExecutionError("Workflow execution timeout", self.config.id)

                # Get input data from context or previous step
                input_data = context.get("data")
                if step.dependencies:
                    # Use the output of the last dependency as input
                    last_dep = step.dependencies[-1]
                    if last_dep in context:
                        dep_result = context[last_dep]
                        if isinstance(dep_result, dict) and "data" in dep_result:
                            input_data = dep_result["data"]

                step_context = {"data": input_data}

                if isinstance(step.config, dict) and "execute" in step.config:
                    result = await step.config["execute"](input_data)
                    return {"data": result}
                elif hasattr(step.config, "execute") and callable(step.config.execute):
                    result = await step.config.execute(input_data)
                    return {"data": result}
                elif isinstance(step.config, StepConfig):
                    if "execute" in step.config.params:
                        result = await step.config.params["execute"](input_data)
                        return {"data": result}
                    result = await processor.process(step_context)
                    if isinstance(result, ProcessorResult):
                        if result.error:
                            raise WorkflowExecutionError(result.error, self.config.id)
                        return result.data
                    return {"data": result}
                else:
                    raise WorkflowExecutionError(f"Invalid step configuration: {step.config}", self.config.id)
            except Exception as e:
                logger.error(f"Step execution failed: {str(e)}")
                raise WorkflowExecutionError(str(e), self.config.id)
        else:
            raise WorkflowExecutionError(f"Unsupported step type: {step.type}", self.config.id)

    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in workflow steps."""
        visited = set()
        path = set()

        def visit(step_id: str) -> bool:
            if step_id in path:
                return True
            if step_id in visited:
                return False

            visited.add(step_id)
            path.add(step_id)

            step = next((s for s in self.config.steps if s.id == step_id), None)
            if step and step.dependencies:
                for dep in step.dependencies:
                    if visit(dep):
                        return True

            path.remove(step_id)
            return False

        return any(visit(step.id) for step in self.config.steps)

    def _has_missing_dependencies(self) -> bool:
        """Check for missing dependencies in workflow steps."""
        step_ids = {step.id for step in self.config.steps}
        for step in self.config.steps:
            if step.dependencies:
                for dep in step.dependencies:
                    if dep not in step_ids:
                        return True
        return False

    def _are_dependencies_satisfied(self, step: WorkflowStep, results: Dict[str, Any]) -> bool:
        """Check if step dependencies are satisfied."""
        if not step.dependencies:
            return True
        return all(dep in results for dep in step.dependencies)

    def _is_timeout(self) -> bool:
        """Check if workflow execution has timed out."""
        if not self.start_time or not self.config.timeout:
            return False
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return elapsed >= self.config.timeout

    def stop(self):
        """Stop workflow execution."""
        self.status = "stopped"
        workflow_id = self.config.id
        self.state_manager.update_workflow_status(workflow_id, WorkflowStatus.FAILED)

    def get_node_status(self, node_id: str) -> Optional[str]:
        """Get status of a specific node."""
        try:
            workflow_id = self.config.id
            return self.state_manager.get_step_status(workflow_id, node_id)
        except Exception as e:
            logger.error(f"Failed to get node status: {str(e)}")
            return None

    def send_input_to_node(self, node_id: str, input_data: Dict[str, Any]) -> bool:
        """Send input to a specific node."""
        try:
            workflow_id = self.config.id
            self.state_manager.set_step_result(workflow_id, node_id, input_data)
            return True
        except Exception as e:
            logger.error(f"Failed to send input to node: {str(e)}")
            return False