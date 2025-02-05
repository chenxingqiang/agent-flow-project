"""Workflow executor module."""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from types import SimpleNamespace
from datetime import datetime
from .workflow_types import (
    WorkflowStep,
    WorkflowStepType,
    StepConfig,
    WorkflowStepStatus,
    Message,
    WorkflowConfig,
    WorkflowStatus
)
from .workflow_state import WorkflowStateManager
from .metrics import MetricsManager, MetricType
from .exceptions import WorkflowExecutionError, StepExecutionError
from .processors.transformers import TransformProcessor, ProcessorResult
from .enums import StepStatus
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# Define valid strategies and protocols
VALID_STRATEGIES = {"federated", "gossip", "hierarchical", "hierarchical_merge", "custom"}
VALID_PROTOCOLS = {"federated", "gossip", "hierarchical", "hierarchical_merge", "unknown"}

class WorkflowExecutor:
    """Workflow executor class."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize workflow executor.

        Args:
            config: Workflow configuration.
        """
        self.config = config
        self.state_manager = WorkflowStateManager()
        self.metrics = MetricsManager()
        self._initialized = False
        self.iteration_count = 0
        self.start_time = None
        self._status = WorkflowStatus.PENDING
        self._step_results: Dict[str, Dict[str, Any]] = {}

    @property
    def status(self) -> WorkflowStatus:
        """Get workflow status."""
        return self._status

    @status.setter
    def status(self, value: WorkflowStatus):
        """Set workflow status."""
        self._status = value
        self.state_manager.update_workflow_status(
            self.config.id,
            value
        )

    async def initialize(self):
        """Initialize the workflow executor."""
        # Initialize state manager
        self.state_manager = WorkflowStateManager()
        
        # Initialize step statuses
        for step in self.config.steps:
            self.state_manager.update_step_status(
                self.config.id,
                step.id,
                WorkflowStepStatus.PENDING
            )
            
        # Set workflow status to pending
        self.status = WorkflowStatus.PENDING

        # Set initialization flag
        self._initialized = True
        self.status = WorkflowStatus.INITIALIZED

    def _validate_input(self, context: Dict[str, Any]) -> None:
        """Validate workflow input.
        
        Args:
            context: Input context to validate
            
        Raises:
            TypeError: If input contains non-serializable types
            ValueError: If required fields are missing
        """
        # Check for required fields
        if "data" not in context:
            raise ValueError("Required field 'data' is missing in input context")
            
        # Check for non-serializable types (like functions)
        def check_serializable(obj: Any) -> None:
            if callable(obj):
                raise TypeError(f"Functions are not allowed in input data: {obj}")
            elif isinstance(obj, dict):
                for value in obj.values():
                    check_serializable(value)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    check_serializable(item)
                    
        check_serializable(context)

    def _validate_workflow_steps(self) -> None:
        """Validate workflow steps.
        
        Raises:
            WorkflowExecutionError: If any step has an invalid strategy or protocol
        """
        for step in self.config.steps:
            if step.config.strategy not in VALID_STRATEGIES:
                raise WorkflowExecutionError(f"Invalid strategy '{step.config.strategy}' in step {step.id}", self.config.id)
            if step.config.params.get("protocol") not in VALID_PROTOCOLS:
                raise WorkflowExecutionError(f"Invalid protocol '{step.config.params.get('protocol')}' in step {step.id}", self.config.id)

    async def execute(self, context: Union[Dict[str, Any], Message]) -> Dict[str, Any]:
        """Execute the workflow with the given context.
        
        Args:
            context: Input context for the workflow
            
        Returns:
            Dict containing workflow execution results
            
        Raises:
            TypeError: If input contains non-serializable types
            ValueError: If required fields are missing
            WorkflowExecutionError: If workflow execution fails
        """
        if not self._initialized:
            raise WorkflowExecutionError("Workflow not initialized. Call initialize() first.", self.config.id)

        if isinstance(context, Message):
            context = {"data": context.content}
        elif not isinstance(context, dict):
            context = {"data": context}
            
        # Validate input and workflow steps
        self._validate_input(context)
        self._validate_workflow_steps()

        try:
            # Update workflow status
            self.state_manager.update_workflow_status(self.config.id, WorkflowStatus.RUNNING)

            # Execute workflow steps with timeout
            result = await asyncio.wait_for(
                self._execute_steps(context),
                timeout=self.config.timeout
            )

            # Update workflow status
            self.state_manager.update_workflow_status(self.config.id, WorkflowStatus.COMPLETED)
            
            # Add status to result
            result["status"] = "completed"

            return result

        except asyncio.TimeoutError as e:
            self.state_manager.update_workflow_status(self.config.id, WorkflowStatus.FAILED)
            raise WorkflowExecutionError("Workflow execution timeout", self.config.id) from e
        except Exception as e:
            self.state_manager.update_workflow_status(self.config.id, WorkflowStatus.FAILED)
            if isinstance(e, StepExecutionError):
                raise WorkflowExecutionError(str(e), self.config.id) from e
            raise WorkflowExecutionError(str(e), self.config.id) from e

    async def _execute_steps(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps.
        
        Args:
            context: Input context for the workflow
            
        Returns:
            Dict containing workflow execution results
        """
        result = {
            "steps": {},
            "status": "completed",
            "final_result": {
                "data": context.get("data"),
                "attempts": 1,
                "execution_time": 0.0
            }
        }
        
        start_time = time.time()
        
        try:
            for step in self.config.steps:
                step_result = await self._execute_step_with_retry(step, context)
                result["steps"][step.id] = {
                    "attempts": step_result.get("attempts", 1),
                    "execution_time": step_result.get("execution_time", 0.0),
                    "result": step_result,
                    "start_time": time.time()
                }
                context.update({"data": step_result.get("data", step_result)})
                
            result["final_result"]["execution_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            raise WorkflowExecutionError(f"Error executing workflow: {str(e)}", self.config.id) from e

    async def _execute_step_with_retry(self, step: WorkflowStep, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step with retry mechanism.
        
        Args:
            step: The workflow step to execute
            step_input: Input data for the step
            
        Returns:
            Dict containing step execution results
            
        Raises:
            WorkflowExecutionError: If step execution fails after all retries
        """
        retry_count = 0
        last_error = None
        start_time = time.time()
        
        while retry_count <= step.config.max_retries:
            try:
                # Execute the step with timeout
                result = await asyncio.wait_for(
                    step.execute(step_input),
                    timeout=self.config.timeout / len(self.config.steps)  # Divide timeout among steps
                )
                
                # Add execution metadata
                if isinstance(result, dict):
                    result["attempts"] = retry_count + 1
                    result["execution_time"] = time.time() - start_time
                    return result
                return {
                    "data": result,
                    "attempts": retry_count + 1,
                    "execution_time": time.time() - start_time
                }
                    
            except asyncio.TimeoutError as e:
                last_error = e
                retry_count += 1
                if retry_count > step.config.max_retries:
                    raise WorkflowExecutionError(f"Step {step.id} execution timed out after {retry_count} retries", self.config.id) from e
                await asyncio.sleep(step.config.retry_delay * (step.config.retry_backoff ** retry_count))
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count > step.config.max_retries:
                    raise WorkflowExecutionError(f"Step {step.id} execution failed after {retry_count} retries: {str(e)}", self.config.id) from e
                await asyncio.sleep(step.config.retry_delay * (step.config.retry_backoff ** retry_count))
        
        # This should never be reached due to the exceptions above, but adding for type safety
        raise WorkflowExecutionError(f"Step {step.id} execution failed: {str(last_error)}", self.config.id)

    def _get_step_data(self, step_result: Dict[str, Any]) -> Any:
        """Extract data from step result."""
        return step_result.get("data", step_result)

    def _get_step_result(self, step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get step result with proper structure."""
        if not step_result:
            return {"data": None, "status": "failed", "continue": False}
        if not isinstance(step_result, dict):
            return {"data": step_result, "status": "completed", "continue": False}
        if "data" not in step_result:
            return {"data": step_result, "status": "completed", "continue": False}
        return step_result

    def _get_parallel_step_groups(self) -> List[List[WorkflowStep]]:
        """Group steps that can be executed in parallel.
        
        Returns:
            List of lists, where each inner list contains steps that can be executed in parallel
        """
        # Track which steps have been assigned to groups
        assigned_steps = set()
        groups = []
        
        while len(assigned_steps) < len(self.config.steps):
            # Find all steps that can be executed next (all dependencies satisfied)
            current_group = []
            for step in self.config.steps:
                if step.id in assigned_steps:
                    continue
                    
                # Check if all dependencies are in assigned steps
                if step.dependencies:
                    if not all(dep in {s.id for s in [s for g in groups for s in g]} for dep in step.dependencies):
                        continue
                
                current_group.append(step)
                assigned_steps.add(step.id)
            
            if current_group:
                groups.append(current_group)
            else:
                # This should never happen if dependencies are properly configured
                raise WorkflowExecutionError("Invalid step dependencies detected")
                
        return groups

    async def _execute_step_group(self, group: List[WorkflowStep], context: Dict[str, Any], step_results: Dict[str, Any]) -> List[Tuple[WorkflowStep, Dict[str, Any]]]:
        """Execute a group of steps in parallel.
        
        Args:
            group: List of steps to execute in parallel
            context: Current workflow context
            step_results: Results from previously executed steps
            
        Returns:
            List of tuples containing (step, result) pairs
        """
        async def execute_single_step(step: WorkflowStep) -> Tuple[WorkflowStep, Dict[str, Any]]:
            try:
                # Prepare step input
                step_input = context.copy()
                if step.dependencies:
                    if len(step.dependencies) == 1:
                        dep_id = step.dependencies[0]
                        if dep_id not in step_results:
                            raise ValueError(f"Dependent step {dep_id} has not been executed")
                        dep_result = step_results[dep_id]
                        if isinstance(dep_result, dict) and "data" in dep_result:
                            step_input["data"] = dep_result["data"]
                        else:
                            step_input["data"] = dep_result
                    else:
                        dependent_results = {}
                        for dep_id in step.dependencies:
                            if dep_id not in step_results:
                                raise ValueError(f"Dependent step {dep_id} has not been executed")
                            dep_result = step_results[dep_id]
                            if isinstance(dep_result, dict) and "data" in dep_result:
                                dependent_results[dep_id] = dep_result["data"]
                            else:
                                dependent_results[dep_id] = dep_result
                        step_input["data"] = dependent_results

                step_input["step_id"] = step.id
                result = await self._execute_step_with_retry(step, step_input)
                return step, result
            except Exception as e:
                error_msg = f"Step {step.name} ({step.id}) failed: {str(e)}"
                return step, {"status": WorkflowStepStatus.FAILED.value, "error": error_msg}

        # Execute all steps in the group concurrently
        return await asyncio.gather(*[execute_single_step(step) for step in group])

    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in workflow."""
        visited = set()
        path = set()

        def dfs(step_id: str) -> bool:
            if step_id in path:
                raise WorkflowExecutionError("Circular dependency", self.config.id)
            if step_id in visited:
                return False

            visited.add(step_id)
            path.add(step_id)

            step = next((s for s in self.config.steps if s.id == step_id), None)
            if step and step.dependencies:
                for dep in step.dependencies:
                    if dfs(dep):
                        return True

            path.remove(step_id)
            return False

        for step in self.config.steps:
            if dfs(step.id):
                return True
        return False

    def _has_missing_dependencies(self) -> bool:
        """Check for missing dependencies in workflow."""
        step_ids = {step.id for step in self.config.steps}
        for step in self.config.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise WorkflowExecutionError(f"Step {step.id} depends on non-existent step {dep}", self.config.id)
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
        elapsed = time.time() - self.start_time
        return elapsed >= self.config.timeout

    def _validate_step_types(self) -> None:
        """Validate step types."""
        valid_step_types = {
            WorkflowStepType.TRANSFORM,
            WorkflowStepType.FILTER,
            WorkflowStepType.AGGREGATE,
            WorkflowStepType.ANALYZE,
            WorkflowStepType.RESEARCH_EXECUTION,
            WorkflowStepType.DOCUMENT_GENERATION,
            WorkflowStepType.AGENT
        }
        for step in self.config.steps:
            if isinstance(step.type, str):
                try:
                    step_type = WorkflowStepType(step.type)
                except ValueError:
                    raise ValueError(f"Invalid step type: {step.type}")
            else:
                step_type = step.type

            if step_type not in valid_step_types:
                raise ValueError(f"Invalid step type: {step_type}")

    async def cleanup(self):
        """Clean up workflow resources."""
        await self.metrics.cleanup()
        self.status = WorkflowStatus.COMPLETED

    def stop(self):
        """Stop workflow execution."""
        self.status = WorkflowStatus.STOPPED

    def get_node_status(self, node_id: str) -> Optional[WorkflowStepStatus]:
        """Get the status of a workflow node."""
        status = self.state_manager.get_step_status(self.config.id, node_id)
        if status is None:
            return WorkflowStepStatus.PENDING
        if isinstance(status, str):
            return WorkflowStepStatus(status)
        return status

    def send_input_to_node(self, node_id: str, input_data: Dict[str, Any]) -> bool:
        """Send input to a specific node."""
        try:
            workflow_id = self.config.id
            self.state_manager.set_step_result(workflow_id, node_id, input_data)
            return True
        except Exception as e:
            logger.error(f"Failed to send input to node: {str(e)}")
            return False

    def validate_context(self, context: Dict[str, Any]) -> None:
        """Validate the workflow context.
        
        Args:
            context: Workflow context to validate
        
        Raises:
            ValueError: If context is missing required fields
        """
        # Define required fields for research workflow
        required_fields = [
            'research_topic', 
            'academic_level', 
            'research_methodology', 
            'deadline'
        ]
        
        for field in required_fields:
            if field not in context:
                raise ValueError(f"Missing required field: {field}")
        
        # Optional additional validation can be added here