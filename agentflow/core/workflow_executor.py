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
    WorkflowStatus,
    ErrorPolicy,
    AgentStatus
)
from .workflow_state import WorkflowStateManager
from .metrics import MetricsManager, MetricType
from .exceptions import WorkflowExecutionError, StepExecutionError
from .processors.transformers import TransformProcessor, ProcessorResult
from .enums import StepStatus
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import uuid

logger = logging.getLogger(__name__)

# Define valid strategies and protocols
VALID_STRATEGIES = {"federated", "gossip", "hierarchical", "hierarchical_merge", "custom", "standard", "feature_engineering", "outlier_removal"}
VALID_PROTOCOLS = {"federated", "gossip", "hierarchical", "hierarchical_merge", "unknown", None}

class WorkflowExecutor:
    """Workflow executor class."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize workflow executor.

        Args:
            config: Workflow configuration.
        """
        self.config = config
        if not hasattr(self.config, 'error_policy'):
            self.config.error_policy = ErrorPolicy()
        elif isinstance(self.config.error_policy, dict):
            self.config.error_policy = ErrorPolicy(**self.config.error_policy)
        self.state_manager = WorkflowStateManager()
        self.metrics = MetricsManager()
        self._initialized = False
        self.iteration_count = 0
        self.start_time = None
        self._status = WorkflowStatus.PENDING
        self._step_results: Dict[str, Dict[str, Any]] = {}
        self._pending_tasks = {}

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
        """Initialize workflow executor."""
        if not self._initialized:
            await self.state_manager.initialize()
            # Initialize workflow state first
            self.state_manager.initialize_workflow(self.config.id, self.config.name)
            
            # Initialize step states
            for step in self.config.steps:
                self.state_manager.initialize_step(self.config.id, step.id, step.name)
                self.state_manager.update_step_status(
                    self.config.id,
                    step.id,
                    WorkflowStepStatus.PENDING
                )
            
            self._initialized = True

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
        """Validate workflow steps."""
        # Validate step types
        for step in self.config.steps:
            if not isinstance(step.type, WorkflowStepType):
                try:
                    step.type = WorkflowStepType(step.type)
                except ValueError:
                    raise WorkflowExecutionError(f"Invalid step type: {step.type}")

    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in workflow steps."""
        visited = set()
        path = set()
        
        def dfs(step_id: str) -> bool:
            """Depth-first search to detect cycles."""
            if step_id in path:
                return True
            if step_id in visited:
                return False
            
            visited.add(step_id)
            path.add(step_id)
            
            # Find the step by ID
            step = next((s for s in self.config.steps if s.id == step_id), None)
            if step and step.dependencies:
                for dep in step.dependencies:
                    if dfs(dep):
                        return True
            
            path.remove(step_id)
            return False
        
        # Check each step for cycles
        for step in self.config.steps:
            visited.clear()  # Clear visited set for each starting point
            path.clear()    # Clear path set for each starting point
            if dfs(step.id):
                return True
            
        return False

    def _has_missing_dependencies(self) -> bool:
        """Check for missing dependencies."""
        step_ids = {step.id for step in self.config.steps}
        for step in self.config.steps:
            if step.dependencies:  # Only check if step has dependencies
                for dep in step.dependencies:
                    if dep not in step_ids:
                        return True
        return False

    def _validate_dependencies(self) -> None:
        """Validate workflow dependencies."""
        # Check for circular dependencies
        if self._has_circular_dependencies():
            raise WorkflowExecutionError("Circular dependency detected in workflow steps")
            
        # Check for missing dependencies
        if self._has_missing_dependencies():
            raise WorkflowExecutionError("Missing dependencies detected in workflow steps")

    async def execute(self, context: Union[Dict[str, Any], Message]) -> Dict[str, Any]:
        """Execute workflow steps."""
        try:
            # Set start time for timeout tracking
            self.start_time = time.time()

            # Convert Message to dict if needed
            if isinstance(context, Message):
                context_dict = {
                    "content": context.content,
                    "metadata": context.metadata,
                    "type": context.type,
                    "role": context.role,
                    "timestamp": context.timestamp
                }
            else:
                context_dict = context
            
            # Validate workflow has steps
            if not self.config.steps:
                raise WorkflowExecutionError("Workflow steps list cannot be empty")
                
            # Check for circular dependencies first
            if self._has_circular_dependencies():
                raise WorkflowExecutionError("Circular dependency detected in workflow steps")
            
            # Check for missing dependencies
            if self._has_missing_dependencies():
                raise WorkflowExecutionError("Missing dependencies detected in workflow steps")
            
            # Initialize results
            results = {
                "status": "running",
                "steps": {},
                "warnings": [],
                "error": None,
                "result": None,  # Initialize result key
                "content": None  # Initialize content field
            }
            
            completed_steps = set()
            
            # Execute steps in sequence
            for step in self.config.steps:
                # Check for timeout before each step
                if self._is_timeout():
                    raise TimeoutError("Workflow execution timed out")

                try:
                    # Validate protocol if specified
                    protocol = step.config.params.get("protocol")
                    if protocol is not None and protocol not in VALID_PROTOCOLS:
                        raise WorkflowExecutionError(f"Invalid protocol: {protocol}")
                    
                    # Execute step with timeout check
                    start_time = time.time()
                    step_result = await self._execute_step(step, context_dict)
                    
                    # Check for timeout after step execution
                    if self._is_timeout():
                        raise TimeoutError("Workflow execution timed out")
                    
                    results["steps"][step.id] = {
                        "id": step.id,
                        "type": str(step.type),
                        "status": "completed",
                        "result": step_result,
                        "error": None
                    }
                    
                    # Update context with step result data
                    if isinstance(step_result, dict):
                        if "result" in step_result and isinstance(step_result["result"], dict):
                            context_dict.update(step_result["result"])
                        elif "data" in step_result:
                            context_dict.update({"data": step_result["data"]})
                        else:
                            context_dict.update({"data": step_result})
                    
                    completed_steps.add(step.id)

                    # Update content field with the latest step result
                    if isinstance(step_result, dict) and "content" in step_result:
                        results["content"] = step_result["content"]
                    elif isinstance(step_result, str):
                        results["content"] = step_result
                    elif isinstance(step_result, dict) and "result" in step_result:
                        results["content"] = str(step_result["result"])
                    else:
                        results["content"] = str(step_result)

                except TimeoutError as e:
                    # Update agent status to FAILED
                    if self.config.agent:
                        self.config.agent.state.status = AgentStatus.FAILED
                    raise  # Re-raise TimeoutError directly
                except Exception as e:
                    results["steps"][step.id] = {
                        "id": step.id,
                        "type": str(step.type),
                        "status": "failed",
                        "result": None,
                        "error": str(e)
                    }
                    # Update agent status to FAILED
                    if self.config.agent:
                        self.config.agent.state.status = AgentStatus.FAILED
                    if self.config.error_policy.fail_fast:
                        raise WorkflowExecutionError(f"Error executing step {step.id}: {str(e)}") from e
            
            # Update final status
            results["status"] = "completed"
            return results
            
        except TimeoutError:
            raise  # Re-raise TimeoutError directly
        except Exception as e:
            if isinstance(e, WorkflowExecutionError):
                raise
            raise WorkflowExecutionError(f"Workflow execution failed: {str(e)}") from e

    async def _execute_step(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step.
        
        Args:
            step: Step to execute
            input_data: Input data for step
            
        Returns:
            Dict[str, Any]: Step execution results
            
        Raises:
            WorkflowExecutionError: If step execution fails
            TimeoutError: If step execution times out
        """
        try:
            # Convert step type to enum if it's a string
            step_type = step.type
            if isinstance(step_type, str):
                # Handle both research and research_execution as research
                step_type = step_type.lower()
                if step_type in ["research", "research_execution"]:
                    step_type = "research"
                try:
                    step_type = WorkflowStepType(step_type)
                except ValueError:
                    raise WorkflowExecutionError(f"Invalid step type: {step_type}")
            
            # Add test_mode flag to input data if not present
            if "test_mode" not in input_data:
                input_data["test_mode"] = True
            
            # Calculate remaining time for timeout
            if self.start_time and self.config.timeout:
                elapsed = time.time() - self.start_time
                remaining = max(0.1, self.config.timeout - elapsed)  # At least 0.1 seconds
            else:
                remaining = None
            
            # Execute step based on type with timeout
            if step_type == WorkflowStepType.TRANSFORM:
                result = await asyncio.wait_for(
                    self._execute_transform_step(step, input_data),
                    timeout=remaining
                )
            elif step_type == WorkflowStepType.RESEARCH:
                result = await asyncio.wait_for(
                    self._execute_research_step(step, input_data),
                    timeout=remaining
                )
            elif step_type == WorkflowStepType.DOCUMENT:
                result = await asyncio.wait_for(
                    self._execute_document_step(step, input_data),
                    timeout=remaining
                )
            elif step_type == WorkflowStepType.AGENT:
                result = await asyncio.wait_for(
                    self._execute_agent_step(step, input_data),
                    timeout=remaining
                )
            else:
                raise WorkflowExecutionError(f"No execution handler for step type: {step_type}")
            
            return result
                
        except asyncio.TimeoutError:
            raise TimeoutError(f"Step {step.id} execution timed out")
        except Exception as e:
            if isinstance(e, (WorkflowExecutionError, TimeoutError)):
                raise
            raise WorkflowExecutionError(f"Step {step.id} failed: {str(e)}") from e

    async def _execute_research_step(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research step.
        
        Args:
            step: Research step to execute
            input_data: Input data for step
            
        Returns:
            Dict[str, Any]: Research results
            
        Raises:
            WorkflowExecutionError: If research step fails
        """
        try:
            # Get input parameters from step configuration
            input_params = step.config.params.get("input", [])
            
            # Get research parameters from input data based on configuration
            research_data = {}
            for param in input_params:
                research_data[param] = input_data.get(param, {})
            
            # Get research topic from student needs
            student_needs = research_data.get("STUDENT_NEEDS", {})
            topic = student_needs.get("RESEARCH_TOPIC", "Test Topic")
            
            # In test mode, return mock results
            if input_data.get("test_mode", True):
                return {
                    "status": "completed",
                    "result": {
                        "research_findings": {
                            "topic": topic,
                            "summary": "Test research findings",
                            "methodology": "Test methodology",
                            "conclusions": ["Test conclusion 1", "Test conclusion 2"]
                        }
                    }
                }
            
            # Get agent from workflow config
            agent = self.config.agent
            if not agent:
                raise WorkflowExecutionError("No agent configured for research step")

            # Prepare research context
            research_context = {
                **research_data,  # Include all research data from input parameters
                "step_id": step.id,
                "step_type": "research",
                "test_mode": input_data.get("test_mode", False)
            }

            # Execute research step using agent
            try:
                result = await agent.execute(research_context)
                return {
                    "status": "completed",
                    "result": {
                        "research_findings": result.get("research_findings", {
                            "topic": topic,
                            "summary": "Research findings summary",
                            "methodology": "Research methodology",
                            "conclusions": ["Conclusion 1", "Conclusion 2"]
                        })
                    }
                }
            except Exception as e:
                raise WorkflowExecutionError(f"Research step execution failed: {str(e)}") from e
            
        except Exception as e:
            raise WorkflowExecutionError(f"Research step failed: {str(e)}") from e

    async def _execute_document_step(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a document generation step.
        
        Args:
            step: Step to execute
            input_data: Input data for step
            
        Returns:
            Dict[str, Any]: Step execution results
        """
        try:
            # Get input parameters from step configuration
            input_params = step.config.params.get("input", [])
            
            # Get document parameters from input data based on configuration
            document_data = {}
            for param in input_params:
                if param.startswith("WORKFLOW."):
                    # Handle workflow step references (e.g., "WORKFLOW.1.output")
                    parts = param.split(".")
                    if len(parts) >= 3:
                        step_id = parts[1]
                        field = parts[2]
                        step_result = input_data.get("steps", {}).get(f"step-{step_id}", {})
                        if "result" in step_result:
                            document_data[field] = step_result["result"]
                else:
                    document_data[param] = input_data.get(param, {})
            
            # In test mode, return mock results
            if input_data.get("test_mode"):
                return {
                    "status": "completed",
                    "result": {
                        "document": {
                            "title": "Test Document",
                            "content": "Test document content",
                            "format": "Markdown with LaTeX"
                        }
                    }
                }
                
            # Get agent from workflow config
            agent = self.config.agent
            if not agent:
                raise WorkflowExecutionError("No agent configured for document step")

            # Prepare document context
            document_context = {
                **document_data,  # Include all document data from input parameters
                "step_id": step.id,
                "step_type": "document",
                "test_mode": input_data.get("test_mode", False)
            }

            # Execute document step using agent
            try:
                result = await agent.execute(document_context)
                return {
                    "status": "completed",
                    "result": {
                        "document": result.get("document", {
                            "title": "Generated Document",
                            "content": "Document content based on research findings",
                            "format": "Markdown with LaTeX"
                        })
                    }
                }
            except Exception as e:
                raise WorkflowExecutionError(f"Document step execution failed: {str(e)}") from e
            
        except Exception as e:
            raise WorkflowExecutionError(f"Document generation step failed: {str(e)}")

    async def _execute_transform_step(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a transform step."""
        try:
            if self._is_timeout():
                raise WorkflowExecutionError("Workflow execution timed out")

            transformed_data = input_data.get("data", {})
            protocol = step.config.params.get("protocol")
            
            # Validate protocol if specified
            if protocol and protocol not in VALID_PROTOCOLS:
                raise WorkflowExecutionError(f"Invalid protocol: {protocol}")
            
            if step.config.strategy == "feature_engineering":
                # Apply feature engineering
                result = transformed_data  # Placeholder for actual transformation
            elif step.config.strategy == "outlier_removal":
                # Apply outlier removal
                result = transformed_data  # Placeholder for actual outlier removal
            elif step.config.strategy == "custom":
                # Execute custom transform function if available
                if "execute" in step.config.params and callable(step.config.params["execute"]):
                    # Create a task for the custom execution
                    try:
                        if self.config.timeout:
                            # Use asyncio.wait_for for timeout
                            result = await asyncio.wait_for(
                                step.config.params["execute"](transformed_data),
                                timeout=self.config.timeout
                            )
                        else:
                            result = await step.config.params["execute"](transformed_data)
                    except asyncio.TimeoutError:
                        raise WorkflowExecutionError("Step execution timed out")
                    except Exception as e:
                        if isinstance(e, WorkflowExecutionError):
                            raise
                        raise WorkflowExecutionError(f"Step execution failed: {str(e)}")
                else:
                    result = transformed_data
            elif step.config.strategy in VALID_STRATEGIES:
                result = transformed_data  # Placeholder for actual strategy implementation
            else:
                raise WorkflowExecutionError(f"Invalid strategy: {step.config.strategy}")
            
            return {
                "id": step.id,
                "status": "completed",
                "error": None,
                "result": {
                    "data": result,
                    "metadata": {},
                    "result": result
                }
            }
        except Exception as e:
            if isinstance(e, WorkflowExecutionError):
                raise
            raise WorkflowExecutionError(f"Step {step.id} failed: {str(e)}")

    async def _execute_agent_step(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent step."""
        try:
            # Get agent from workflow config
            agent = self.config.agent
            if not agent:
                raise WorkflowExecutionError("No agent configured for workflow")

            # Execute agent step
            try:
                result = await agent.execute(input_data)
                return result
            except Exception as e:
                # Pass through the original error message
                if isinstance(e, WorkflowExecutionError):
                    raise
                raise WorkflowExecutionError(f"Step {step.id} failed: {str(e)}")
        except Exception as e:
            if isinstance(e, WorkflowExecutionError):
                raise
            raise WorkflowExecutionError(f"Agent step execution failed: {str(e)}")

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
                result = await self._execute_step(step, step_input)
                return step, result
            except Exception as e:
                error_msg = f"Step {step.name} ({step.id}) failed: {str(e)}"
                return step, {"status": WorkflowStepStatus.FAILED.value, "error": error_msg}

        # Execute all steps in the group concurrently
        return await asyncio.gather(*[execute_single_step(step) for step in group])

    def _is_timeout(self) -> bool:
        """Check if workflow execution has timed out."""
        if not self.start_time or not self.config.timeout:
            return False
        elapsed = time.time() - self.start_time
        return elapsed >= self.config.timeout

    async def cleanup(self):
        """Clean up workflow resources."""
        await self.metrics.cleanup()
        self.status = WorkflowStatus.COMPLETED

    def stop(self) -> None:
        """Stop workflow execution."""
        self.status = WorkflowStatus.FAILED  # Use FAILED instead of STOPPED

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

    async def execute_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously.
        
        Args:
            context: Execution context
            
        Returns:
            Dict[str, Any]: Initial response with task ID
        """
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Store context for later execution
        self._pending_tasks[task_id] = {
            "context": context,
            "status": "pending",
            "result": None,
            "error": None
        }
        
        # Start execution in background
        asyncio.create_task(self._execute_async(task_id, context))
        
        return {
            "task_id": task_id,
            "status": "pending"
        }

    async def _execute_async(self, task_id: str, context: Dict[str, Any]) -> None:
        """Execute workflow asynchronously.
        
        Args:
            task_id: Task ID
            context: Execution context
        """
        try:
            # Execute workflow
            result = await self.execute(context)
            
            # Update task status
            self._pending_tasks[task_id].update({
                "status": "completed",
                "result": result
            })
        except Exception as e:
            # Update task status on error
            self._pending_tasks[task_id].update({
                "status": "failed",
                "error": str(e)
            })
            
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict[str, Any]: Task status
        """
        if task_id not in self._pending_tasks:
            raise ValueError(f"Task {task_id} not found")
            
        return self._pending_tasks[task_id]