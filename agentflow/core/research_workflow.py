"""
Research Workflow Module for AgentFlow
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging
from functools import partial

from .base_workflow import BaseWorkflow
from .config import WorkflowConfig, AgentConfig, ExecutionPolicies
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class DistributedStep:
    """
    Represents a distributed step in a research workflow
    """
    id: str
    name: str
    input_type: str
    output_type: str
    agents: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)
    completed: bool = False
    
    def add_agent(self, agent_id: str):
        """
        Add an agent to the step
        
        Args:
            agent_id: ID of the agent to add
        """
        self.agents.append(agent_id)
    
    def mark_completed(self):
        """
        Mark the step as completed
        """
        self.completed = True

class ResearchWorkflow(BaseWorkflow):
    """
    Specialized workflow for research-oriented tasks
    """
    
    def __init__(
        self, 
        workflow_def: Union[Dict[str, Any], WorkflowConfig]
    ):
        """
        Initialize a research workflow
        
        Args:
            workflow_def: Workflow configuration or dict
        """
        # Convert dict to WorkflowConfig if needed
        if isinstance(workflow_def, dict):
            workflow_def = WorkflowConfig(
                id=workflow_def.get('name', 'research_workflow'),
                name=workflow_def.get('name', 'Research Workflow'),
                description=workflow_def.get('description', ''),
                agents=workflow_def.get('agents', []),
                execution_policies=ExecutionPolicies(**workflow_def.get('execution_policies', {}))
            )

        super().__init__(workflow_def)
        self.research_steps: list = []
        self.agents: list = []
        
        # Set error_handling to an empty dictionary
        self.error_handling = {}

        # Initialize agents from config
        if workflow_def.agents:
            from .agent import Agent
            for agent_config in workflow_def.agents:
                agent = Agent(config=agent_config)
                self.agents.append(agent)

        # Initialize research steps from config
        if workflow_def.execution_policies and workflow_def.execution_policies.steps:
            for step in workflow_def.execution_policies.steps:
                self.add_research_step(DistributedStep(
                    id=str(step.get('step', 1)),
                    name=step.get('name', f'Step {step.get("step", 1)}'),
                    input_type=step.get('input_type', 'dict'),
                    output_type=step.get('output_type', 'dict'),
                    agents=step.get('agents', [])
                ))

    def add_research_step(self, step: DistributedStep):
        """Add a research step to the workflow"""
        self.research_steps.append(step)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the research workflow
        
        Args:
            input_data: Input data for the workflow
        
        Returns:
            Dict containing the workflow results
        """
        results = {}
        current_input = input_data.copy()

        # Use a semaphore to control concurrency
        semaphore = asyncio.Semaphore(min(len(self.research_steps), 4))

        # Prepare tasks for each step
        step_tasks = []
        for step in self.research_steps:
            # Create agents for this step if needed
            if not step.agents:
                step.agents = self.agents

            # Create a task for each step
            task = self._execute_step_with_semaphore(semaphore, step, current_input)
            step_tasks.append(task)

        # Execute steps concurrently
        step_results = await asyncio.gather(*step_tasks)

        # Process results
        for step, step_result in zip(self.research_steps, step_results):
            # Store the result
            results[f'step_{step.id}_result'] = step_result
            
            # Update current input for next step
            if isinstance(step_result, dict):
                current_input.update(step_result)
            else:
                current_input[f'step_{step.id}_output'] = step_result

            step.mark_completed()

        return results

    async def _execute_step_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                           step: DistributedStep, 
                                           input_data: Dict[str, Any]) -> Any:
        """
        Execute a step with concurrency control
        
        Args:
            semaphore: Concurrency control semaphore
            step: The step to execute
            input_data: Input data for the step
        
        Returns:
            Result of the step execution
        """
        async with semaphore:
            try:
                # Add a small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
                
                # Execute the step
                return await self._execute_step(step, input_data)
            except Exception as e:
                # Log the error and re-raise
                print(f"Error in step {step.id}: {str(e)}")
                raise

    async def _execute_step(self, step: DistributedStep, input_data: Dict[str, Any]) -> Any:
        """
        Execute a single step in the workflow
        
        Args:
            step: The step to execute
            input_data: Input data for the step
            
        Returns:
            Result of the step execution
        """
        # Execute the step using the execute_step method
        try:
            result = await self.execute_step(step.id, input_data)
            step.completed = True
            return result
        except Exception as e:
            logger.error(f"Error in step {step.id}: {str(e)}")
            raise

    async def execute_step(self, step_id: str, input_data: Dict[str, Any]) -> Any:
        """
        Execute a single step in the workflow. This method should be overridden by subclasses.

        Args:
            step_id: ID of the step to execute
            input_data: Input data for the step

        Returns:
            Result of the step execution
        """
        raise NotImplementedError("execute_step must be implemented by subclass")

    def initialize_state(self):
        """Initialize workflow state"""
        super().initialize_state()
        for step in self.research_steps:
            step.completed = False

    def get_agent(self, agent_id: str) -> Optional['Agent']:
        """
        Get agent by ID
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent if found, None otherwise
        """
        return next((agent for agent in self.agents if agent.id == agent_id), None)
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get current workflow status
        
        Returns:
            Workflow status dictionary
        """
        return {
            "total_steps": len(self.research_steps),
            "completed_steps": len([step for step in self.research_steps if step.completed]),
            "agents": [agent.id for agent in self.agents]
        }
    
    def stop(self):
        """
        Stop the research workflow
        """
        # Placeholder implementation for stopping workflow
        for agent in self.agents:
            agent.stop()
