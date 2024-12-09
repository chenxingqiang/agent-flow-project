"""
Research Workflow Module for AgentFlow
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from .base_workflow import BaseWorkflow
from .config import WorkflowConfig, AgentConfig, ExecutionPolicies
from .agent import Agent
import asyncio

@dataclass
class DistributedStep:
    """
    Represents a distributed step in a research workflow
    """
    id: str
    name: str
    input_type: str
    output_type: str
    agents: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
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
            # Prepare execution policies
            execution_policies = {
                'required_fields': workflow_def.get('required_inputs', []),
                'default_status': workflow_def.get('default_status', 'initialized'),
                'error_handling': workflow_def.get('error_handling', {}),
                'steps': workflow_def.get('steps', [
                    {
                        'step': workflow_def.get('step', 1),
                        'type': workflow_def.get('type', 'research'),
                        'name': workflow_def.get('name', 'Research Workflow'),
                        'description': workflow_def.get('description', ''),
                        'input_type': 'dict',
                        'output_type': 'dict',
                        'agents': [],
                        'outputs': workflow_def.get('outputs', [])
                    }
                ])
            }

            workflow_def = WorkflowConfig(
                id=workflow_def.get('name', 'research_workflow'),
                name=workflow_def.get('name', 'Research Workflow'),
                description=workflow_def.get('description', ''),
                agents=[],  # No agents by default
                execution_policies=ExecutionPolicies(**execution_policies)
            )

        super().__init__(workflow_def)
        self.research_steps: List[DistributedStep] = []
        self.agents: List[Agent] = []

        # Initialize agents from config
        if workflow_def.agents:
            for agent_config in workflow_def.agents:
                agent = Agent(config=agent_config)
                self.agents.append(agent)

        # Initialize research steps from config
        for step in workflow_def.execution_policies.steps:
            self.add_research_step(DistributedStep(
                id=str(step.get('step', 1)),  # Convert step to string
                name=step.get('name', f'Step {step.get("step", 1)}'),
                input_type=step.get('input_type', 'dict'),
                output_type=step.get('output_type', 'dict'),
                agents=step.get('agents', [])
            ))

    def initialize_state(self):
        """Initialize workflow state"""
        super().initialize_state()
        for step in self.research_steps:
            step.completed = False

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the research workflow
        
        Args:
            input_data: Input data for the workflow
        
        Returns:
            Workflow execution results
        """
        results = {}
        
        # Validate required inputs
        required_inputs = self.config.execution_policies.required_fields
        for required in required_inputs:
            if required not in input_data:
                raise ValueError(f"Missing required input: {required}")
        
        # Execute research steps in sequence
        for step in self.research_steps:
            step_results = {}
            
            # Execute each agent in the step
            for agent_id in step.agents:
                agent = self.get_agent(agent_id)
                if agent:
                    try:
                        agent_result = await agent.process(input_data)
                        step_results[agent_id] = agent_result
                    except Exception as e:
                        step_results[agent_id] = {"error": str(e)}
                        
            results[step.id] = step_results
            step.mark_completed()
            
        return results
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get agent by ID
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent if found, None otherwise
        """
        return next((agent for agent in self.agents if agent.id == agent_id), None)
    
    def add_research_step(self, step: Union[DistributedStep, Dict[str, Any]]):
        """
        Add a research workflow step
        
        Args:
            step: Configuration for the research step
        """
        if isinstance(step, dict):
            step = DistributedStep(**step)
        self.research_steps.append(step)
    
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
            
    def execute_step(self, step: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in the research workflow
        
        Args:
            step: Current step definition
            inputs: Input data for the step
            
        Returns:
            Dict containing step results
        """
        # If no agents are defined, return inputs
        if not step.get('agents', []):
            return inputs
        
        # Find an agent for this step
        agent_id = step['agents'][0]  # Take the first agent
        agent = self.get_agent(agent_id)
        
        if not agent:
            raise ValueError(f"No agent found for step: {step}")
        
        # Process the step with the selected agent
        try:
            result = asyncio.run(agent.process(inputs))
            
            # Validate the step output
            self.validate_step_output(step.get('step', 1), result)
            
            return result
        except Exception as e:
            # Handle step execution errors
            error_handler = self.error_handling.get('handler', self._default_error_handler)
            return error_handler(e, inputs)
