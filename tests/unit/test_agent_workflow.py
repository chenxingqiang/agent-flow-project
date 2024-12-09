import pytest
from unittest.mock import Mock, patch
import ray
from agentflow.core.research_workflow import ResearchWorkflow, DistributedStep
from agentflow.core.workflow import Workflow
from agentflow.core.config import WorkflowConfig, ExecutionPolicies, AgentConfig
from agentflow.core.node import Node, NodeState
from agentflow.core.rate_limiter import ModelRateLimiter
import tenacity
import os

@pytest.fixture
def mock_ell():
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return {"result": "mocked_result"}
        return wrapper
    return decorator

@pytest.fixture
def test_workflow_def():
    return WorkflowConfig(
        name="test_research_workflow",
        description="Test research workflow",
        execution_policies=ExecutionPolicies(
            required_fields=["research_topic", "deadline", "academic_level"],
            default_status="pending",
            error_handling={
                "missing_input_error": "Missing required inputs",
                "missing_field_error": "Missing required fields"
            },
            steps=[
                {
                    "step": 1,
                    "id": "1",
                    "name": "Research Step",
                    "agents": ["research_agent_1", "research_agent_2"],
                    "input_type": "research_topic",
                    "output_type": "research_findings",
                    "input": ["research_topic"],
                    "output": {"type": "research"}
                }
            ]
        ),
        agents=[
            AgentConfig(
                id="research_agent_1",
                name="Research Agent 1",
                model={
                    "name": "test-model",
                    "provider": "default"
                }
            ),
            AgentConfig(
                id="research_agent_2", 
                name="Research Agent 2",
                model={
                    "name": "test-model",
                    "provider": "default"
                }
            )
        ]
    )

@pytest.fixture
def minimal_agent_workflow_def():
    return {
        "name": "minimal_agent_workflow",
        "description": "Minimal agent workflow without execution policies",
        "agents": [
            AgentConfig(
                id="basic_agent",
                name="Basic Agent",
                model={
                    "name": "test-model",
                    "provider": "default"
                }
            ).model_dump()
        ]
    }

class TestResearchWorkflow(ResearchWorkflow):
    async def execute_step(self, step_id, input_data):
        return {"result": "test_result"}

    def initialize_state(self):
        super().initialize_state()

@pytest.fixture
def test_workflow(test_workflow_def):
    return TestResearchWorkflow(test_workflow_def)

def test_workflow_creation(test_workflow_def):
    """Test workflow creation"""
    workflow = TestResearchWorkflow(test_workflow_def)
    assert workflow is not None
    assert workflow.config.name == "test_research_workflow"
    assert len(workflow.config.agents) == 2
    assert len(workflow.config.execution_policies.steps) == 1

@pytest.mark.asyncio
async def test_workflow_execution(test_workflow):
    """Test workflow execution"""
    input_data = {
        "research_topic": "AI Ethics",
        "deadline": "2024-12-31",
        "academic_level": "Graduate"
    }
    
    result = await test_workflow.execute(input_data)
    assert result is not None
    assert isinstance(result, dict)

def test_workflow_step_processing(test_workflow):
    """Test individual step processing"""
    step = test_workflow.research_steps[0]
    assert step.id == "1"
    assert len(step.agents) == 2
    assert step.input_type == "research_topic"
    assert step.output_type == "research_findings"

def test_workflow_state_management(test_workflow):
    """Test workflow state management"""
    test_workflow.initialize_state()
    assert test_workflow.state["status"] == "pending"
    assert test_workflow.state["current_step"] == 0

def test_rate_limiter_integration():
    """Test rate limiter integration"""
    rate_limiter = ModelRateLimiter()
    assert rate_limiter is not None
    assert rate_limiter.max_retries == 3
    assert rate_limiter.retry_delay == 1

@pytest.mark.asyncio
async def test_distributed_step_execution():
    """Test distributed step execution"""
    step = DistributedStep(
        id="test_step",
        name="Test Step",
        input_type="test_input",
        output_type="test_output"
    )
    assert step.id == "test_step"
    assert step.name == "Test Step"
    assert not step.completed

@pytest.mark.asyncio
async def test_distributed_step_error_handling():
    """Test error handling in distributed step"""
    step = DistributedStep(
        id="error_step",
        name="Error Step",
        input_type="test_input",
        output_type="test_output"
    )
    assert not step.completed
    step.mark_completed()
    assert step.completed

@pytest.mark.asyncio
async def test_distributed_step_retry_mechanism():
    """Test retry mechanism in distributed step"""
    step = DistributedStep(
        id="retry_step",
        name="Retry Step",
        input_type="test_input",
        output_type="test_output"
    )
    step.add_agent("test_agent")
    assert len(step.agents) == 1

@pytest.mark.asyncio
async def test_workflow_error_propagation(test_workflow):
    """Test error propagation in workflow"""
    test_workflow.initialize_state()
    assert test_workflow.state["errors"] == []

def test_workflow_step_validation(test_workflow):
    """Test step validation"""
    step = test_workflow.research_steps[0]
    assert step.id == "1"
    assert step.name == "Research Step"

@pytest.mark.asyncio
async def test_minimal_workflow_initialization(minimal_agent_workflow_def):
    """Test initialization of workflow without execution policies."""
    workflow = ResearchWorkflow(minimal_agent_workflow_def)
    assert workflow.required_fields == []
    assert workflow.error_handling == {}
    assert workflow.default_status == 'initialized'

@pytest.fixture(autouse=True)
def setup_ray():
    """Initialize Ray for testing"""
    if not ray.is_initialized():
        ray.init(local_mode=True)
    yield
    if ray.is_initialized():
        ray.shutdown()

if __name__ == "__main__":
    pytest.main([__file__])
