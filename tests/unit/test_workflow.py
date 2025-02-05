"""Tests for workflow functionality."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agentflow.core.workflow import WorkflowEngine, WorkflowInstance
from agentflow.core.metrics import MetricType
from agentflow.core.workflow_types import (
    WorkflowConfig, WorkflowStep, StepConfig, WorkflowStepType
)
from agentflow.core.enums import WorkflowStatus
from agentflow.agents.agent_types import AgentType

class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name, async_mode=False):
        self.name = name
        self.async_mode = async_mode
        self.executed = False
    
    async def initialize(self):
        """Initialize mock agent."""
        pass
    
    async def execute(self, context):
        self.executed = True
        return {"result": f"{self.name}_result"}

class TestWorkflowExecution:
    """Test workflow execution."""
    
    @pytest.fixture
    def workflow_config(self):
        """Create workflow configuration."""
        return WorkflowConfig(
            id="test-workflow",
            name="Test Workflow",
            max_iterations=5,
            timeout=3600,
            error_policy={
                "fail_fast": True,
                "ignore_warnings": False,
                "max_errors": 10,
                "retry_policy": {
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "backoff": 2.0,
                    "max_delay": 60.0
                }
            },
            steps=[
                WorkflowStep(
                    id="step1",
                    name="Agent Step",
                    type=WorkflowStepType.AGENT,
                    config=StepConfig(
                        type=WorkflowStepType.AGENT,
                        agent_type=AgentType.GENERIC,
                        config=None
                    )
                )
            ]
        )
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, workflow_config):
        """Test workflow execution."""
        engine = WorkflowEngine(workflow_config)
        await engine.initialize()
        
        # Create workflow instance
        instance = await engine.create_workflow("test_workflow", workflow_config)
        instance.context["test_mode"] = True
        
        # Execute workflow
        result = await engine.execute_workflow(instance)
        
        assert result is not None
        assert result["status"] == WorkflowStatus.COMPLETED.value
        assert "result" in result
        assert "steps" in result["result"]
        assert len(result["result"]["steps"]) == 1
        assert result["result"]["steps"][0]["id"] == "step1"
        assert result["result"]["steps"][0]["status"] == WorkflowStatus.COMPLETED.value
    
    @pytest.mark.asyncio
    async def test_workflow_validation(self, workflow_config):
        """Test workflow validation."""
        engine = WorkflowEngine(workflow_config)
        await engine.initialize()
        
        # Test with invalid workflow instance
        with pytest.raises(ValueError, match="Workflow instance has no steps"):
            instance = await engine.create_workflow("test_workflow")
            await engine.execute_workflow(instance)
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, workflow_config):
        """Test workflow error handling."""
        engine = WorkflowEngine(workflow_config)
        await engine.initialize()
        
        # Create workflow instance with failing step
        instance = await engine.create_workflow("test_workflow", workflow_config)
        instance.steps[0].type = WorkflowStepType.AGENT  # Change to agent type to trigger error
        
        # Execute workflow
        result = await engine.execute_workflow(instance)
        
        assert result["status"] == WorkflowStatus.FAILED.value
        assert result["error"] is not None
    
    @pytest.mark.asyncio
    async def test_workflow_metrics(self, workflow_config):
        """Test workflow metrics collection."""
        engine = WorkflowEngine(workflow_config)
        await engine.initialize()
        
        # Create workflow instance
        instance = await engine.create_workflow("test_workflow", workflow_config)
        instance.context = {
            "test_mode": True,
            "metrics": {
                MetricType.LATENCY.value: [
                    {"value": 100, "timestamp": 1234567890}
                ]
            }
        }
        
        # Execute workflow
        result = await engine.execute_workflow(instance)
        
        assert result is not None
        assert result["status"] == WorkflowStatus.COMPLETED.value
        assert "result" in result
        assert "steps" in result["result"]