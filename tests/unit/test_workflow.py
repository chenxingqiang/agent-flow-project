"""Tests for workflow functionality."""
import pytest
from unittest.mock import patch, MagicMock
from agentflow.core.workflow import WorkflowEngine
from agentflow.core.metrics import MetricType
from agentflow.core.workflow_types import WorkflowConfig

class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name, async_mode=False):
        self.name = name
        self.async_mode = async_mode
        self.executed = False
    
    async def execute(self, context):
        self.executed = True
        return {"result": f"{self.name}_result"}

class TestWorkflowExecution:
    """Test workflow execution."""
    
    @pytest.fixture
    def sequential_workflow_config(self):
        """Create sequential workflow config."""
        return {
            "COLLABORATION": {
                "MODE": "SEQUENTIAL",
                "WORKFLOW": [
                    {"name": "agent1", "async": True},
                    {"name": "agent2", "async": False}
                ]
            },
            "required_fields": ["research_topic"],
            "missing_input_error": "Empty input data",
            "validation_rules": {
                "research_topic": {
                    "type": "string",
                    "required": True
                }
            },
            "steps": []
        }
    
    @pytest.fixture
    def parallel_workflow_config(self):
        """Create parallel workflow config."""
        return {
            "COLLABORATION": {
                "MODE": "PARALLEL",
                "COMMUNICATION_PROTOCOL": {
                    "TYPE": "SEMANTIC"
                },
                "WORKFLOW": [
                    {"name": "agent1", "async": True},
                    {"name": "agent2", "async": True}
                ]
            },
            "required_fields": ["research_topic"],
            "missing_input_error": "Empty input data",
            "validation_rules": {
                "research_topic": {
                    "type": "string",
                    "required": True
                }
            },
            "steps": []
        }
    
    @pytest.fixture
    def workflow_config(self):
        """Create workflow configuration."""
        return WorkflowConfig(
            id="test-workflow",
            name="Test Workflow",
            max_iterations=5,
            timeout=3600,
            steps=[]
        )
    
    @pytest.mark.asyncio
    @patch('agentflow.core.workflow.WorkflowEngine._create_agent')
    async def test_sequential_workflow(self, mock_create_agent, sequential_workflow_config, workflow_config):
        """Test sequential workflow execution."""
        # Setup mock agents
        agent1 = MockAgent("agent1", async_mode=True)
        agent2 = MockAgent("agent2", async_mode=False)
        mock_create_agent.side_effect = [agent1, agent2]
        
        # Initial context
        initial_context = {
            "research_topic": "AI Ethics",
            "metrics": {
                MetricType.LATENCY.value: [{"value": 100, "timestamp": 1234567890}]
            }
        }
        
        # Create and execute workflow
        workflow = WorkflowEngine(sequential_workflow_config, workflow_config)
        result = await workflow.execute(initial_context)
        
        # Verify execution
        assert agent1.executed
        assert agent2.executed
        assert "agent1_result" in str(result)
        assert "agent2_result" in str(result)
    
    @pytest.mark.asyncio
    @patch('agentflow.core.workflow.WorkflowEngine._create_agent')
    async def test_parallel_workflow(self, mock_create_agent, parallel_workflow_config, workflow_config):
        """Test parallel workflow execution."""
        # Setup mock agents
        agent1 = MockAgent("agent1", async_mode=True)
        agent2 = MockAgent("agent2", async_mode=True)
        mock_create_agent.side_effect = [agent1, agent2]
        
        # Initial context with metrics
        initial_context = {
            "research_topic": "AI Ethics",
            "metrics": {
                MetricType.LATENCY.value: [{"value": 100, "timestamp": 1234567890}],
                MetricType.TOKEN_COUNT.value: [{"value": 1000, "timestamp": 1234567890}]
            }
        }
        
        # Create and execute workflow
        workflow = WorkflowEngine(parallel_workflow_config, workflow_config)
        result = await workflow.execute(initial_context)
        
        # Verify execution
        assert agent1.executed
        assert agent2.executed
        assert "agent1_result" in str(result)
        assert "agent2_result" in str(result)
    
    @pytest.mark.asyncio
    async def test_workflow_validation(self, sequential_workflow_config, workflow_config):
        """Test workflow input validation."""
        workflow = WorkflowEngine(sequential_workflow_config, workflow_config)
        
        # Test missing required field
        with pytest.raises(ValueError, match="Empty input data"):
            await workflow.execute({})
        
        # Test invalid field type
        with pytest.raises(ValueError, match="Invalid input"):
            await workflow.execute({"research_topic": 123})
    
    @pytest.mark.asyncio
    async def test_workflow_context_propagation(self, sequential_workflow_config, workflow_config):
        """Test context propagation through workflow."""
        workflow = WorkflowEngine(sequential_workflow_config, workflow_config)
        
        initial_context = {
            "research_topic": "AI Ethics",
            "additional_data": "test"
        }
        
        result = await workflow.execute(initial_context)
        assert "research_topic" in str(result)
        assert "additional_data" in str(result)
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, sequential_workflow_config, workflow_config):
        """Test workflow error handling."""
        workflow = WorkflowEngine(sequential_workflow_config, workflow_config)
        
        # Test with invalid agent configuration
        with pytest.raises(ValueError):
            await workflow.execute({"research_topic": "test"}, max_retries=0)
    
    @pytest.mark.asyncio
    async def test_workflow_metrics(self, sequential_workflow_config, workflow_config):
        """Test workflow metrics collection."""
        workflow = WorkflowEngine(sequential_workflow_config, workflow_config)
        
        context = {
            "research_topic": "AI Ethics",
            "metrics": {
                MetricType.LATENCY.value: [
                    {"value": 100, "timestamp": 1234567890}
                ]
            }
        }
        
        result = await workflow.execute(context)
        assert MetricType.LATENCY.value in str(result)