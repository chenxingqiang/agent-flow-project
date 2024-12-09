"""Tests for workflow functionality."""
import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
from agentflow.core.workflow import WorkflowEngine, WorkflowEngineError
from agentflow.core.metrics import MetricType

class MockAgent:
    """Mock agent for testing."""
    def __init__(self, name: str, fail: bool = False, async_mode: bool = False):
        self.name = name
        self.executed_contexts = []
        self.fail = fail
        self.async_mode = async_mode
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent synchronously."""
        if self.async_mode:
            raise RuntimeError("Agent is in async mode but called synchronously")
        return self._process(context)
        
    async def execute_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent asynchronously."""
        if not self.async_mode:
            raise RuntimeError("Agent is in sync mode but called asynchronously")
        return self._process(context)
    
    def _process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the context."""
        self.executed_contexts.append(context)
        if self.fail:
            raise Exception(f"Agent {self.name} failed")
        context[f'{self.name}_processed'] = True
        return context

@pytest.fixture
def base_workflow_config():
    """Base workflow configuration."""
    return {
        "execution_policies": {
            "required_fields": ["research_topic"],
            "default_status": "initialized",
            "error_handling": {
                "missing_input_error": "Empty input data",
                "missing_field_error": "Missing required fields: {}"
            },
            "steps": []
        }
    }

@pytest.fixture
def minimal_workflow_config():
    """Minimal workflow configuration without execution policies."""
    return {
        "name": "minimal_workflow",
        "description": "Minimal workflow without execution policies"
    }

@pytest.fixture
def sequential_workflow_config(base_workflow_config):
    """Configuration for sequential workflow."""
    config = base_workflow_config.copy()
    config.update({
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "WORKFLOW": [
                {"name": "agent1", "async": True},
                {"name": "agent2", "async": False}
            ]
        }
    })
    return config

@pytest.fixture
def parallel_workflow_config(base_workflow_config):
    """Configuration for parallel workflow."""
    config = base_workflow_config.copy()
    config.update({
        "COLLABORATION": {
            "MODE": "PARALLEL",
            "WORKFLOW": [
                {"name": "agent1", "async": True},
                {"name": "agent2", "async": True}
            ],
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "SEMANTIC"
            }
        }
    })
    return config

@pytest.fixture
def dynamic_routing_workflow_config(base_workflow_config):
    """Configuration for dynamic routing workflow."""
    config = base_workflow_config.copy()
    config.update({
        "COLLABORATION": {
            "MODE": "DYNAMIC_ROUTING",
            "WORKFLOW": {
                "agent1": {
                    "dependencies": [],
                    "config_path": "/path/to/agent1_config.json",
                    "async": True
                },
                "agent2": {
                    "dependencies": ["agent1_processed"],
                    "config_path": "/path/to/agent2_config.json",
                    "async": False
                }
            },
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "RPC"
            }
        }
    })
    return config

class TestWorkflowExecution:
    """Test cases for workflow execution."""
    
    @pytest.mark.asyncio
    @patch('agentflow.core.workflow.WorkflowEngine._create_agent')
    async def test_sequential_workflow(self, mock_create_agent, sequential_workflow_config):
        """Test sequential workflow execution."""
        # Setup mock agents
        agent1 = MockAgent("agent1", async_mode=True)
        agent2 = MockAgent("agent2", async_mode=False)
        mock_create_agent.side_effect = [agent1, agent2]
        
        # Initial context
        initial_context = {
            "research_topic": "AI Ethics",
            "metrics": {
                MetricType.ERROR_RATE.value: [{"value": 0.05, "timestamp": 1234567890}]
            }
        }
        
        # Create workflow engine with execute_step implementation
        class TestWorkflowEngine(WorkflowEngine):
            async def execute_step(self, step_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
                agent = self._create_agent(step_config)
                if step_config.get('async', False):
                    return await agent.execute_async(context)
                return agent.execute(context)
        
        # Create and execute workflow
        workflow = TestWorkflowEngine(sequential_workflow_config)
        result = await workflow.execute(initial_context)
        
        # Verify results
        assert result['agent1_processed'] == True
        assert result['agent2_processed'] == True
        assert len(agent1.executed_contexts) == 1
        assert len(agent2.executed_contexts) == 1
        assert agent2.executed_contexts[0]['agent1_processed'] == True

    @pytest.mark.asyncio
    @patch('agentflow.core.workflow.WorkflowEngine._create_agent')
    async def test_parallel_workflow(self, mock_create_agent, parallel_workflow_config):
        """Test parallel workflow execution."""
        # Setup mock agents
        agent1 = MockAgent("agent1", async_mode=True)
        agent2 = MockAgent("agent2", async_mode=True)
        mock_create_agent.side_effect = [agent1, agent2]

        # Initial context with metrics
        initial_context = {
            "research_topic": "AI Ethics",
            "metrics": {
                MetricType.ERROR_RATE.value: [{"value": 0.05, "timestamp": 1234567890}],
                MetricType.LATENCY.value: [{"value": 100, "timestamp": 1234567890}]
            }
        }

        # Create workflow engine with execute_step implementation
        class TestWorkflowEngine(WorkflowEngine):
            async def execute_step(self, step_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
                agent = self._create_agent(step_config)
                if step_config.get('async', False):
                    return await agent.execute_async(context)
                return agent.execute(context)
        
        # Create and execute workflow
        workflow = TestWorkflowEngine(parallel_workflow_config)
        result = await workflow.execute(initial_context)

        # Verify results
        assert result['agent1_processed'] == True
        assert result['agent2_processed'] == True
        assert len(agent1.executed_contexts) == 1
        assert len(agent2.executed_contexts) == 1

    @pytest.mark.asyncio
    @patch('agentflow.core.workflow.WorkflowEngine._create_agent')
    async def test_dynamic_routing_workflow(self, mock_create_agent, dynamic_routing_workflow_config):
        """Test dynamic routing workflow execution."""
        # Setup mock agents
        agent1 = MockAgent("agent1", async_mode=True)
        agent2 = MockAgent("agent2", async_mode=False)
        mock_create_agent.side_effect = [agent1, agent2]

        # Initial context with validation results
        initial_context = {
            "research_topic": "AI Ethics",
            "validations": [
                {"result": {"is_valid": True}},
                {"result": {"is_valid": True}}
            ]
        }

        # Create workflow engine with execute_step implementation
        class TestWorkflowEngine(WorkflowEngine):
            async def execute_step(self, step_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
                agent = self._create_agent(step_config)
                if step_config.get('async', False):
                    return await agent.execute_async(context)
                return agent.execute(context)
        
        # Create and execute workflow
        workflow = TestWorkflowEngine(dynamic_routing_workflow_config)
        result = await workflow.execute(initial_context)

        # Verify results
        assert result['agent1_processed'] == True
        assert result['agent2_processed'] == True
        assert len(agent1.executed_contexts) == 1
        assert len(agent2.executed_contexts) == 1

    @pytest.mark.asyncio
    @patch('agentflow.core.workflow.WorkflowEngine._create_agent')
    async def test_workflow_agent_failure(self, mock_create_agent, sequential_workflow_config):
        """Test workflow error handling when an agent fails."""
        # Setup mock agents with failure
        agent1 = MockAgent("agent1", async_mode=True)
        agent2 = MockAgent("agent2", fail=True, async_mode=False)
        mock_create_agent.side_effect = [agent1, agent2]

        # Initial context
        initial_context = {"research_topic": "AI Ethics"}

        # Create workflow engine with execute_step implementation
        class TestWorkflowEngine(WorkflowEngine):
            async def execute_step(self, step_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
                agent = self._create_agent(step_config)
                if step_config.get('async', False):
                    return await agent.execute_async(context)
                return agent.execute(context)
        
        # Create workflow
        workflow = TestWorkflowEngine(sequential_workflow_config)

        # Verify error handling
        with pytest.raises(WorkflowEngineError, match="Agent执行失败"):
            await workflow.execute(initial_context)

    @pytest.mark.asyncio
    async def test_workflow_invalid_mode(self, base_workflow_config):
        """Test workflow with invalid execution mode."""
        invalid_workflow_config = base_workflow_config.copy()
        invalid_workflow_config.update({
            "COLLABORATION": {
                "MODE": "INVALID_MODE"
            }
        })
    
        # Create workflow engine with execute_step implementation
        class TestWorkflowEngine(WorkflowEngine):
            async def execute_step(self, step_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
                agent = self._create_agent(step_config)
                if step_config.get('async', False):
                    return await agent.execute_async(context)
                return agent.execute(context)
    
        # Expect WorkflowEngineError to be raised during initialization
        with pytest.raises(WorkflowEngineError, match="不支持的工作流模式"):
            TestWorkflowEngine(invalid_workflow_config)

    def test_workflow_dependencies(self, base_workflow_config):
        """Test workflow dependency checking."""
        workflow_config = base_workflow_config.copy()
        workflow_config.update({
            "COLLABORATION": {
                "MODE": "DYNAMIC_ROUTING",
                "WORKFLOW": {
                    "agent1": {
                        "dependencies": []
                    },
                    "agent2": {
                        "dependencies": ["agent1_processed", "validation_passed"]
                    }
                }
            }
        })
    
        # Create workflow engine with execute_step implementation
        class TestWorkflowEngine(WorkflowEngine):
            async def execute_step(self, step_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
                agent = self._create_agent(step_config)
                if step_config.get('async', False):
                    return await agent.execute_async(context)
                return agent.execute(context)
    
        workflow = TestWorkflowEngine(workflow_config)
    
        # Test dependency checking
        context = {}
        assert not workflow._check_agent_dependencies(["test"], context)
    
        context = {"agent1_processed": True}
        assert not workflow._check_agent_dependencies(["agent1_processed", "validation_passed"], context)
    
        context = {"agent1_processed": True, "validation_passed": True}
        assert workflow._check_agent_dependencies(["agent1_processed", "validation_passed"], context)

    def test_communication_protocols(self, base_workflow_config):
        """Test different communication protocols."""
        # Create workflow engine with execute_step implementation
        class TestWorkflowEngine(WorkflowEngine):
            async def execute_step(self, step_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
                agent = self._create_agent(step_config)
                if step_config.get('async', False):
                    return await agent.execute_async(context)
                return agent.execute(context)
    
        workflow = TestWorkflowEngine(base_workflow_config)
    
        # Test semantic message merging
        semantic_results = [
            {"key1": "value1", "shared": "old"},
            {"key2": "value2", "shared": "new"}
        ]
        merged = workflow._semantic_message_merge(semantic_results)
        assert merged == {"key1": "value1", "key2": "value2", "shared": "new"}

    @pytest.mark.asyncio
    async def test_agent_cache(self, base_workflow_config):
        """Test agent caching mechanism."""
        class TestWorkflowEngine(WorkflowEngine):
            def _create_agent(self, agent_config: Dict[str, Any]) -> Any:
                """Override _create_agent to return a simple object for testing"""
                agent_name = agent_config.get('name')
                
                # Check cache first
                if agent_name and agent_name in self._agent_cache:
                    return self._agent_cache[agent_name]
                
                # Create a simple object to represent an agent
                agent = type('TestAgent', (), {'name': agent_name})()
                
                # Cache the agent
                if agent_name:
                    self._agent_cache[agent_name] = agent
                
                return agent
            
            async def execute_step(self, step_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
                agent = self._create_agent(step_config)
                return context

        # Create workflow engine
        workflow = TestWorkflowEngine(base_workflow_config)
        
        # First, verify that no agents are in the cache initially
        assert len(workflow._agent_cache) == 0
        
        # Execute the first workflow step
        context1 = await workflow.execute_step(
            {"name": "agent1", "config_path": "/path/to/agent1.json"}, 
            {"research_topic": "test1"}
        )
        
        # Verify that an agent is now in the cache
        assert len(workflow._agent_cache) == 1
        first_agent = workflow._agent_cache.get('agent1')
        assert first_agent is not None
        
        # Execute another step with the same agent name
        context2 = await workflow.execute_step(
            {"name": "agent1", "config_path": "/path/to/agent1.json"}, 
            {"research_topic": "test2"}
        )
        
        # Verify that the agent is still the same and cache size hasn't changed
        assert len(workflow._agent_cache) == 1
        second_agent = workflow._agent_cache.get('agent1')
        assert second_agent is first_agent

    @pytest.mark.asyncio
    async def test_workflow_without_execution_policies(self, minimal_workflow_config):
        """Test workflow initialization without execution policies."""
        workflow = WorkflowEngine(minimal_workflow_config)
        assert workflow.required_fields == []
        assert workflow.error_handling == {}
        assert workflow.default_status is None
        assert workflow.steps == []

if __name__ == "__main__":
    pytest.main([__file__])