import os
import sys
import subprocess

def run_diagnostic_command(command):
    """Run a shell command and print its output"""
    print(f"\n=== Running Command: {command} ===")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return Code:", result.returncode)
    except Exception as e:
        print(f"Error running command {command}: {e}")

# Run diagnostic commands
print("\n" + "="*50)
print("SYSTEM DIAGNOSTIC INFORMATION")
print("="*50)

# Python and environment information
print("\nPython Information:")
print("Executable:", sys.executable)
print("Version:", sys.version)
print("Platform:", sys.platform)
print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)

# Run diagnostic commands
diagnostic_commands = [
    "which python3",
    "which pytest",
    "python3 --version",
    "pip list",
    "pwd",
    "ls /Users/xingqiangchen/TASK/APOS"
]

for cmd in diagnostic_commands:
    run_diagnostic_command(cmd)

print("\n" + "="*50)
print("BEGINNING ACTUAL TEST IMPORTS")
print("="*50)

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from agentflow.core.workflow_executor import (
    WorkflowExecutor,
    WorkflowManager,
    NodeState,
    WorkflowExecutionError
)
from agentflow.core.config_manager import (
    WorkflowConfig, 
    AgentConfig, 
    ModelConfig, 
    ConnectionConfig
)
from agentflow.api.monitor_service import LogEntry
from agentflow.core.processors.transformers import (
    FilterProcessor,
    TransformProcessor,
    AggregateProcessor
)

# Constants
ASYNC_TEST_TIMEOUT = 5  # seconds

@pytest.fixture(scope="function")
def event_loop():
    """Create an event loop for each test function"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def sample_workflow_config():
    """Create a sample workflow configuration for testing"""
    return {
        "id": "test-workflow",
        "name": "Test Workflow",
        "description": "A test workflow",
        "agents": [
            {
                "id": "agent1",
                "name": "Agent 1",
                "type": "test",
                "description": "Test agent 1",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai"
                },
                "system_prompt": "You are a test agent",
                "config": {}  
            },
            {
                "id": "agent2", 
                "name": "Agent 2",
                "type": "test",
                "description": "Test agent 2",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai"
                },
                "system_prompt": "You are another test agent",
                "config": {}  
            }
        ],
        "processors": [
            {
                "id": "processor1",
                "name": "Test Processor",
                "type": "processor",
                "description": "Test processor with process_data method",
                "processor": "tests.core.test_workflow_executor.TestProcessor",
                "config": {}
            }
        ],
        "connections": [
            {
                "source_id": "agent1",
                "target_id": "agent2",
                "source_port": "output",
                "target_port": "input"
            },
            {
                "source_id": "agent1",
                "target_id": "processor1",
                "source_port": "output",
                "target_port": "input"
            }
        ]
    }

@pytest.fixture
def mock_agent():
    """Create a mock agent with async methods"""
    mock = AsyncMock()
    mock.process = AsyncMock(return_value={"result": "test-output"})
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def test_agent():
    """Fixture for test agent"""
    def _create_agent(config):
        agent = TestAgent(config)
        return agent
    return _create_agent

class TestAgent:
    """Test agent class"""
    def __init__(self, config):
        self.config = config
        self.token_count = 0
        self.last_latency = 0
        self.memory_usage = 0
        self.history = []

    async def process(self, input_data):
        """Process input data"""
        if isinstance(input_data, dict) and "message" in input_data:
            return await self.process_message(input_data["message"])
        return {"processed": True, "input": input_data}

    async def process_message(self, message):
        """Process a message"""
        self.history.append(message)
        return {"response": f"Processed: {message}"}

    async def initialize(self):
        """Initialize agent"""
        pass

    async def cleanup(self):
        """Cleanup agent"""
        self.history = []

@pytest.fixture
def test_processor():
    """Fixture for test processor"""
    def _create_processor(config):
        processor = TestProcessor(config)
        return processor
    return _create_processor

class TestProcessor:
    """Test processor for workflow executor tests"""
    def __init__(self, config):
        self.config = config
        self.processed_data = []

    async def process_data(self, input_data):
        # Ensure the input is a dictionary
        if not isinstance(input_data, dict):
            input_data = {"value": input_data}

        # Simulate processing
        processed_data = input_data.copy()
        processed_data['result'] = f"Processed {input_data}"
        return processed_data

    async def cleanup(self):
        """Cleanup processor"""
        self.processed_data = []

@pytest.mark.asyncio
async def test_workflow_executor_initialization(sample_workflow_config):
    """Test workflow executor initialization"""
    executor = WorkflowExecutor(sample_workflow_config)
    
    assert len(executor.nodes) == 3
    assert len(executor.connections) == 2
    
    # Check initial node states
    for node in executor.nodes.values():
        assert node.state == NodeState.PENDING

@pytest.mark.asyncio
async def test_workflow_executor_execution(sample_workflow_config):
    """Test workflow execution with mocked agents"""
    print("\n" + "="*50)
    print("Starting test_workflow_executor_execution")
    print("="*50)
    
    # Create executor
    executor = WorkflowExecutor(sample_workflow_config)
    
    # Create mock agents with controlled behavior
    mock_agent_1 = AsyncMock()
    mock_agent_1.process = AsyncMock(return_value={"result": "agent-1-output"})
    mock_agent_1.initialize = AsyncMock()
    mock_agent_1.cleanup = AsyncMock()
    mock_agent_1.token_count = 0
    mock_agent_1.last_latency = 0
    mock_agent_1.memory_usage = 0
    
    mock_agent_2 = AsyncMock()
    mock_agent_2.process = AsyncMock(return_value={"result": "agent-2-output"})
    mock_agent_2.initialize = AsyncMock()
    mock_agent_2.cleanup = AsyncMock()
    mock_agent_2.token_count = 0
    mock_agent_2.last_latency = 0
    mock_agent_2.memory_usage = 0
    
    # Set mock agents
    executor.nodes["agent1"].agent = mock_agent_1
    executor.nodes["agent2"].agent = mock_agent_2
    
    try:
        # Execute workflow
        input_data = {"input": "test"}
        result = await asyncio.wait_for(executor.execute(input_data), timeout=ASYNC_TEST_TIMEOUT)
        
        # Verify the workflow executed correctly
        assert result["status"] == "completed", "Workflow should complete successfully"
        assert mock_agent_1.process.called, "Agent 1 process method should be called"
        assert mock_agent_2.process.called, "Agent 2 process method should be called"
        assert mock_agent_1.process.call_count == 1, "Agent 1 should be called exactly once"
        assert mock_agent_2.process.call_count == 1, "Agent 2 should be called once"
        
        # Check node states
        for node in executor.nodes.values():
            assert node.state == NodeState.COMPLETED, f"Node {node.id} should be completed"
            
        # Verify cleanup was called
        assert mock_agent_1.cleanup.called, "Agent 1 cleanup should be called"
        assert mock_agent_2.cleanup.called, "Agent 2 cleanup should be called"
            
    except asyncio.TimeoutError:
        print("Test timed out!")
        raise
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise
    finally:
        # Cleanup
        for node in executor.nodes.values():
            if node.agent and hasattr(node.agent, 'cleanup'):
                await node.agent.cleanup()

@pytest.mark.asyncio
async def test_workflow_input_routing(sample_workflow_config):
    """Test workflow input routing between nodes"""
    executor = WorkflowExecutor(sample_workflow_config)
    
    # Create mock agents
    mock_agent_1 = AsyncMock()
    mock_agent_1.process = AsyncMock(return_value={"result": "agent-1-output"})
    mock_agent_1.initialize = AsyncMock()
    mock_agent_1.cleanup = AsyncMock()
    mock_agent_1.token_count = 0
    mock_agent_1.last_latency = 0
    mock_agent_1.memory_usage = 0
    
    mock_agent_2 = AsyncMock()
    mock_agent_2.process = AsyncMock(return_value={"result": "agent-2-output"})
    mock_agent_2.initialize = AsyncMock()
    mock_agent_2.cleanup = AsyncMock()
    mock_agent_2.token_count = 0
    mock_agent_2.last_latency = 0
    mock_agent_2.memory_usage = 0
    
    # Set up mock agents in nodes
    executor.nodes["agent1"].agent = mock_agent_1
    executor.nodes["agent2"].agent = mock_agent_2
    
    # Execute workflow
    input_data = {"input": "test"}
    result = await asyncio.wait_for(executor.execute(input_data), timeout=ASYNC_TEST_TIMEOUT)
    
    # Verify workflow completed
    assert result["status"] == "completed"
    assert result["nodes"]["agent1"] == "completed"
    assert result["nodes"]["agent2"] == "completed"
    
    # Verify agent calls
    mock_agent_1.process.assert_called_once_with(input_data)
    mock_agent_2.process.assert_called_once()
    
    # Verify output routing
    agent1_output = await executor.nodes["agent1"].output_queue.get()
    assert agent1_output == {"result": "agent-1-output"}
    
    agent2_output = await executor.nodes["agent2"].output_queue.get()
    assert agent2_output == {"result": "agent-2-output"}

@pytest.mark.asyncio
async def test_workflow_error_handling(sample_workflow_config):
    """Test workflow error handling"""
    executor = WorkflowExecutor(sample_workflow_config)
    
    # Create mock agent that raises an error
    mock_error_agent = AsyncMock()
    mock_error_agent.process = AsyncMock(side_effect=ValueError("Test error"))
    mock_error_agent.initialize = AsyncMock()
    mock_error_agent.cleanup = AsyncMock()
    mock_error_agent.token_count = 0
    mock_error_agent.last_latency = 0
    mock_error_agent.memory_usage = 0
    
    # Set mock agent
    executor.nodes["agent1"].agent = mock_error_agent
    
    # Execute workflow
    with pytest.raises(WorkflowExecutionError) as excinfo:
        await asyncio.wait_for(executor.execute({"input": "test"}), timeout=ASYNC_TEST_TIMEOUT)
    
    # Verify error details
    error = excinfo.value
    assert "Test error" in str(error)
    assert isinstance(error.details, dict)
    assert "node_config" in error.details
    assert "input_data" in error.details
    assert "error" in error.details
    
    # Verify node states
    assert executor.nodes["agent1"].state == NodeState.ERROR
    assert executor.nodes["agent2"].state == NodeState.PENDING
    
    # Verify cleanup was called
    assert mock_error_agent.cleanup.called, "Agent cleanup should be called even on error"

@pytest.mark.asyncio
async def test_workflow_manager_basic_operations(sample_workflow_config):
    """Test basic workflow manager operations"""
    manager = WorkflowManager()
    
    # Add workflow
    workflow_id = await manager.add_workflow(sample_workflow_config)
    assert workflow_id in manager.active_workflows
    
    # Create test agents
    executor = manager.active_workflows[workflow_id]
    test_agent_1 = TestAgent(executor.nodes["agent1"].config)
    test_agent_2 = TestAgent(executor.nodes["agent2"].config)
    executor.nodes["agent1"].agent = test_agent_1
    executor.nodes["agent2"].agent = test_agent_2
    
    # Execute workflow with input
    input_data = {"input": "test"}
    result = await asyncio.wait_for(
        manager.execute_workflow(workflow_id, input_data),
        timeout=ASYNC_TEST_TIMEOUT
    )
    
    # Verify execution result
    assert result is not None
    assert result["status"] == "completed"
    assert all(node == "completed" for node in result["nodes"].values())
    
    # Get status
    status = await manager.get_workflow_status(workflow_id)
    assert status is not None
    assert "agent1" in status
    assert "agent2" in status
    print("\nStatus values:", {node_id: node["status"] for node_id, node in status.items()})
    assert all(node["status"] == "completed" for node in status.values())
    
    # Verify metrics were updated
    assert status["agent1"]["metrics"]["tokens"] > 0
    assert status["agent1"]["metrics"]["latency"] > 0
    assert status["agent1"]["metrics"]["memory"] > 0
    assert status["agent2"]["metrics"]["tokens"] > 0
    assert status["agent2"]["metrics"]["latency"] > 0
    assert status["agent2"]["metrics"]["memory"] > 0
    
    # Stop workflow
    await manager.stop_workflow(workflow_id)
    with pytest.raises(ValueError):
        await manager.get_workflow_status(workflow_id)

@pytest.mark.asyncio
async def test_workflow_executor_process_data_support(sample_workflow_config):
    """Test workflow executor support for process_data method"""
    # Modify workflow config to use test processor
    sample_workflow_config['processors'] = [
        {
            "id": "test_processor",
            "type": "processor",
            "processor": TestProcessor,
            "config": {}
        }
    ]
    sample_workflow_config['connections'] = [
        {
            "source_id": "test_processor",
            "target_id": "agent1",
            "source_port": "output",
            "target_port": "input"
        }
    ]
    
    # Create executor
    executor = WorkflowExecutor(sample_workflow_config)
    
    # Create mock agents
    mock_agent = AsyncMock()
    mock_agent.process = AsyncMock(return_value={"result": "processed"})
    
    # Set mock agents and processors
    test_processor = TestProcessor({})
    executor.nodes["test_processor"].processor = test_processor
    executor.nodes["agent1"].agent = mock_agent
    
    # Execute workflow with different input types
    # Test single dict input
    input_data = {"value": 100}
    result = await executor.execute(input_data)
    
    assert mock_agent.process.called
    assert mock_agent.process.call_args.args[0]['result'] == f"Processed {input_data}"
    
    # Test list input
    list_input = [{"value": 1}, {"value": 2}]
    result = await executor.execute(list_input)
    
    assert mock_agent.process.called
    assert mock_agent.process.call_args.args[0]['result'] == f"Processed {list_input}"

@pytest.mark.asyncio
async def test_workflow_executor_processor_methods(sample_workflow_config):
    """Test workflow executor with different processor methods"""
    # Create processors with different method implementations
    class ProcessMethodProcessor:
        def __init__(self, config):
            self.config = config
        
        async def process(self, input_data):
            """Traditional process method"""
            return {"processed_by": "process"}
    
    class ProcessDataMethodProcessor:
        def __init__(self, config):
            self.config = config
        
        async def process_data(self, input_data):
            """New process_data method"""
            return {"processed_by": "process_data"}
    
    # Modify workflow config to use different processors
    sample_workflow_config['processors'] = [
        {
            "id": "process_method_processor",
            "type": "processor",
            "processor": ProcessMethodProcessor,
            "config": {}
        },
        {
            "id": "process_data_method_processor",
            "type": "processor",
            "processor": ProcessDataMethodProcessor,
            "config": {}
        }
    ]
    sample_workflow_config['connections'] = [
        {
            "source_id": "process_method_processor",
            "target_id": "process_data_method_processor",
            "source_port": "output",
            "target_port": "input"
        }
    ]
    
    # Create executor
    executor = WorkflowExecutor(sample_workflow_config)
    
    # Execute workflow
    input_data = {"value": 100}
    result = await executor.execute(input_data)
    
    # Verify nodes completed successfully
    assert result["status"] == "completed"
    assert result["nodes"]["process_method_processor"] == "completed"
    assert result["nodes"]["process_data_method_processor"] == "completed"
    
    # Verify outputs through output queues
    process_method_output = await executor.nodes["process_method_processor"].output_queue.get()
    process_data_output = await executor.nodes["process_data_method_processor"].output_queue.get()
    
    assert process_method_output["processed_by"] == "process"
    assert process_data_output["processed_by"] == "process_data"

@pytest.mark.asyncio
async def test_workflow_executor_error_handling_process_data(sample_workflow_config):
    """Test workflow executor error handling with process_data method"""
    class ErrorProcessor:
        def __init__(self, config):
            self.config = config
        
        async def process_data(self, input_data):
            """Raise an error during processing"""
            raise ValueError("Processing error in process_data")
    
    # Modify workflow config to use error processor
    sample_workflow_config['processors'] = [
        {
            "id": "error_processor",
            "type": "processor",
            "processor": ErrorProcessor,
            "config": {}
        }
    ]
    sample_workflow_config['connections'] = []
    
    # Create executor
    executor = WorkflowExecutor(sample_workflow_config)
    
    # Execute workflow and expect error
    with pytest.raises(WorkflowExecutionError) as excinfo:
        await executor.execute({"value": 100})
    
    # Verify error details
    error = excinfo.value
    assert "Processing error in process_data" in str(error)
    assert isinstance(error.details, dict)
    assert "node_config" in error.details
    assert "input_data" in error.details
    assert "error" in error.details
    
    # Verify node state
    assert executor.nodes["error_processor"].state == NodeState.ERROR

@pytest.mark.asyncio
async def test_processor_cleanup(sample_workflow_config):
    """Test processor cleanup is called after execution"""
    cleanup_called = False
    
    class CleanupProcessor:
        def __init__(self, config):
            self.config = config
        
        async def process_data(self, input_data):
            return {"processed": True}
            
        async def cleanup(self):
            nonlocal cleanup_called
            cleanup_called = True
    
    # Modify workflow config to use cleanup processor
    sample_workflow_config['processors'] = [
        {
            "id": "cleanup_processor",
            "type": "processor",
            "processor": CleanupProcessor,
            "config": {}
        }
    ]
    sample_workflow_config['connections'] = []
    
    # Create and execute workflow
    executor = WorkflowExecutor(sample_workflow_config)
    result = await executor.execute({"value": 100})
    
    # Verify cleanup was called
    assert cleanup_called, "Processor cleanup method should be called"
    assert result["status"] == "completed"
    assert result["nodes"]["cleanup_processor"] == "completed"

if __name__ == "__main__":
    pytest.main([__file__])
