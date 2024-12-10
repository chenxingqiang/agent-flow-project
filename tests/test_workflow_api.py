import pytest
import requests
import subprocess
import time
import os
import sys
import signal
import logging
import socket
import atexit
from requests.exceptions import HTTPError, ConnectionError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Find an available port
def find_free_port():
    """Find a free port for testing"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

class ServerManager:
    """Manage server startup and teardown for tests"""
    def __init__(self):
        self.port = find_free_port()
        self.process = None
        self.base_url = f"http://localhost:{self.port}"
        
    def start_server(self):
        """Start the workflow server in a separate process"""
        import subprocess
        import sys
        import time
        import os
        import traceback

        # Construct the command to run the server
        server_script = os.path.join(os.path.dirname(__file__), '..', 'agentflow', 'api', 'workflow_server.py')

        # Start the server process
        self.process = subprocess.Popen(
            [sys.executable, server_script, str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for the server to start
        max_wait_time = 20  # Increased wait time
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                # Try to connect to the server
                response = requests.get(f"{self.base_url}/docs", timeout=2)
                if response.status_code == 200:
                    logger.info(f"Server started successfully on port {self.port}")
                    return
            except (requests.ConnectionError, requests.Timeout) as e:
                # Print out any connection errors
                logger.warning(f"Connection attempt failed: {e}")
                time.sleep(1)

        # Print out process output for debugging
        stdout, stderr = self.process.communicate()
        logger.error(f"Server stdout: {stdout}")
        logger.error(f"Server stderr: {stderr}")

        # If we get here, server didn't start
        raise RuntimeError(f"Server failed to start within {max_wait_time} seconds")
    
    def stop_server(self):
        """Stop the server process"""
        if self.process:
            self.process.terminate()
            try:
                stdout, stderr = self.process.communicate(timeout=5)
                logger.debug(f"Server stdout: {stdout}")
                logger.debug(f"Server stderr: {stderr}")
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

@pytest.fixture(scope="module")
def server():
    """Pytest fixture to manage server lifecycle"""
    server_manager = ServerManager()
    try:
        server_manager.start_server()
        yield server_manager
    finally:
        server_manager.stop_server()

# Global variable to store the base URL
BASE_URL = None

def pytest_configure(config):
    """Configure the test session"""
    global BASE_URL
    # This will be set by the server fixture
    BASE_URL = f"http://localhost:{server().port}" if hasattr(server(), 'port') else None

def test_sync_workflow_execution(server):
    """Test synchronous workflow execution"""
    url = f"{server.base_url}/workflow/execute"

    workflow_config = {
        "workflow": {
            "WORKFLOW": [
                {
                    "step": 1,
                    "type": "research",
                    "name": "Research Step",
                    "description": "Perform research on the given topic",
                    "input": ["research_topic", "deadline", "academic_level"],
                    "output": {"type": "research"},
                    "agent_config": {}
                },
                {
                    "step": 2,
                    "type": "document",
                    "name": "Document Generation Step",
                    "description": "Generate document from research findings",
                    "input": ["WORKFLOW.1"],
                    "output": {"type": "document"},
                    "agent_config": {}
                }
            ]
        },
        "config": {
            "workflow_name": "research_workflow",
            "max_retries": 3,
            "retry_delay": 1.0,
            "max_concurrent_steps": 2,
            "model": {
                "provider": "openai",
                "name": "gpt-3.5-turbo"
            },
            "type": "research",
            "system_prompt": "You are a research assistant",
            "workflow": {
                "name": "research_workflow",
                "description": "Research workflow execution",
                "agents": [],
                "execution_policies": {
                    "required_inputs": [],
                    "default_status": "initialized",
                    "error_handling": {},
                    "steps": [
                        {
                            "step": 1,
                            "type": "research",
                            "name": "Step 1",
                            "description": "Perform research on the given topic",
                            "input_type": "dict",
                            "output_type": "dict",
                            "agents": [{}],
                            "input": ["research_topic", "deadline", "academic_level"],
                            "output": "step_1_output"
                        },
                        {
                            "step": 2,
                            "type": "document",
                            "name": "Step 2",
                            "description": "Generate document from research findings",
                            "input_type": "dict",
                            "output_type": "dict",
                            "agents": [{}],
                            "input": ["WORKFLOW.1"],
                            "output": "step_2_output"
                        }
                    ]
                }
            },
            "step_1_config": {
                "timeout": 60,
                "max_retries": 3,
                "additional_inputs": {},
                "agent_config": {}
            },
            "step_2_config": {
                "timeout": 60,
                "max_retries": 3,
                "additional_inputs": {},
                "agent_config": {}
            },
            "step_config": {
                "timeout": 60,
                "max_retries": 3
            }
        },
        "input_data": {
            "research_topic": "API Testing in Distributed Systems",
            "deadline": "2024-05-15",
            "academic_level": "Master"
        },
        "config": {
            "workflow_name": "research_workflow"
        }
    }

    # Add logging to capture the full request and response
    logger.setLevel(logging.DEBUG)

    try:
        response = requests.post(url, json=workflow_config)
        logger.debug(f"Request Payload: {workflow_config}")
        logger.debug(f"Sync Response Status: {response.status_code}")
        logger.debug(f"Sync Response Content: {response.text}")

        # If the response is not 200, print out the full error details
        if response.status_code != 200:
            logger.error(f"Full error response: {response.text}")
            raise requests.exceptions.HTTPError(f"Request failed with status {response.status_code}: {response.text}")

        result = response.json()
        assert isinstance(result, dict)
        assert 'output' in result
        assert result['output']['type'] in ['research', 'document']

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise

def test_async_workflow_execution(server):
    """Test asynchronous workflow execution"""
    execute_url = f"{server.base_url}/workflow/execute_async"

    workflow_config = {
        "workflow": {
            "WORKFLOW": [
                {
                    "step": 1,
                    "type": "research",
                    "name": "Research Step",
                    "description": "Perform research on the given topic",
                    "input": ["research_topic", "deadline", "academic_level"],
                    "output": {"type": "research"},
                    "agent_config": {}
                },
                {
                    "step": 2,
                    "type": "document",
                    "name": "Document Generation Step",
                    "description": "Generate document from research findings",
                    "input": ["WORKFLOW.1"],
                    "output": {"type": "document"},
                    "agent_config": {}
                }
            ]
        },
        "config": {
            "workflow_name": "research_workflow",
            "max_retries": 3,
            "retry_delay": 1.0,
            "max_concurrent_steps": 2,
            "model": {
                "provider": "openai",
                "name": "gpt-3.5-turbo"
            },
            "type": "research",
            "system_prompt": "You are a research assistant",
            "step_1_config": {
                "timeout": 60,
                "max_retries": 3,
                "additional_inputs": {},
                "agent_config": {}
            },
            "step_2_config": {
                "timeout": 60,
                "max_retries": 3,
                "additional_inputs": {},
                "agent_config": {}
            },
            "step_config": {
                "timeout": 60,
                "max_retries": 3
            }
        },
        "input_data": {
            "research_topic": "Async API Testing",
            "deadline": "2024-06-15",
            "academic_level": "PhD"
        },
        "config": {
            "workflow_name": "research_workflow"
        }
    }

    try:
        # Initiate async workflow
        async_response = requests.post(execute_url, json=workflow_config)
        logger.debug(f"Async Response Status: {async_response.status_code}")
        logger.debug(f"Async Response Content: {async_response.text}")

        # If the response is not 200, print out the full error details
        if async_response.status_code != 200:
            logger.error(f"Full async error response: {async_response.text}")
            raise requests.exceptions.HTTPError(f"Async request failed with status {async_response.status_code}: {async_response.text}")

        async_result = async_response.json()
        assert 'workflow_id' in async_result
        workflow_id = async_result['workflow_id']

        # Poll for workflow completion
        status_url = f"{server.base_url}/workflow/status/{workflow_id}"
        max_retries = 10
        retry_delay = 1

        for _ in range(max_retries):
            status_response = requests.get(status_url)
            if status_response.status_code != 200:
                logger.error(f"Status check failed: {status_response.text}")
                raise requests.exceptions.HTTPError(f"Status check failed with status {status_response.status_code}")
            
            status_result = status_response.json()
            logger.debug(f"Workflow status: {status_result}")
            
            if status_result.get('status') == 'completed':
                break
            
            time.sleep(retry_delay)
        else:
            raise TimeoutError("Workflow did not complete within expected time")

    except requests.exceptions.RequestException as e:
        logger.error(f"Async request failed: {str(e)}")
        raise

def test_invalid_workflow(server):
    """Test handling of invalid workflow configuration"""
    url = f"{server.base_url}/workflow/execute"

    invalid_workflow_config = {
        "workflow": {
            "workflow_steps": []  # Empty workflow steps
        },
        "input_data": {}
    }

    try:
        response = requests.post(url, json=invalid_workflow_config)
        logger.debug(f"Invalid Workflow Response Status: {response.status_code}")
        logger.debug(f"Invalid Workflow Response Content: {response.text}")

        # Verify error response
        assert response.status_code == 422  # Validation error status code
        error_data = response.json()
        assert "detail" in error_data
        assert any("No workflow steps found" in str(detail) for detail in error_data["detail"])

    except Exception as e:
        logger.error(f"Invalid workflow request failed: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main([__file__])
