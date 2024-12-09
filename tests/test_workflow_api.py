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
        max_wait_time = 10  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Try to connect to the server
                response = requests.get(f"{self.base_url}/docs", timeout=1)
                if response.status_code == 200:
                    logger.info(f"Server started successfully on port {self.port}")
                    return
            except (requests.ConnectionError, requests.Timeout):
                # Server not ready yet, wait a bit
                time.sleep(0.5)
        
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
        "workflow_steps": [
            {
                "input": ["research_topic", "deadline", "academic_level"],
                "output": "research",
                "step": 1,
                "agent_config": {
                    "model": {
                        "provider": "openai",
                        "name": "gpt-3.5-turbo"
                    },
                    "type": "research_agent",
                    "system_prompt": "You are a research assistant",
                    "id": "research_step"
                }
            },
            {
                "input": ["research_step_output"],
                "output": "document",
                "step": 2,
                "agent_config": {
                    "model": {
                        "provider": "openai",
                        "name": "gpt-3.5-turbo"
                    },
                    "type": "document_agent",
                    "system_prompt": "You are a document generation assistant",
                    "id": "document_step"
                }
            }
        ],
        "input_data": {
            "research_topic": "API Testing in Distributed Systems",
            "deadline": "2024-05-15",
            "academic_level": "Master"
        }
    }

    try:
        response = requests.post(url, json=workflow_config)
        logger.debug(f"Sync Response Status: {response.status_code}")
        logger.debug(f"Sync Response Content: {response.text}")

        # If the response is not 200, print out the full error details
        if response.status_code != 200:
            logger.error(f"Full error response: {response.text}")
            raise HTTPError(f"Request failed with status {response.status_code}: {response.text}")

        # Validate response
        result = response.json()
        assert 'status' in result and result['status'] == 'success'
        assert 'result' in result
        assert 'execution_time' in result

    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise

def test_async_workflow_execution(server):
    """Test asynchronous workflow execution"""
    execute_url = f"{server.base_url}/workflow/execute_async"

    workflow_config = {
        "workflow_steps": [
            {
                "input": ["research_topic", "deadline", "academic_level"],
                "output": "research",
                "step": 1,
                "agent_config": {
                    "model": {
                        "provider": "openai",
                        "name": "gpt-3.5-turbo"
                    },
                    "type": "research_agent",
                    "system_prompt": "You are a research assistant",
                    "id": "research_step"
                }
            },
            {
                "input": ["research_step_output"],
                "output": "document",
                "step": 2,
                "agent_config": {
                    "model": {
                        "provider": "openai",
                        "name": "gpt-3.5-turbo"
                    },
                    "type": "document_agent",
                    "system_prompt": "You are a document generation assistant",
                    "id": "document_step"
                }
            }
        ],
        "input_data": {
            "research_topic": "Async API Testing",
            "deadline": "2024-06-15",
            "academic_level": "PhD"
        }
    }

    # Initiate async workflow
    try:
        async_response = requests.post(execute_url, json=workflow_config)
        logger.debug(f"Async Response Status: {async_response.status_code}")
        logger.debug(f"Async Response Content: {async_response.text}")

        # If the response is not 200, print out the full error details
        if async_response.status_code != 200:
            logger.error(f"Full async error response: {async_response.text}")
            raise HTTPError(f"Async request failed with status {async_response.status_code}: {async_response.text}")

        # Validate response
        result = async_response.json()
        assert 'status' in result and result['status'] == 'success'
        assert 'result_ref' in result
        assert 'workflow_id' in result
        assert 'execution_time' in result

        # Retrieve workflow result
        result_url = f"{server.base_url}/workflow/result/{result['result_ref']}"
        result_response = requests.get(result_url)
        
        assert result_response.status_code == 200, f"Result retrieval failed with status {result_response.status_code}"
        result_data = result_response.json()
        
        assert 'status' in result_data and result_data['status'] == 'success'
        assert 'result' in result_data

    except Exception as e:
        logger.error(f"Async request failed: {str(e)}")
        raise

def test_invalid_workflow(server):
    """Test handling of invalid workflow configuration"""
    url = f"{server.base_url}/workflow/execute"

    invalid_workflow_config = {
        "workflow": {},  # Invalid workflow
        "input_data": {}
    }

    try:
        response = requests.post(url, json=invalid_workflow_config)
        logger.debug(f"Invalid Workflow Response Status: {response.status_code}")
        logger.debug(f"Invalid Workflow Response Content: {response.text}")

        # If the response is not 500, print out the full error details
        if response.status_code != 500:
            logger.error(f"Invalid workflow error response: {response.text}")
            raise HTTPError(f"Invalid workflow request failed with status {response.status_code}: {response.text}")

        # Validate error response
        result = response.json()
        assert 'status' in result and result['status'] == 'error'
        assert 'detail' in result

    except Exception as e:
        logger.error(f"Invalid workflow request failed: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main([__file__])
