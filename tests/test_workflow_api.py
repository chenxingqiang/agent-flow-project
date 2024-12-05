import pytest
import requests

BASE_URL = "http://localhost:8000"

def test_sync_workflow_execution():
    """Test synchronous workflow execution"""
    url = f"{BASE_URL}/workflow/execute"
    
    workflow_config = {
        "workflow": {
            "WORKFLOW": [
                {
                    "input": ["research_topic", "deadline", "academic_level"],
                    "output": {"type": "research"},
                    "step": 1
                },
                {
                    "input": ["WORKFLOW.1"],
                    "output": {"type": "document"},
                    "step": 2
                }
            ]
        },
        "input_data": {
            "research_topic": "API Testing in Distributed Systems",
            "deadline": "2024-05-15",
            "academic_level": "Master"
        }
    }

    response = requests.post(url, json=workflow_config)
    
    assert response.status_code == 200, f"Request failed with status {response.status_code}"
    
    result = response.json()
    assert "result" in result, "No result in response"
    assert "status" in result and result["status"] == "success"
    assert "execution_time" in result

def test_async_workflow_execution():
    """Test asynchronous workflow execution"""
    execute_url = f"{BASE_URL}/workflow/execute_async"
    
    workflow_config = {
        "workflow": {
            "WORKFLOW": [
                {
                    "input": ["research_topic", "deadline", "academic_level"],
                    "output": {"type": "research"},
                    "step": 1
                },
                {
                    "input": ["WORKFLOW.1"],
                    "output": {"type": "document"},
                    "step": 2
                }
            ]
        },
        "input_data": {
            "research_topic": "Async API Testing",
            "deadline": "2024-06-15",
            "academic_level": "PhD"
        }
    }

    # Initiate async workflow
    async_response = requests.post(execute_url, json=workflow_config)
    
    assert async_response.status_code == 200, f"Async request failed with status {async_response.status_code}"
    
    async_result = async_response.json()
    assert "result_ref" in async_result, "No result reference in async response"
    assert "status" in async_result and async_result["status"] == "async_initiated"

    # Retrieve result (with timeout)
    result_url = f"{BASE_URL}/workflow/result/{async_result['result_ref']}"
    result_response = requests.get(result_url)
    
    assert result_response.status_code in [200, 404], f"Result retrieval failed with status {result_response.status_code}"

def test_invalid_workflow():
    """Test handling of invalid workflow configuration"""
    url = f"{BASE_URL}/workflow/execute"
    
    invalid_workflow_config = {
        "workflow": {},  # Invalid workflow
        "input_data": {}
    }

    response = requests.post(url, json=invalid_workflow_config)
    
    assert response.status_code == 500, "Expected server error for invalid workflow"

if __name__ == "__main__":
    pytest.main([__file__])
