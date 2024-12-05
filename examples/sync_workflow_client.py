import requests
import json

def execute_workflow():
    # API endpoint
    url = "http://localhost:8000/workflow/execute"

    # Workflow configuration
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
            "research_topic": "Advanced Machine Learning Techniques in Distributed Systems",
            "deadline": "2024-06-30",
            "academic_level": "PhD"
        },
        "config": {
            "logging_level": "INFO",
            "max_iterations": 10
        }
    }

    try:
        # Send POST request
        response = requests.post(url, json=workflow_config)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("Workflow Execution Successful!")
            print("Execution Time:", result.get('execution_time', 'N/A'), "seconds")
            print("\nWorkflow Result:")
            print(json.dumps(result.get('result', {}), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")

if __name__ == "__main__":
    execute_workflow()
