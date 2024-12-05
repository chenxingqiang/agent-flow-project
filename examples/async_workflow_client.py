import requests
import time
import json

def execute_async_workflow():
    # API endpoints
    execute_url = "http://localhost:8000/workflow/execute_async"
    result_url = "http://localhost:8000/workflow/result/{}"

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
            "research_topic": "Quantum Computing in Distributed Machine Learning",
            "deadline": "2024-12-31",
            "academic_level": "PhD"
        },
        "config": {
            "logging_level": "INFO",
            "max_iterations": 15
        }
    }

    try:
        # Initiate async workflow
        async_response = requests.post(execute_url, json=workflow_config)
        
        if async_response.status_code == 200:
            result_ref = async_response.json().get('result_ref')
            print(f"Async Workflow Initiated. Result Reference: {result_ref}")

            # Poll for result
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                result_response = requests.get(result_url.format(result_ref))
                
                if result_response.status_code == 200:
                    result = result_response.json()
                    print("\nWorkflow Execution Completed!")
                    print("Retrieval Time:", result.get('retrieval_time', 'N/A'), "seconds")
                    print("\nWorkflow Result:")
                    print(json.dumps(result.get('result', {}), indent=2))
                    break
                elif result_response.status_code == 404:
                    print(f"Waiting for result... (Attempt {attempt + 1}/{max_attempts})")
                    time.sleep(2)  # Wait before next attempt
                    attempt += 1
                else:
                    print(f"Error: {result_response.status_code}")
                    print(result_response.text)
                    break

            if attempt == max_attempts:
                print("Max attempts reached. Result not available.")

        else:
            print(f"Async Workflow Initiation Error: {async_response.status_code}")
            print(async_response.text)

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")

if __name__ == "__main__":
    execute_async_workflow()
