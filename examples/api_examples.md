# Workflow API Examples

## Synchronous Workflow Execution (curl)

```bash
curl -X POST http://localhost:8000/workflow/execute \
     -H "Content-Type: application/json" \
     -d '{
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
             "research_topic": "AI Ethics in Distributed Computing",
             "deadline": "2024-09-15",
             "academic_level": "Master"
         }
     }'
```

## Asynchronous Workflow Execution (curl)

```bash
# Initiate Async Workflow
curl -X POST http://localhost:8000/workflow/execute_async \
     -H "Content-Type: application/json" \
     -d '{
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
             "research_topic": "Edge Computing Optimization",
             "deadline": "2024-11-30",
             "academic_level": "PhD"
         }
     }'

# Retrieve Async Result (replace {result_ref} with actual reference)
curl -X GET http://localhost:8000/workflow/result/{result_ref}
```

## Postman Collection

### Sync Workflow Request

- Method: POST
- URL: `http://localhost:8000/workflow/execute`
- Body (raw JSON):

```json
{
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
        "research_topic": "Blockchain in Distributed AI",
        "deadline": "2024-07-15",
        "academic_level": "PhD"
    }
}
```

### Async Workflow Request

- Method: POST
- URL: `http://localhost:8000/workflow/execute_async`
- Body (same format as sync request)

### Retrieve Async Result

- Method: GET
- URL: `http://localhost:8000/workflow/result/{result_ref}`

```
