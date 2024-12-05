import json
import os
from pathlib import Path

def create_test_data():
    """Create test data files"""
    data_dir = Path(__file__).parent
    
    # Create test config
    config = {
        "variables": {
            "test_var": "test_value"
        },
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.5
        },
        "workflow": {
            "max_iterations": 10,
            "logging_level": "INFO",
            "distributed": False,
            "steps": [
                {
                    "type": "research_planning",
                    "config": {
                        "depth": "comprehensive"
                    }
                },
                {
                    "type": "document_generation",
                    "config": {
                        "format": "academic"
                    }
                }
            ]
        }
    }
    
    with open(data_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create test workflow
    workflow = {
        "WORKFLOW": [
            {
                "step": 1,
                "input": ["research_topic", "deadline", "academic_level"],
                "output": {"type": "research"}
            },
            {
                "step": 2,
                "input": ["WORKFLOW.1"],
                "output": {"type": "document"}
            }
        ]
    }
    
    with open(data_dir / 'workflow.json', 'w') as f:
        json.dump(workflow, f, indent=2)
    
    # Create test student needs
    student_needs = {
        "research_topic": "Test Research Topic",
        "deadline": "2024-12-31",
        "academic_level": "PhD",
        "field": "Computer Science",
        "special_requirements": "Focus on practical applications"
    }
    
    with open(data_dir / 'student_needs.json', 'w') as f:
        json.dump(student_needs, f, indent=2)

if __name__ == "__main__":
    create_test_data() 