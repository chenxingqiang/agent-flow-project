import pytest
import json
import os
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get path to test data directory"""
    return Path(__file__).parent / 'data'

@pytest.fixture
def test_config(test_data_dir):
    """Get test configuration"""
    with open(test_data_dir / 'config.json') as f:
        return json.load(f)

@pytest.fixture
def test_workflow(test_data_dir):
    """Create a mock workflow definition for testing"""
    return {
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