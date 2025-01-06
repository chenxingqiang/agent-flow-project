import pytest
import tempfile
import json
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test config file
        config_data = {
            "AGENT": {
                "name": "TestAgent",
                "type": "research",
                "version": "1.0.0",
                "workflow_path": "test_workflow.yaml"
            }
        }
        
        # Create config directory
        config_dir = Path(tmpdir)
        config_dir.mkdir(exist_ok=True)
        
        # Write test config file
        with open(config_dir / "config.json", "w") as f:
            json.dump(config_data, f)
            
        yield config_dir 