"""Tests for output processor."""
import pytest
from agentflow.core.output_processor import OutputProcessor, OutputMode

@pytest.fixture
def output_spec():
    return {
        "MODES": ["RETURN", "FORWARD"],
        "STRATEGIES": {
            "RETURN": {
                "options": ["FULL_RESULT", "SUMMARY"]
            },
            "FORWARD": {
                "routing_options": ["TRANSFORM"]
            }
        },
        "TRANSFORMATION": {
            "ENABLED": True,
            "METHODS": ["FILTER", "MAP"]
        }
    }

@pytest.fixture
def processor(output_spec):
    return OutputProcessor(output_spec)

def test_init(processor, output_spec):
    """Test initialization"""
    assert len(processor.modes) == 2
    assert OutputMode.RETURN in processor.modes
    assert processor.transformation.enabled == True

def test_process_return_full(processor):
    """Test return processing with full result"""
    data = {"key": "value"}
    result = processor.process_output(data, OutputMode.RETURN)
    assert "summary" in result
    assert result["summary"] == "{'key': 'value'}"

def test_process_return_summary(processor):
    """Test return processing with summary"""
    data = {"key": "value"}
    processor.strategies["RETURN"]["options"] = ["SUMMARY"]
    result = processor.process_output(data, OutputMode.RETURN)
    assert "summary" in result

def test_process_forward(processor):
    """Test forward processing"""
    data = {"key": "value"}
    result = processor.process_output(data, OutputMode.FORWARD)
    assert "forward_data" in result
    assert isinstance(result["forward_data"], dict)

def test_transformation_filter(processor):
    """Test filter transformation"""
    data = {"key1": "value1", "key2": None}
    processor.transformation.methods = ["FILTER"]
    result = processor._filter_output(data)
    assert "key2" not in result

def test_transformation_map(processor):
    """Test map transformation"""
    data = {"key": 123}
    processor.transformation.methods = ["MAP"]
    result = processor._map_output(data)
    assert isinstance(result["key"], str)

def test_unsupported_mode(processor):
    """Test unsupported output mode"""
    with pytest.raises(ValueError):
        processor.process_output({}, OutputMode.STORE)
