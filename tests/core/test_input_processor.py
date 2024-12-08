"""Tests for input processor."""
import pytest
from agentflow.core.input_processor import InputProcessor, InputMode, InputType
from agentflow.core.exceptions import ValidationError

@pytest.fixture
def input_spec():
    return {
        "MODES": ["DIRECT_INPUT", "CONTEXT_INJECTION"],
        "TYPES": {
            "CONTEXT": {
                "sources": ["PREVIOUS_AGENT_OUTPUT", "GLOBAL_MEMORY"]
            }
        },
        "VALIDATION": {
            "STRICT_MODE": True,
            "SCHEMA_VALIDATION": True,
            "TRANSFORM_STRATEGIES": ["TYPE_COERCION"]
        }
    }

@pytest.fixture
def processor(input_spec):
    return InputProcessor(input_spec)

def test_init(processor, input_spec):
    """Test initialization"""
    assert len(processor.modes) == 2
    assert InputMode.DIRECT_INPUT in processor.modes
    assert processor.validation.strict_mode == True

def test_process_direct_input(processor):
    """Test direct input processing"""
    data = {"key": "value"}
    result = processor.process_input(data, InputMode.DIRECT_INPUT)
    assert result == {"direct_data": data}

def test_process_context_injection(processor):
    """Test context injection processing"""
    context = {
        "PREVIOUS_AGENT_OUTPUT": {"result": "data"},
        "GLOBAL_MEMORY": {"cache": "value"}
    }
    result = processor.process_input(context, InputMode.CONTEXT_INJECTION)
    assert result == context

def test_invalid_context_source(processor):
    """Test invalid context source"""
    context = {"INVALID_SOURCE": "data"}
    with pytest.raises(ValidationError):
        processor.process_input(context, InputMode.CONTEXT_INJECTION)

def test_unsupported_mode(processor):
    """Test unsupported input mode"""
    with pytest.raises(ValueError):
        processor.process_input({}, InputMode.STREAM_INPUT)

def test_validation_strict_mode(processor):
    """Test strict mode validation"""
    # 这里可以添加更多具体的验证测试
    pass
