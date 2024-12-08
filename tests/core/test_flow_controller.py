"""Tests for flow controller."""
import pytest
from agentflow.core.flow_controller import FlowController, ErrorStrategy, RouteCondition

@pytest.fixture
def flow_config():
    return {
        "ROUTING_RULES": {
            "DEFAULT_BEHAVIOR": "FORWARD_ALL",
            "CONDITIONAL_ROUTING": {
                "CONDITIONS": [
                    {
                        "when": "data.get('type') == 'special'",
                        "action": "TRANSFORM"
                    }
                ]
            }
        },
        "ERROR_HANDLING": {
            "STRATEGIES": ["SKIP", "RETRY"],
            "MAX_RETRIES": 2
        }
    }

@pytest.fixture
def controller(flow_config):
    return FlowController(flow_config)

def test_init(controller, flow_config):
    """Test initialization"""
    assert controller.default_behavior == "FORWARD_ALL"
    assert len(controller.conditions) == 1
    assert len(controller.error_config.strategies) == 2

def test_route_data_default(controller):
    """Test default routing behavior"""
    data = {"key": "value"}
    result = controller.route_data(data)
    assert result == {"forward": data}

def test_route_data_conditional(controller):
    """Test conditional routing"""
    data = {"type": "special", "content": "test"}
    result = controller.route_data(data)
    assert "transformed" in result

def test_error_handling_skip(controller):
    """Test error handling with skip strategy"""
    def failing_operation():
        raise ValueError("Test error")
        
    result = controller._handle_error(ValueError("Test error"), {})
    assert result["status"] == "skipped"

def test_error_handling_retry(controller):
    """Test error handling with retry strategy"""
    data = {"key": "value"}
    controller.error_config.strategies = [ErrorStrategy.RETRY]
    
    with pytest.raises(ValueError):
        for _ in range(controller.error_config.max_retries + 1):
            controller._retry_operation(data)

def test_transform_data(controller):
    """Test data transformation"""
    data = {"key": 123}
    result = controller._transform_data(data)
    assert isinstance(result["transformed"]["key"], str)

def test_filter_data(controller):
    """Test data filtering"""
    data = {"key1": "value", "key2": None}
    result = controller._filter_data(data)
    assert "key2" not in result["filtered"]

def test_evaluate_condition(controller):
    """Test condition evaluation"""
    data = {"type": "special"}
    condition = "data.get('type') == 'special'"
    assert controller._evaluate_condition(condition, data) == True
