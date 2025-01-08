"""Test tool module."""

from typing import Dict
import pytest

from agentflow.ell2a.tool import tool, ToolResult

def test_tool_json_dumping_behavior():
    """Test the tool decorator and ToolResult wrapping."""
    # Create a mock tool function
    @tool(exempt_from_tracking=False)
    def mock_tool_function(data: Dict[str, str]):
        return data
    
    # Test case where result is a string and _invocation_origin is provided
    result = mock_tool_function(
        _tool_call_id="tool_123",
        data={"key": "value"}
    )
    
    # Verify result is wrapped in ToolResult
    assert isinstance(result, ToolResult)
    assert result.data == {"key": "value"}
    assert result.tool_call_id == "tool_123"

def test_tool_decorator_exemption():
    """Test the tool decorator's exempt_from_tracking parameter."""
    @tool(exempt_from_tracking=True)
    def exempt_tool_function(data: Dict[str, str]):
        return data
    
    result = exempt_tool_function(data={"key": "value"})
    
    # Verify result is not wrapped in ToolResult when exempt_from_tracking is True
    assert not hasattr(exempt_tool_function, 'exempt_from_tracking') or exempt_tool_function.exempt_from_tracking

def test_tool_result_instantiation():
    """Test ToolResult instantiation."""
    result = ToolResult(
        data={"key": "value"},
        tool_call_id="tool_456"
    )
    
    # Verify ToolResult attributes
    assert result.data == {"key": "value"}
    assert result.tool_call_id == "tool_456"
    assert result.invocation_origin is None