"""Test tool module."""

from typing import Dict
from unittest.mock import patch
from ell2a import tool
from ell2a.tool import ToolResult

def test_tool_json_dumping_behavior():
    # Create a mock tool function
    @tool(exempt_from_tracking=False)
    def mock_tool_function(data : Dict[str, str]):
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
    assert result.invocation_origin is None