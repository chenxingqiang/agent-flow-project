"""Tests for ELL2A message types."""

import time
import pytest
from typing import Union, List

from agentflow.ell2a.types.message import (
    ContentBlock, 
    MessageRole, 
    Message, 
    system, 
    user, 
    ToolResult
)

def test_content_block_creation():
    """Test ContentBlock creation."""
    # Test with text
    text_block = ContentBlock(text="Hello, world!")
    assert text_block.text == "Hello, world!"
    assert text_block.image is None
    assert text_block.image_detail is None
    assert text_block.parsed is None

    # Test with image
    image_block = ContentBlock(
        image="base64_encoded_image", 
        image_detail="high_resolution"
    )
    assert image_block.image == "base64_encoded_image"
    assert image_block.image_detail == "high_resolution"

def test_message_role_enum():
    """Test MessageRole enum."""
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"

def test_message_creation():
    """Test Message class creation."""
    # Test with string content
    msg1 = Message(role=MessageRole.USER, content="Hello")
    assert msg1.role == MessageRole.USER
    assert msg1.content == "Hello"
    assert msg1.timestamp is not None
    assert isinstance(msg1.timestamp, float)
    assert msg1.metadata == {}

    # Test with content block
    content_block = ContentBlock(text="Test block")
    msg2 = Message(
        role=MessageRole.ASSISTANT, 
        content=[content_block],
        metadata={"source": "test"}
    )
    assert msg2.role == MessageRole.ASSISTANT
    assert isinstance(msg2.content, list)
    assert msg2.content[0].text == "Test block"
    assert msg2.metadata == {"source": "test"}

def test_message_timestamp():
    """Test Message timestamp behavior."""
    # Verify timestamp is set to current time if not provided
    before = time.time()
    msg = Message(role=MessageRole.USER, content="Test")
    after = time.time()
    
    assert before <= msg.timestamp <= after

def test_message_factory_functions():
    """Test system and user message factory functions."""
    # Test system message
    sys_msg = system("System instruction")
    assert sys_msg.role == MessageRole.SYSTEM
    assert sys_msg.content == "System instruction"

    # Test user message
    user_msg = user("User input")
    assert user_msg.role == MessageRole.USER
    assert user_msg.content == "User input"

def test_tool_result():
    """Test ToolResult creation."""
    # Create content blocks
    blocks = [
        ContentBlock(text="Result 1"),
        ContentBlock(text="Result 2")
    ]
    
    # Create ToolResult
    tool_result = ToolResult(
        tool_call_id="test_call_123", 
        result=blocks
    )
    
    assert tool_result.tool_call_id == "test_call_123"
    assert len(tool_result.result) == 2
    assert tool_result.result[0].text == "Result 1"
    assert tool_result.result[1].text == "Result 2"
