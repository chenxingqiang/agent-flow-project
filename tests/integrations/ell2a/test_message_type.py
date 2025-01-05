"""Test message type module."""

import pytest
from typing import List, Dict, Any
from agentflow.ell2a.types.message import Message, MessageRole, ELL, system, user

def test_message_creation():
    """Test message creation."""
    msg = Message(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"
    assert msg.timestamp is not None
    assert msg.metadata == {}

def test_system_message():
    """Test system message creation."""
    msg = system("System message")
    assert msg.role == MessageRole.SYSTEM
    assert msg.content == "System message"

def test_user_message():
    """Test user message creation."""
    msg = user("User message")
    assert msg.role == MessageRole.USER
    assert msg.content == "User message"

def test_ell_message_management():
    """Test ELL message management."""
    ell = ELL()
    msg1 = system("System message")
    msg2 = user("User message")
    
    ell.add_message(msg1)
    ell.add_message(msg2)
    
    messages = ell.get_messages()
    assert len(messages) == 2
    assert messages[0] == msg1
    assert messages[1] == msg2
    
    ell.clear_messages()
    assert len(ell.get_messages()) == 0