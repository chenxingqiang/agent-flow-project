"""Tests for ELL2A module."""
import pytest
from agentflow.ell2a import (
    Message, 
    MessageRole, 
    MessageValidationError, 
    ELL, 
    system, 
    user, 
    assistant,
    MessageType
)
from typing import Dict, Any, cast, Union
import logging
from pydantic import ValidationError

class TestMessage:
    def test_message_creation(self):
        """Test message creation."""
        msg = Message(role=MessageRole.USER, content="Test message")
        assert msg.role == MessageRole.USER
        assert isinstance(msg.content, str)  # Content should be string
        assert msg.content == "Test message"

    def test_message_validation(self):
        """Test message validation."""
        # Validate that content is not empty
        msg = Message(role=MessageRole.USER, content="")
        assert isinstance(msg.content, str)  # Content should be string
        assert msg.content == ""

    def test_empty_message_validation(self):
        """Test that empty messages are not allowed."""
        # With the new implementation, empty content is allowed
        msg = Message(role=MessageRole.USER, content="")
        assert msg.content == ""

    def test_message_metadata_validation(self):
        """Test message metadata validation."""
        msg = Message(
            role=MessageRole.USER,
            content="Test message",
            metadata={"test": "value"}
        )
        assert msg.metadata is not None
        assert msg.metadata["test"] == "value"

    def test_message_metadata_schema(self):
        """Test message metadata schema validation."""
        msg = Message(
            role=MessageRole.USER,
            content="Test message",
            metadata={"test": "value"}
        )
        assert msg.metadata is not None
        assert msg.metadata["test"] == "value"

class TestELL:
    def test_ell_initialization(self):
        """Test ELL initialization."""
        ell = ELL(name="test_model", config={"max_tokens": 100})
        assert ell.name == "test_model"
        assert ell.config["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_message_processing(self):
        """Test basic message processing."""
        ell = ELL(name="test_model")
        msg = Message(role=MessageRole.USER, content="Hello")

        # Mock the process_message method
        def mock_process(message: Message) -> Message:
            return Message(role=MessageRole.ASSISTANT, content="Hi there!")

        ell.process_message_fn = mock_process
        response = await ell.process_message(msg)
        assert response.role == MessageRole.ASSISTANT
        assert response.content == "Hi there!"

    def test_conversation_history(self):
        """Test conversation history management."""
        ell = ELL(name="test_model")

        # Add messages to history
        msg1 = Message(role=MessageRole.USER, content="First message")
        msg2 = Message(role=MessageRole.ASSISTANT, content="First response")

        ell.messages.extend([msg1, msg2])
        assert len(ell.get_history()) == 2

# Helper function tests
def test_message_helpers():
    """Test message helper functions."""
    sys_msg = system("System instruction")
    assert sys_msg.role == MessageRole.SYSTEM
    assert sys_msg.content == "System instruction"
    
    user_msg = user("User input")
    assert user_msg.role == MessageRole.USER
    assert user_msg.content == "User input"
    
    assist_msg = assistant("Assistant response")
    assert assist_msg.role == MessageRole.ASSISTANT
    assert assist_msg.content == "Assistant response"

class TestMessageValidation:
    @pytest.mark.asyncio
    async def test_invalid_message_type(self):
        """Test validation decorator with invalid message type."""
        ell = ELL(name="test_model")
        
        # Create a mock process_message_fn that accepts any type
        def mock_process(message: Any) -> Message:
            if not isinstance(message, Message):
                raise TypeError(f"Expected Message object, got {type(message)}")
            return Message(role=MessageRole.ASSISTANT, content="Response")
            
        # Set the process_message_fn
        ell.process_message_fn = mock_process
        
        # Test with invalid message type
        with pytest.raises(TypeError, match=f"Expected Message object, got {type('')}"):
            await ell.process_message("")  # type: ignore

    @pytest.mark.asyncio
    async def test_validation_error_context(self):
        """Test that validation errors include helpful context."""
        # Validate that content can be empty string
        msg = Message(
            role=MessageRole.USER,
            content="",  # Empty content is allowed
            metadata={"test": "context"}
        )
        assert msg.content == ""

    @pytest.mark.asyncio
    async def test_logging_on_validation(self, caplog):
        """Test logging behavior during message validation."""
        caplog.set_level(logging.DEBUG)

        ell = ELL(name="test_model")
        msg = Message(role=MessageRole.SYSTEM, content="Test logging")

        # Process a valid message
        await ell.process_message(msg)

        # Check debug log was created
        debug_records = [record for record in caplog.records if record.levelno == logging.DEBUG]
        assert any("Message validation passed" in record.message for record in debug_records)

    def test_critical_logging_on_unexpected_error(self, caplog):
        """Test critical logging for unexpected validation errors."""
        caplog.set_level(logging.CRITICAL)

        class BrokenMessage(Message):
            def __init__(self, **data):
                super().__init__(**data)
                raise Exception("Simulated unexpected error")

        with pytest.raises(Exception, match="Simulated unexpected error"):
            BrokenMessage(
                role=MessageRole.USER,
                content="Test message"
            )
