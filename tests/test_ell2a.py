"""Tests for ELL2A module."""
import pytest
from agentflow.ell2a import (
    Message, 
    MessageRole, 
    MessageValidationError, 
    ELL, 
    system, 
    user, 
    assistant
)
from typing import Dict, Any
import logging

class TestMessage:
    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(
            role=MessageRole.USER, 
            content="Hello, world!", 
            metadata={"context": "test"}
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert msg.metadata == {"context": "test"}

    def test_message_validation(self):
        """Test message validation."""
        with pytest.raises(MessageValidationError):
            Message(
                role=MessageRole.USER, 
                content=None  # Invalid: content cannot be None
            )

    def test_message_metadata_validation(self):
        """Test metadata validation."""
        # Test metadata with complex nested structure
        metadata = {
            "nested": {
                "key1": "value1",
                "key2": 42
            },
            "list": [1, 2, 3]
        }
        msg = Message(
            role=MessageRole.SYSTEM, 
            content="Test metadata", 
            metadata=metadata
        )
        assert msg.metadata == metadata

    def test_message_metadata_schema(self):
        """Test metadata schema validation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }
        
        # Valid metadata
        msg = Message(
            role=MessageRole.USER, 
            content="Profile", 
            metadata={"name": "John", "age": 30},
            metadata_schema=schema
        )
        
        # Invalid metadata should raise an error
        with pytest.raises(Exception):  # Specific exception depends on jsonschema implementation
            Message(
                role=MessageRole.USER, 
                content="Invalid Profile", 
                metadata={"age": -1},
                metadata_schema=schema
            )

class TestELL:
    def test_ell_initialization(self):
        """Test ELL initialization."""
        ell = ELL(model_name="test_model", config={"max_tokens": 100})
        assert ell.model_name == "test_model"
        assert ell.config == {"max_tokens": 100}
        assert len(ell.conversation_history) == 0

    def test_message_processing(self):
        """Test basic message processing."""
        ell = ELL(model_name="test_model")
        msg = Message(role=MessageRole.USER, content="Hello")
        
        # Mock the process_message method (in a real scenario, this would be replaced with actual processing)
        def mock_process(message):
            return Message(role=MessageRole.ASSISTANT, content="Hi there!")
        
        ell.process_message = mock_process
        response = ell.process_message(msg)
        
        assert response.role == MessageRole.ASSISTANT
        assert response.content == "Hi there!"

    def test_conversation_history(self):
        """Test conversation history management."""
        ell = ELL(model_name="test_model")
        
        # Add messages to history
        msg1 = Message(role=MessageRole.USER, content="First message")
        msg2 = Message(role=MessageRole.ASSISTANT, content="First response")
        
        ell.conversation_history.extend([msg1, msg2])
        
        assert len(ell.get_history()) == 2
        
        ell.clear_history()
        assert len(ell.get_history()) == 0

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
        ell = ELL(model_name="test_model")
        
        with pytest.raises(TypeError, match="Expected Message"):
            await ell.process_message("Not a Message object")
    
    def test_validation_error_context(self):
        """Test that validation errors include helpful context."""
        with pytest.raises(MessageValidationError) as excinfo:
            # Create an invalid message (e.g., None content)
            Message(
                role=MessageRole.USER, 
                content=None,
                metadata={"test": "context"}
            )
        
        # Check that the error includes context information
        error_str = str(excinfo.value)
        assert "Context:" in error_str
        assert "'role': 'MessageRole.USER'" in error_str
        assert "'content_length': 0" in error_str
        assert "'metadata_keys': ['test']" in error_str

    @pytest.mark.asyncio
    async def test_logging_on_validation(self, caplog):
        """Test logging behavior during message validation."""
        caplog.set_level(logging.DEBUG)
        
        ell = ELL(model_name="test_model")
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
            def __post_init__(self):
                raise Exception("Simulated unexpected error")
        
        with pytest.raises(Exception, match="Simulated unexpected error"):
            BrokenMessage(role=MessageRole.USER, content="Broken")
