"""Tests for ELL2A Message class."""

import pytest
from agentflow.ell2a import Message, MessageRole, MessageValidationError
import json
import jsonschema
import logging
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

def test_message_creation_with_minimal_args():
    """Test creating a message with minimal arguments."""
    message = Message(role=MessageRole.USER, content="Hello, world!")
    
    assert message.role == MessageRole.USER
    assert message.content == "Hello, world!"
    assert message.metadata is None


def test_message_creation_with_metadata():
    """Test creating a message with metadata."""
    metadata = {"timestamp": "2025-01-08", "source": "test"}
    message = Message(
        role=MessageRole.ASSISTANT, 
        content="This is a test response", 
        metadata=metadata
    )
    
    assert message.role == MessageRole.ASSISTANT
    assert message.content == "This is a test response"
    assert message.metadata == metadata


def test_message_to_dict_minimal():
    """Test converting a minimal message to dictionary."""
    message = Message(role=MessageRole.SYSTEM, content="System initialization")
    message_dict = message.to_dict()
    
    assert message_dict == {
        "role": "system",
        "content": "System initialization",
        "metadata": {}
    }


def test_message_to_dict_with_metadata():
    """Test converting a message with metadata to dictionary."""
    metadata = {"context": "unit test", "priority": "high"}
    message = Message(
        role=MessageRole.FUNCTION, 
        content="Function call result", 
        metadata=metadata
    )
    message_dict = message.to_dict()
    
    assert message_dict == {
        "role": "function",
        "content": "Function call result",
        "metadata": metadata
    }


def test_message_roles():
    """Test all defined message roles."""
    roles = [
        (MessageRole.SYSTEM, "system"),
        (MessageRole.USER, "user"),
        (MessageRole.ASSISTANT, "assistant"),
        (MessageRole.FUNCTION, "function")
    ]
    
    for role_enum, role_value in roles:
        message = Message(role=role_enum, content="Test message")
        assert message.role == role_enum
        assert message.to_dict()["role"] == role_value


def test_message_immutability():
    """Test that message attributes cannot be modified after creation."""
    message = Message(role=MessageRole.USER, content="Original content")
    
    with pytest.raises(ValidationError):
        message.role = MessageRole.SYSTEM
    
    with pytest.raises(ValidationError):
        message.content = "Modified content"
    
    with pytest.raises(ValidationError):
        message.metadata = {"new": "metadata"}


"""Advanced tests for ELL2A Message class."""

import pytest
from typing import Dict, Any
from agentflow.ell2a import Message, MessageRole


def test_message_creation_edge_cases():
    """Test message creation with various edge case inputs."""
    # Empty content
    empty_message = Message(role=MessageRole.SYSTEM, content="")
    assert empty_message.content == ""

    # Whitespace-only content
    whitespace_message = Message(role=MessageRole.USER, content="   \t\n")
    assert whitespace_message.content == "   \t\n"

    # Unicode content
    unicode_message = Message(role=MessageRole.ASSISTANT, content="こんにちは 🌍")
    assert unicode_message.content == "こんにちは 🌍"


def test_message_metadata_variations():
    """Test message creation with different metadata types and scenarios."""
    # Nested metadata
    nested_metadata = {
        "context": {
            "source": "test",
            "nested": {"key": "value"}
        },
        "tags": ["important", "urgent"]
    }
    nested_message = Message(
        role=MessageRole.FUNCTION, 
        content="Nested metadata test", 
        metadata=nested_metadata
    )
    assert nested_message.metadata == nested_metadata

    # Metadata with different types
    complex_metadata = {
        "numeric": 42,
        "boolean": True,
        "none_value": None,
        "list": [1, 2, 3],
        "nested_dict": {"a": 1, "b": 2}
    }
    complex_message = Message(
        role=MessageRole.USER, 
        content="Complex metadata test", 
        metadata=complex_metadata
    )
    assert complex_message.metadata == complex_metadata


def test_message_to_dict_comprehensive():
    """Comprehensive test of to_dict method with various input scenarios."""
    test_cases = [
        # Minimal message
        {
            "message": Message(role=MessageRole.SYSTEM, content="Minimal"),
            "expected": {
                "role": "system",
                "content": "Minimal",
                "metadata": {}
            }
        },
        # Message with complex metadata
        {
            "message": Message(
                role=MessageRole.USER, 
                content="Complex metadata", 
                metadata={
                    "context": {"level": "test"},
                    "priority": 1,
                    "tags": ["important"]
                }
            ),
            "expected": {
                "role": "user",
                "content": "Complex metadata",
                "metadata": {
                    "context": {"level": "test"},
                    "priority": 1,
                    "tags": ["important"]
                }
            }
        }
    ]

    for case in test_cases:
        assert case["message"].to_dict() == case["expected"]


def test_message_role_consistency():
    """Ensure message roles are consistent across different operations."""
    roles = [
        (MessageRole.SYSTEM, "Initialization message"),
        (MessageRole.USER, "User input"),
        (MessageRole.ASSISTANT, "AI response"),
        (MessageRole.FUNCTION, "Function call result")
    ]

    for role, content in roles:
        message = Message(role=role, content=content)
        
        # Check role in message object
        assert message.role == role
        
        # Check role in dictionary representation
        message_dict = message.to_dict()
        assert message_dict["role"] == role.value


def test_message_equality():
    """Test message equality and hashing."""
    # Messages with same content and role should be considered equal
    message1 = Message(role=MessageRole.USER, content="Test")
    message2 = Message(role=MessageRole.USER, content="Test")
    
    assert message1 == message2
    assert hash(message1) == hash(message2)

    # Messages with different content or role should not be equal
    message3 = Message(role=MessageRole.SYSTEM, content="Test")
    message4 = Message(role=MessageRole.USER, content="Different")
    
    assert message1 != message3
    assert message1 != message4
    assert hash(message1) != hash(message3)
    assert hash(message1) != hash(message4)


def test_message_repr():
    """Test string representation of Message."""
    message = Message(
        role=MessageRole.ASSISTANT, 
        content="Hello, world!", 
        metadata={"context": "test"}
    )
    
    # Check that repr contains key information
    repr_str = repr(message)
    assert "Message" in repr_str
    assert "ASSISTANT" in repr_str
    assert "Hello, world!" in repr_str
    assert "context" in repr_str


def test_message_serialization():
    """Test basic serialization compatibility."""
    import json

    message = Message(
        role=MessageRole.FUNCTION, 
        content="Serialization test", 
        metadata={"id": 123}
    )

    # Convert to dictionary
    message_dict = message.to_dict()

    try:
        # Attempt JSON serialization
        json_str = json.dumps(message_dict)
        deserialized = json.loads(json_str)
        
        # Verify deserialized data matches original
        assert deserialized["role"] == "function"
        assert deserialized["content"] == "Serialization test"
        assert deserialized["metadata"] == {"id": 123}
    except Exception as e:
        pytest.fail(f"Serialization failed: {e}")


def test_message_with_special_characters():
    """Test message creation with special characters and formatting."""
    special_content_cases = [
        "Line1\nLine2\tTabbed",
        "Quotes \"Test\" and 'Another'",
        "Special chars: !@#$%^&*()_+",
        "Emoji test: 🚀🌈🤖"
    ]

    for content in special_content_cases:
        message = Message(role=MessageRole.USER, content=content)
        assert message.content == content
        assert message.to_dict()["content"] == content


import sys
import time
import pytest
from typing import Dict, Any, List
from agentflow.ell2a import Message, MessageRole


def test_message_performance():
    """Test performance of message creation and to_dict method."""
    # Large number of messages creation
    start_time = time.time()
    messages = [
        Message(
            role=MessageRole.USER, 
            content=f"Performance test message {i}", 
            metadata={"index": i, "timestamp": time.time()}
        ) for i in range(10000)
    ]
    creation_time = time.time() - start_time

    # Measure to_dict performance
    start_time = time.time()
    message_dicts = [msg.to_dict() for msg in messages]
    to_dict_time = time.time() - start_time

    # Performance assertions
    assert creation_time < 1.0, f"Message creation took too long: {creation_time} seconds"
    assert to_dict_time < 1.0, f"to_dict conversion took too long: {to_dict_time} seconds"
    assert len(messages) == 10000
    assert len(message_dicts) == 10000


def test_message_memory_efficiency():
    """Test memory efficiency of Message instances."""
    # Create a large number of messages
    messages = [
        Message(
            role=MessageRole.ASSISTANT, 
            content=f"Memory test {i}", 
            metadata={"test": "efficiency"}
        ) for i in range(100000)
    ]

    # Check memory usage
    message_size = sys.getsizeof(messages[0])
    total_size = sum(sys.getsizeof(msg) for msg in messages)

    # Rough estimate of total memory usage
    estimated_total_size = message_size * len(messages)

    # Assertions to check memory efficiency
    assert message_size < 200, f"Individual message size too large: {message_size} bytes"
    assert total_size < estimated_total_size * 1.5, "Total memory usage is inefficient"


def test_message_logging_and_validation():
    """Test logging and validation mechanisms."""
    import logging

    # Capture log messages
    log_capture = []
    handler = logging.StreamHandler()
    logger = logging.getLogger('agentflow.ell2a')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    def log_capture_handler(record):
        log_capture.append(record.getMessage())

    handler.emit = log_capture_handler

    # Test validation failures with logging
    try:
        Message(
            role=MessageRole.USER,
            content=123,  # Invalid content type
            metadata={"invalid": object()}  # Unsupported metadata type
        )
        assert False, "Should have raised a validation error"
    except MessageValidationError as e:
        # Check that both errors were logged
        assert any("Unsupported metadata type" in msg for msg in log_capture), "Missing metadata validation error"
        assert any("Metadata contains unsupported types" in msg for msg in log_capture), "Missing metadata validation error"


def test_message_validation():
    """Advanced validation tests for Message creation."""
    # Test invalid role type
    with pytest.raises(MessageValidationError, match="role must be a MessageRole"):
        Message(role="invalid_role", content="Test")  # type: ignore

def test_message_metadata_validation():
    """Test advanced metadata validation."""
    # Test nested metadata validation
    valid_nested_metadata = {
        "context": {
            "source": "test",
            "details": {
                "priority": 1,
                "tags": ["important"]
            }
        },
        "timestamp": 1704672000,
        "active": True
    }

    # Should pass validation
    message = Message(
        role=MessageRole.USER,
        content="Nested metadata test",
        metadata=valid_nested_metadata
    )
    assert message.metadata == valid_nested_metadata

    # Test invalid metadata scenarios
    invalid_cases = [
        # Unsupported type in metadata
        {
            "complex_obj": object()
        },
        # Unsupported type in nested list
        {
            "list_with_invalid": [1, 2, {"invalid": object()}]
        }
    ]

    for invalid_metadata in invalid_cases:
        with pytest.raises(MessageValidationError):
            Message(
                role=MessageRole.USER,
                content="Invalid metadata test",
                metadata=invalid_metadata
            )


def test_message_extreme_metadata():
    """Test message creation with extreme metadata scenarios."""
    # Very large metadata
    large_metadata = {
        "nested_" + str(i): {"deep_" + str(j): j for j in range(10)} 
        for i in range(1000)
    }
    large_message = Message(
        role=MessageRole.FUNCTION, 
        content="Large metadata test", 
        metadata=large_metadata
    )
    assert large_message.metadata == large_metadata

    # Metadata with complex nested structures
    complex_nested_metadata = {
        "list_of_dicts": [
            {"a": 1, "b": 2},
            {"c": 3, "d": 4}
        ],
        "nested_complex": {
            "inner": {
                "deep": {
                    "very_deep": "value"
                }
            }
        }
    }

    # Circular reference test (should not raise an error)
    complex_nested_metadata["circular_ref"] = complex_nested_metadata
    complex_message = Message(
        role=MessageRole.USER, 
        content="Complex nested metadata", 
        metadata=complex_nested_metadata
    )
    assert complex_message.metadata == complex_nested_metadata


def test_message_thread_safety():
    """Basic thread safety test for Message class."""
    import threading
    import traceback

    # Shared list to collect results and exceptions
    results: List[Message] = []
    exceptions: List[Exception] = []

    # Lock to synchronize access
    lock = threading.Lock()

    def create_message(role: MessageRole, content: str, metadata: Optional[Dict[str, Any]] = None):
        try:
            # Ensure metadata is a dictionary or None
            if metadata is None:
                metadata = {}
            
            message = Message(role=role, content=content, metadata=metadata)
            with lock:
                results.append(message)
        except Exception as e:
            with lock:
                exceptions.append(e)
                traceback.print_exc()

    # Create multiple threads to create messages concurrently
    threads = [
        threading.Thread(target=create_message, args=(MessageRole.USER, f"Thread message {i}"))
        for i in range(100)
    ]

    # Start threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Print any exceptions for debugging
    if exceptions:
        print(f"Encountered {len(exceptions)} exceptions during thread safety test:")
        for ex in exceptions:
            print(ex)

    # Verify all messages were created
    assert len(results) == 100, f"Expected 100 messages, got {len(results)}. Exceptions: {exceptions}"


def test_message_comparison_with_none():
    """Test message comparison with None and other types."""
    message = Message(role=MessageRole.SYSTEM, content="Test")

    # Comparison with None
    assert message != None  # type: ignore
    assert not message == None  # type: ignore

    # Comparison with other types
    assert message != "not a message"
    assert message != 42
    assert message != {}


def test_message_hash_distribution():
    """Test hash distribution to ensure good uniqueness."""
    # Create a large number of unique messages
    messages = [
        Message(
            role=MessageRole.USER, 
            content=f"Unique message {i}", 
            metadata={"index": i}
        ) for i in range(10000)
    ]

    # Collect hashes
    message_hashes = [hash(msg) for msg in messages]

    # Check for hash collisions
    unique_hashes = len(set(message_hashes))
    assert unique_hashes == len(messages), "Hash collision detected"

    # Check hash distribution (rough statistical test)
    hash_distribution = {}
    for h in message_hashes:
        bucket = h % 100
        hash_distribution[bucket] = hash_distribution.get(bucket, 0) + 1

    # Ensure relatively uniform distribution
    max_count = max(hash_distribution.values())
    min_count = min(hash_distribution.values())
    assert max_count < len(messages) * 0.02, "Poor hash distribution"
    assert min_count > len(messages) * 0.0005, "Poor hash distribution"


def test_message_metadata_validation():
    """Test advanced metadata validation."""
    # Test nested metadata validation
    valid_nested_metadata = {
        "context": {
            "source": "test",
            "details": {
                "priority": 1,
                "tags": ["important"]
            }
        },
        "timestamp": 1704672000,
        "active": True
    }
    
    # Should pass validation
    message = Message(
        role=MessageRole.USER, 
        content="Nested metadata test", 
        metadata=valid_nested_metadata
    )
    assert message.metadata == valid_nested_metadata

    # Test invalid metadata scenarios
    invalid_cases = [
        # Unsupported type in metadata
        {
            "complex_obj": object()
        },
        # Unsupported type in nested list
        {
            "list_with_invalid": [1, 2, {"invalid": object()}]
        }
    ]

    for invalid_metadata in invalid_cases:
        with pytest.raises(MessageValidationError):
            Message(
                role=MessageRole.USER,
                content="Invalid metadata test",
                metadata=invalid_metadata
            )

    # Deeply nested metadata (should pass)
    deep_nested_metadata = {
        "a": {"b": {"c": {"d": {"e": 1}}}}
    }
    deep_message = Message(
        role=MessageRole.USER, 
        content="Deep nested test", 
        metadata=deep_nested_metadata
    )
    assert deep_message.metadata == deep_nested_metadata


def test_message_with_metadata():
    """Test metadata manipulation methods."""
    # Create initial message
    original_message = Message(
        role=MessageRole.USER, 
        content="Original message", 
        metadata={"source": "test"}
    )

    # Add new metadata
    updated_message = original_message.with_metadata(
        priority="high", 
        timestamp=1704672000
    )

    # Verify original message is unchanged
    assert original_message.metadata == {"source": "test"}
    
    # Verify updated message has new metadata
    assert updated_message.metadata == {
        "source": "test", 
        "priority": "high", 
        "timestamp": 1704672000
    }

    # Verify other attributes remain the same
    assert updated_message.role == original_message.role
    assert updated_message.content == original_message.content


def test_message_truncation():
    """Test message content truncation."""
    # Create a long message
    long_message = Message(
        role=MessageRole.ASSISTANT, 
        content="This is a very long message that needs to be truncated for various reasons.",
        metadata={"source": "truncation_test"}
    )

    # Truncate to different lengths
    truncate_cases = [
        (10, "This is a "),
        (5, "This "),
        (0, "")
    ]

    for max_length, expected_content in truncate_cases:
        truncated_message = long_message.truncate(max_length)
        
        # Verify truncation
        assert truncated_message.content == expected_content
        
        # Verify other attributes remain the same
        assert truncated_message.role == long_message.role
        assert truncated_message.metadata == long_message.metadata


def test_message_string_representations():
    """Test string representations of Message."""
    message = Message(
        role=MessageRole.FUNCTION, 
        content="Complex function call result", 
        metadata={"id": 123, "context": "test"}
    )

    # Test __str__ method
    str_repr = str(message)
    assert "FUNCTION" in str_repr
    assert "Complex function call result" in str_repr
    assert "(metadata: 2 items)" in str_repr

    # Test __repr__ method
    repr_repr = repr(message)
    assert "Message(role=MessageRole.FUNCTION" in repr_repr
    assert "content='Complex function call result'" in repr_repr
    assert "'id': 123" in repr_repr
    assert "'context': 'test'" in repr_repr


def test_message_advanced_hashing():
    """Test advanced hashing capabilities."""
    # Messages with complex nested metadata
    message1 = Message(
        role=MessageRole.USER, 
        content="Test message", 
        metadata={
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}]
        }
    )

    message2 = Message(
        role=MessageRole.USER, 
        content="Test message", 
        metadata={
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}]
        }
    )

    message3 = Message(
        role=MessageRole.USER, 
        content="Different message", 
        metadata={
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}]
        }
    )

    # Verify hash consistency
    assert hash(message1) == hash(message2)
    assert hash(message1) != hash(message3)

    # Create a set to test uniqueness
    message_set = {message1, message2, message3}
    assert len(message_set) == 2  # Only two unique messages


def test_message_metadata_retrieval():
    """Test advanced metadata retrieval methods."""
    metadata = {
        "source": "test_data",
        "priority": 1,
        "tags": ["important", "urgent"],
        "details": {
            "category": "high_level"
        }
    }
    message = Message(
        role=MessageRole.USER,
        content="Test metadata retrieval",
        metadata=metadata
    )

    # Test get_metadata with existing and non-existing keys
    assert message.get_metadata("source") == "test_data"
    assert message.get_metadata("non_existent", "default") == "default"
    assert message.get_metadata("details") == {"category": "high_level"}

    # Test filter_metadata
    filtered_high_priority = message.filter_metadata(
        lambda k, v: k == "priority" and v > 0
    )
    assert filtered_high_priority == {"priority": 1}

    filtered_tags = message.filter_metadata(
        lambda k, v: k == "tags" and isinstance(v, list)
    )
    assert filtered_tags == {"tags": ["important", "urgent"]}

    # Test map_metadata
    mapped_metadata = message.map_metadata(
        lambda k, v: v.upper() if isinstance(v, str) else v
    )
    assert mapped_metadata["source"] == "TEST_DATA"
    assert mapped_metadata["priority"] == 1
    assert mapped_metadata["details"] == {"category": "high_level"}


def test_message_metadata_edge_cases():
    """Test metadata methods with edge case scenarios."""
    # Empty metadata
    empty_message = Message(
        role=MessageRole.SYSTEM,
        content="Empty metadata test"
    )
    assert empty_message.get_metadata("key", "default") == "default"
    assert empty_message.filter_metadata(lambda k, v: True) == {}
    assert empty_message.map_metadata(lambda k, v: v) == {}

    # Nested metadata transformations
    nested_metadata = {
        "user": {
            "name": "John",
            "age": 30
        },
        "settings": {
            "theme": "dark"
        }
    }
    nested_message = Message(
        role=MessageRole.USER,
        content="Nested metadata test",
        metadata=nested_metadata
    )

    # Complex filter
    filtered_nested = nested_message.filter_metadata(
        lambda k, v: isinstance(v, dict) and "name" in v
    )
    assert filtered_nested == {"user": {"name": "John", "age": 30}}

    # Complex map
    mapped_nested = nested_message.map_metadata(
        lambda k, v: {
            "user": lambda x: {"full_name": x.get("name", "")},
            "settings": lambda x: {"ui_theme": x.get("theme", "")}
        }.get(k, lambda x: x)(v)
    )
    assert mapped_nested == {
        "user": {"full_name": "John"},
        "settings": {"ui_theme": "dark"}
    }

def test_metadata_schema_validation():
    """Test metadata schema validation."""
    # Define a JSON schema
    user_metadata_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "age"],
        "additionalProperties": False
    }

    # Valid metadata
    valid_metadata = {
        "name": "John Doe",
        "age": 30,
        "tags": ["developer", "engineer"]
    }
    valid_message = Message(
        role=MessageRole.USER,
        content="Schema validation test",
        metadata=valid_metadata,
        metadata_schema=user_metadata_schema
    )
    assert valid_message.metadata == valid_metadata

    # Invalid metadata (should raise validation error)
    invalid_cases = [
        # Missing required field
        {"name": "Jane"},
        # Wrong type
        {"name": "Alice", "age": "25"},
        # Additional properties not allowed
        {"name": "Bob", "age": 40, "extra": "field"}
    ]

    for case in invalid_cases:
        with pytest.raises(MessageValidationError):
            Message(
                role=MessageRole.USER,
                content="Invalid schema test",
                metadata=case,
                metadata_schema=user_metadata_schema
            )

def test_advanced_metadata_transformation():
    """Test advanced metadata transformation."""
    original_metadata = {
        "user": {
            "name": "John",
            "age": 30
        },
        "preferences": {
            "theme": "dark",
            "notifications": True
        }
    }

    message = Message(
        role=MessageRole.USER,
        content="Transformation test",
        metadata=original_metadata
    )

    # Complex transformation function
    def transform_metadata(metadata):
        # Uppercase user name, add a new field
        metadata['user']['name'] = metadata['user']['name'].upper()
        metadata['user']['is_adult'] = metadata['user']['age'] >= 18
        
        # Rename theme to ui_theme
        metadata['preferences']['ui_theme'] = metadata['preferences'].pop('theme')
        
        return metadata

    transformed_message = message.advanced_metadata_transform(transform_metadata)

    # Verify transformation
    assert transformed_message.metadata == {
        "user": {
            "name": "JOHN",
            "age": 30,
            "is_adult": True
        },
        "preferences": {
            "notifications": True,
            "ui_theme": "dark"
        }
    }

def test_json_serialization():
    """Test JSON serialization and deserialization."""
    # Metadata with complex structure
    metadata = {
        "context": {
            "source": "test",
            "timestamp": 1609459200
        },
        "tags": ["important", "urgent"]
    }

    # Create a message
    original_message = Message(
        role=MessageRole.SYSTEM,
        content="JSON serialization test",
        metadata=metadata
    )

    # Serialize to JSON
    json_str = original_message.to_json(indent=2)
    
    # Deserialize from JSON
    reconstructed_message = Message.from_json(json_str)

    # Verify reconstruction
    assert reconstructed_message.role == original_message.role
    assert reconstructed_message.content == original_message.content
    assert reconstructed_message.metadata == original_message.metadata

    # Test with custom schema
    schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "timestamp": {"type": "integer"}
        }
    }
    reconstructed_with_schema = Message.from_json(
        json_str, 
        metadata_schema=schema
    )
    assert reconstructed_with_schema.metadata == original_message.metadata
