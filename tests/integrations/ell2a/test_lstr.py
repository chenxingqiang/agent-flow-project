"""Tests for ELL2A _lstr type."""

import pytest
from typing import Dict, Any

from agentflow.ell2a.types._lstr import _lstr

def test_lstr_basic_creation():
    """Test basic _lstr creation."""
    # Create with minimal arguments
    lstr1 = _lstr(text="Hello, world!")
    
    assert lstr1.text == "Hello, world!"
    assert lstr1.language is None
    assert lstr1.metadata is None
    assert str(lstr1) == "Hello, world!"

def test_lstr_full_creation():
    """Test _lstr creation with all optional arguments."""
    metadata: Dict[str, Any] = {
        "source": "test",
        "confidence": 0.95
    }
    
    lstr2 = _lstr(
        text="Bonjour, monde!", 
        language="fr",
        metadata=metadata
    )
    
    assert lstr2.text == "Bonjour, monde!"
    assert lstr2.language == "fr"
    assert lstr2.metadata == metadata

def test_lstr_string_conversion():
    """Test string conversion methods."""
    lstr = _lstr(text="Test string")
    
    # Test __str__ method
    assert str(lstr) == "Test string"

def test_lstr_pydantic_validation():
    """Test Pydantic validation and configuration."""
    # Verify that arbitrary types are allowed in metadata
    class CustomType:
        def __init__(self, value):
            self.value = value

    custom_obj = CustomType("test")
    
    lstr_with_custom = _lstr(
        text="Custom metadata", 
        metadata={"custom": custom_obj}
    )
    
    assert lstr_with_custom.metadata is not None
    assert lstr_with_custom.metadata["custom"] == custom_obj

def test_lstr_equality():
    """Test equality comparisons."""
    # Create identical _lstr objects
    lstr1 = _lstr(text="Test", language="en")
    lstr2 = _lstr(text="Test", language="en")
    lstr3 = _lstr(text="Different", language="fr")
    
    # Pydantic models should support equality comparison
    assert lstr1 == lstr2
    assert lstr1 != lstr3