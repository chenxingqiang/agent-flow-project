"""Test _lstr module."""

import pytest
from agentflow.ell2a.types._lstr import _lstr

def test_lstr_creation():
    """Test _lstr creation."""
    lstr = _lstr(text="Hello", language="en")
    assert lstr.text == "Hello"
    assert lstr.language == "en"
    assert lstr.metadata is None

def test_lstr_str_representation():
    """Test _lstr string representation."""
    lstr = _lstr(text="Hello")
    assert str(lstr) == "Hello"