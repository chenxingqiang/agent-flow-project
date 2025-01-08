"""Tests for ELL2A LMP types."""

import pytest
from agentflow.ell2a.types.lmp import LMPType

def test_lmp_type_enum_values():
    """Test LMPType enum values."""
    # Verify enum values
    assert LMPType.LM.value == "LM"
    assert LMPType.TOOL.value == "TOOL"
    assert LMPType.LABELER.value == "LABELER"
    assert LMPType.FUNCTION.value == "FUNCTION"
    assert LMPType.OTHER.value == "OTHER"

def test_lmp_type_enum_comparison():
    """Test LMPType enum comparison and type."""
    # Verify each enum is of the correct type
    assert isinstance(LMPType.LM, LMPType)
    assert isinstance(LMPType.TOOL, LMPType)
    assert isinstance(LMPType.LABELER, LMPType)
    assert isinstance(LMPType.FUNCTION, LMPType)
    assert isinstance(LMPType.OTHER, LMPType)

def test_lmp_type_enum_string_conversion():
    """Test string conversion of LMPType."""
    # Verify string conversion
    assert LMPType.LM.value == "LM"
    assert LMPType.TOOL.value == "TOOL"
    assert LMPType.LABELER.value == "LABELER"
    assert LMPType.FUNCTION.value == "FUNCTION"
    assert LMPType.OTHER.value == "OTHER"

def test_lmp_type_enum_iteration():
    """Test iteration and membership of LMPType."""
    # Get all enum values
    lmp_types = list(LMPType)
    
    # Verify number of enum members
    assert len(lmp_types) == 5
    
    # Verify all expected types are present
    expected_types = {"LM", "TOOL", "LABELER", "FUNCTION", "OTHER"}
    assert {t.value for t in lmp_types} == expected_types

def test_lmp_type_enum_equality():
    """Test equality comparisons of LMPType."""
    # Verify equality works correctly
    assert LMPType.LM == LMPType.LM
    assert LMPType.TOOL == LMPType.TOOL
    
    # Verify inequality works correctly
    assert LMPType.LM != LMPType.TOOL
    assert LMPType.FUNCTION != LMPType.LABELER
