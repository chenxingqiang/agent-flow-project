"""Test SQL store module."""

import pytest
from typing import List, Dict, Any, Generator

from agentflow.ell2a.stores.sql import SQLStore, SerializedLMP, Base
from sqlmodel import Session, select
from sqlalchemy import Engine, create_engine, func

from agentflow.ell2a.types.lmp import LMPType
from agentflow.ell2a.util.time import utc_now


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)  # Create all tables
    yield engine
    Base.metadata.drop_all(engine)  # Clean up tables

@pytest.fixture
def sql_store(in_memory_db):
    """Create a SQL store for testing."""
    store = SQLStore(in_memory_db)
    yield store

def test_write_lmp(sql_store: SQLStore):
    """Test writing LMP to store."""
    # Arrange
    lmp_data = {
        "type": LMPType.LM,
        "name": "test_lmp",
        "description": "Test LMP",
        "parameters": {"param1": "value1"},
        "api_params": {"api_param1": "value1"}
    }
    
    # Act
    result = sql_store.write_lmp(lmp_data)
    
    # Assert
    assert result.id is not None
    assert result.type == LMPType.LM
    assert result.name == "test_lmp"
    assert result.description == "Test LMP"
    assert result.parameters == {"param1": "value1"}
    assert result.api_params == {"api_param1": "value1"}
    assert result.created_at is not None
    assert result.updated_at is not None