"""Test SQL store module."""

import pytest
from typing import List, Dict, Any

from agentflow.ell2a.stores.sql import SQLStore, SerializedLMP, Base
from sqlmodel import Session, select
from sqlalchemy import Engine, create_engine, func

from agentflow.ell2a.types.lmp import LMPType
from agentflow.ell2a.util.time import utc_now


@pytest.fixture
def in_memory_db():
    """Create in-memory database engine."""
    return create_engine("sqlite:///:memory:", echo=True)

@pytest.fixture
def sql_store(in_memory_db: Engine) -> SQLStore:
    """Create SQL store with in-memory database."""
    store = SQLStore(in_memory_db)
    return store

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