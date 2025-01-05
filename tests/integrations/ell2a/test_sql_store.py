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
    store = SQLStore("sqlite:///:memory:")
    store.engine = in_memory_db  # Use the same engine instance
    Base.metadata.create_all(in_memory_db)  # Create tables in the test database
    return store

def test_write_lmp(sql_store: SQLStore):
    # Arrange
    lmp_id = "test_lmp_1"
    name = "Test LMP"
    source = "def test_function(): pass"
    dependencies = str(["dep1", "dep2"])
    api_params = {"param1": "value1"}
    version_number = 1
    uses = {"used_lmp_1": {}, "used_lmp_2": {}}
    global_vars = {"global_var1": "value1"}
    free_vars = {"free_var1": "value2"}
    commit_message = "Initial commit"
    created_at = utc_now()
    assert created_at.tzinfo is not None

    # Create SerializedLMP object
    serialized_lmp = SerializedLMP(
        lmp_id=lmp_id,
        name=name,
        source=source,
        dependencies=dependencies,
        lmp_type=LMPType.LM,
        api_params=api_params,
        version_number=version_number,
        initial_global_vars=global_vars,
        initial_free_vars=free_vars,
        commit_message=commit_message,
        created_at=created_at
    )

    # Act
    sql_store.write_lmp(serialized_lmp, uses)

    # Assert
    with Session(sql_store.engine) as session:
        result = session.execute(
            select(SerializedLMP).where(SerializedLMP.lmp_id == lmp_id)
        ).scalar_one_or_none()
        assert result is not None
        assert result.lmp_id == lmp_id
        assert result.name == name
        assert result.source == source
        assert result.dependencies == dependencies
        assert result.lmp_type == LMPType.LM
        assert result.api_params == api_params
        assert result.version_number == version_number
        assert result.initial_global_vars == global_vars
        assert result.initial_free_vars == free_vars
        assert result.commit_message == commit_message
        assert result.created_at == created_at

    # Test that writing the same LMP again doesn't create a duplicate
    sql_store.write_lmp(serialized_lmp, uses)
    with Session(sql_store.engine) as session:
        count = session.execute(
            select(func.count()).select_from(SerializedLMP).where(SerializedLMP.lmp_id == lmp_id)
        ).scalar_one()
        assert count == 1