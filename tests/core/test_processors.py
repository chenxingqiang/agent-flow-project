"""
Tests for processor functionality
"""

import pytest
from agentflow.core.processors.transformers import (
    FilterProcessor,
    TransformProcessor,
    AggregateProcessor
)

@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return {
        "id": 1,
        "name": "Test",
        "value": 100,
        "nested": {
            "field": "test"
        }
    }

@pytest.mark.asyncio
async def test_filter_processor():
    """Test FilterProcessor functionality"""
    # Test equals condition
    processor = FilterProcessor({
        "conditions": [
            {
                "field": "value",
                "operator": "eq",
                "value": 100
            }
        ]
    })
    
    result = await processor.process({
        "value": 100
    })
    assert "filtered" in result.metadata
    assert result.metadata["filtered"] == "false"
    
    result = await processor.process({
        "value": 200
    })
    assert result.metadata["filtered"] == "true"
    
    # Test multiple conditions
    processor = FilterProcessor({
        "conditions": [
            {
                "field": "value",
                "operator": "gt",
                "value": 50
            },
            {
                "field": "name",
                "operator": "contains",
                "value": "test"
            }
        ]
    })
    
    result = await processor.process({
        "value": 100,
        "name": "test_name"
    })
    assert result.metadata["filtered"] == "false"

@pytest.mark.asyncio
async def test_filter_processor_process_data():
    """Test FilterProcessor process_data method"""
    processor = FilterProcessor({
        "conditions": [
            {
                "field": "value",
                "operator": "gt",
                "value": 50
            }
        ]
    })
    
    # Test single dict filtering
    single_result = await processor.process_data({"value": 100})
    assert single_result == {"value": 100}
    
    # Test list filtering
    list_input = [
        {"value": 30},
        {"value": 60},
        {"value": 40},
        {"value": 70}
    ]
    list_result = await processor.process_data(list_input)
    assert len(list_result) == 2
    assert all(item["value"] > 50 for item in list_result)

@pytest.mark.asyncio
async def test_transform_processor():
    """Test TransformProcessor functionality"""
    processor = TransformProcessor({
        "transformations": {
            "id": "$input.id",
            "full_name": "$input.name",
            "nested_value": "$input.nested.field"
        }
    })
    
    result = await processor.process({
        "id": 1,
        "name": "Test",
        "nested": {
            "field": "test_value"
        }
    })
    
    # Check transformed data
    assert result.output["id"] == 1
    assert result.output["full_name"] == "Test"
    assert result.output["nested_value"] == "test_value"
    
    # Check metadata
    assert "transformed_fields" in result.metadata
    assert set(result.metadata["transformed_fields"]) == {"id", "full_name", "nested_value"}

@pytest.mark.asyncio
async def test_transform_processor_process_data():
    """Test TransformProcessor process_data method"""
    processor = TransformProcessor({
        "transformations": {
            "doubled_value": "$input.value",
            "name_upper": "$input.name"
        }
    })
    
    # Test single dict transformation
    single_result = await processor.process_data({
        "value": 10,
        "name": "test"
    })
    
    # Apply transformations after getting values
    assert single_result["doubled_value"] == 10 * 2
    assert single_result["name_upper"] == "TEST"
    
    # Test list transformation
    list_input = [
        {"value": 10, "name": "test1"},
        {"value": 20, "name": "test2"}
    ]
    list_result = await processor.process_data(list_input)
    assert len(list_result) == 2
    assert list_result[0]["doubled_value"] == 20
    assert list_result[1]["name_upper"] == "TEST2"

@pytest.mark.asyncio
async def test_aggregate_processor():
    """Test AggregateProcessor functionality"""
    processor = AggregateProcessor({
        "group_by": "category",
        "aggregations": {
            "total": {"type": "sum", "field": "value"},
            "average": {"type": "avg", "field": "value"},
            "count": {"type": "count", "field": "value"}
        }
    })
    
    input_data = [
        {"category": "A", "value": 100},
        {"category": "A", "value": 200},
        {"category": "B", "value": 300},
        {"category": "B", "value": 400}
    ]
    
    result = await processor.process(input_data)
    
    # Verify aggregated data
    assert "A" in result.data
    assert "B" in result.data
    
    # Check group A aggregations
    assert result.data["A"]["total"] == 300
    assert result.data["A"]["average"] == 150
    assert result.data["A"]["count"] == 2
    
    # Check group B aggregations
    assert result.data["B"]["total"] == 700
    assert result.data["B"]["average"] == 350
    assert result.data["B"]["count"] == 2
    
    # Verify metadata
    assert "group_count" in result.metadata
    assert result.metadata["group_count"] == "2"

@pytest.mark.asyncio
async def test_aggregate_processor_process_data():
    """Test AggregateProcessor process_data method"""
    processor = AggregateProcessor({
        "group_by": "category",
        "aggregations": {
            "total_value": {
                "type": "sum",
                "field": "value"
            },
            "avg_value": {
                "type": "avg",
                "field": "value"
            }
        }
    })
    
    # Test list input
    list_input = [
        {"category": "A", "value": 100},
        {"category": "A", "value": 200},
        {"category": "B", "value": 300},
        {"category": "B", "value": 400}
    ]
    
    result = await processor.process_data(list_input)
    
    # Check group A aggregations
    assert "A" in result
    assert result["A"]["total_value"] == 300
    assert result["A"]["avg_value"] == 150
    
    # Check group B aggregations
    assert "B" in result
    assert result["B"]["total_value"] == 700
    assert result["B"]["avg_value"] == 350

@pytest.mark.asyncio
async def test_processor_error_handling():
    """Test processor error handling"""
    # Test with invalid input
    processor = FilterProcessor({
        "conditions": [
            {
                "field": "invalid.field",
                "operator": "eq",
                "value": 100
            }
        ]
    })
    
    result = await processor.process({})
    assert result.error is not None
    
    # Test with invalid operator
    processor = FilterProcessor({
        "conditions": [
            {
                "field": "value",
                "operator": "invalid",
                "value": 100
            }
        ]
    })
    
    result = await processor.process({"value": 100})
    assert result.error is not None

@pytest.mark.asyncio
async def test_processor_validation():
    """Test processor input validation"""
    processor = TransformProcessor({
        "transformations": {
            "output": "input"
        }
    })
    
    # Test valid input
    assert processor.validate_input({"input": "test"})
    
    # Test invalid input
    assert not processor.validate_input("invalid")
    assert not processor.validate_input(123)
    
    # Test schema generation
    input_schema = processor.get_input_schema()
    assert input_schema["type"] == "object"
    
    output_schema = processor.get_output_schema()
    assert output_schema["type"] == "object"
    assert "properties" in output_schema

@pytest.mark.asyncio
async def test_transform_processor_error_handling():
    """Test TransformProcessor error handling"""
    processor = TransformProcessor({
        "transformations": {
            "missing_field": "$input.does_not_exist",
            "nested_missing": "$input.nested.missing"
        }
    })
    
    result = await processor.process({
        "id": 1,
        "name": "Test"
    })
    
    # Check that missing fields are handled gracefully
    assert "missing_field" not in result.output
    assert "nested_missing" not in result.output
    assert "transformed_fields" in result.metadata
    assert len(result.metadata["transformed_fields"]) == 0
