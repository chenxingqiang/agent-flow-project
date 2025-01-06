"""Transform processor module."""

import numpy as np
from typing import Dict, Any, Union, Callable, List
from dataclasses import dataclass
from ..workflow_types import StepConfig
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessorResult:
    """Processor result."""
    data: Dict[str, Any]
    metadata: Dict[str, str] = None
    error: str = None

class TransformationPipeline:
    """Pipeline for executing a sequence of transformations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize transformation pipeline.
        
        Args:
            config: Pipeline configuration containing processors and their configs
        """
        self.processors = []
        self.processor_map = {
            "filter": FilterProcessor,
            "transform": TransformProcessor,
            "aggregate": AggregateProcessor
        }
        
        # Initialize processors from config
        for processor_config in config.get("processors", []):
            processor_type = processor_config.get("type")
            if processor_type in self.processor_map:
                processor_class = self.processor_map[processor_type]
                processor = processor_class(processor_config.get("config", {}))
                self.processors.append(processor)
    
    async def process(self, data: Dict[str, Any]) -> ProcessorResult:
        """Process data through the pipeline.
        
        Args:
            data: Input data to process
            
        Returns:
            ProcessorResult: Result of pipeline execution
        """
        current_data = data
        metadata = {}
        
        try:
            for processor in self.processors:
                result = await processor.process(current_data)
                
                if result.error:
                    return ProcessorResult(
                        data={},
                        metadata=metadata,
                        error=f"Pipeline error in {processor.__class__.__name__}: {result.error}"
                    )
                
                current_data = result.data
                if result.metadata:
                    metadata.update(result.metadata)
            
            return ProcessorResult(data=current_data, metadata=metadata)
            
        except Exception as e:
            return ProcessorResult(
                data={},
                metadata=metadata,
                error=f"Pipeline execution error: {str(e)}"
            )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate pipeline configuration.
        
        Args:
            config: Pipeline configuration to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
            
        processors = config.get("processors", [])
        if not isinstance(processors, list):
            return False
            
        for processor_config in processors:
            if not isinstance(processor_config, dict):
                return False
                
            processor_type = processor_config.get("type")
            if processor_type not in self.processor_map:
                return False
                
            if not isinstance(processor_config.get("config", {}), dict):
                return False
        
        return True

class FilterProcessor:
    """Filter processor class."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize filter processor."""
        self.conditions = config.get("conditions", [])

    async def process(self, data: Dict[str, Any]) -> ProcessorResult:
        """Process data using the configured filters."""
        try:
            # Process data
            result = await self.process_data(data)
            
            # Determine if data was filtered
            filtered = False
            if isinstance(data, dict):
                filtered = not self._check_conditions(data)
            elif isinstance(data, list):
                filtered = len(result) < len(data)
            
            # Create metadata
            metadata = {"filtered": str(filtered).lower()}
            
            return ProcessorResult(data=result, metadata=metadata)
        except Exception as e:
            return ProcessorResult(data={}, metadata={}, error=str(e))

    async def process_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Process input data and apply filters."""
        if isinstance(data, list):
            return [item for item in data if self._check_conditions(item)]
        return data if self._check_conditions(data) else {}

    def _check_conditions(self, data: Dict[str, Any]) -> bool:
        """Check if data matches all conditions."""
        for condition in self.conditions:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            
            if not all([field, operator, value is not None]):
                continue
            
            data_value = data.get(field)
            if data_value is None:
                return False
            
            if not self._apply_operator(operator, data_value, value):
                return False
        
        return True

    def _apply_operator(self, operator: str, data_value: Any, condition_value: Any) -> bool:
        """Apply operator to compare values."""
        if operator == "eq":
            return data_value == condition_value
        elif operator == "gt":
            return data_value > condition_value
        elif operator == "lt":
            return data_value < condition_value
        elif operator == "gte":
            return data_value >= condition_value
        elif operator == "lte":
            return data_value <= condition_value
        elif operator == "contains":
            return str(condition_value).lower() in str(data_value).lower()
        elif operator == "startswith":
            return str(data_value).startswith(str(condition_value))
        elif operator == "endswith":
            return str(data_value).endswith(str(condition_value))
        else:
            raise ValueError(f"Unsupported operator: {operator}")

class TransformProcessor:
    """Transform processor class."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize transform processor."""
        self.transformations = config.get("transformations", {})

    async def process(self, data: Dict[str, Any]) -> ProcessorResult:
        """Process data using the configured transformations."""
        try:
            # Transform the data
            transformed_data = {}
            transformed_fields = []
            
            for output_field, input_path in self.transformations.items():
                # Remove the '$input.' prefix
                path = input_path.replace("$input.", "")
                value = self._get_nested_value(data, path.split("."))
                
                if value is not None:
                    transformed_data[output_field] = value
                    transformed_fields.append(output_field)
            
            return ProcessorResult(
                output=transformed_data,
                metadata={"transformed_fields": transformed_fields}
            )
            
        except Exception as e:
            return ProcessorResult(
                output={},
                metadata={"error": str(e)},
                error=str(e)
            )

    def _get_nested_value(self, data: Dict[str, Any], path: List[str]) -> Any:
        """Get value from nested dictionary using path list."""
        current = data
        for key in path:
            if isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return None
            else:
                return None
        return current

    def validate_input(self, data: Any) -> bool:
        """Validate input data."""
        if not isinstance(data, dict):
            return False
        if "data" not in data:
            return False
        input_data = data.get("data")
        if not isinstance(input_data, np.ndarray):
            return False
        return True

    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema.
        
        Returns:
            Dict[str, Any]: JSON schema for input data
        """
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": ["object", "array"],
                    "items": {"type": "object"}
                }
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema.
        
        Returns:
            Dict[str, Any]: JSON schema for output data
        """
        return {
            "type": "object",
            "properties": {
                "output": {"type": "object"},
                "metadata": {"type": "object"},
                "error": {"type": ["string", "null"]}
            }
        }

class AggregateProcessor:
    """Aggregate processor class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize aggregate processor."""
        self.group_by = config.get("group_by")
        self.aggregations = config.get("aggregations", {})
    
    async def process(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> ProcessorResult:
        """Process data using the configured aggregations."""
        try:
            if isinstance(data, dict):
                data = [data]
                
            if not self.validate_input(data):
                raise ValueError("Invalid input data format")
            
            result = await self.process_data(data)
            
            metadata = {
                "group_count": str(len(result)),
                "total_records": str(len(data))
            }
            
            return ProcessorResult(
                output=result,
                metadata=metadata
            )
        except Exception as e:
            return ProcessorResult(
                output={},
                metadata={},
                error=str(e)
            )
    
    async def process_data(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Process input data and apply aggregations."""
        groups = {}
        
        # Group the data
        for item in data:
            group_key = str(item.get(self.group_by, "default"))
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
        
        # Calculate aggregations for each group
        result = {}
        for group_key, group_data in groups.items():
            result[group_key] = {}
            for agg_name, agg_config in self.aggregations.items():
                field = agg_config.get("field")
                agg_type = agg_config.get("type")
                
                values = [item.get(field) for item in group_data if item.get(field) is not None]
                
                if not values:
                    continue
                    
                if agg_type == "sum":
                    result[group_key][agg_name] = sum(values)
                elif agg_type == "avg":
                    result[group_key][agg_name] = sum(values) / len(values)
                elif agg_type == "count":
                    result[group_key][agg_name] = len(values)
                
        return result

    def validate_input(self, data: Any) -> bool:
        """Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(data, (dict, list)):
            return False
        if isinstance(data, list):
            return all(isinstance(item, dict) for item in data)
        return True

    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema.
        
        Returns:
            Dict[str, Any]: JSON schema for input data
        """
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": ["object", "array"],
                    "items": {"type": "object"}
                }
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema.
        
        Returns:
            Dict[str, Any]: JSON schema for output data
        """
        return {
            "type": "object",
            "properties": {
                "output": {"type": "object"},
                "metadata": {"type": "object"},
                "error": {"type": ["string", "null"]}
            }
        }
