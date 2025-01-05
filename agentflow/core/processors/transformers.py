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
        self.config = config
        self.strategy = config.get("strategy", "")
        self.params = config.get("params", {})
        self.transformations = config.get("transformations", {})
        self.transform_methods = {
            "FILTER": self._filter_output,
            "MAP": self._map_output,
            "feature_engineering": self._feature_engineering,
            "outlier_removal": self._outlier_removal
        }

    async def process(self, data: Dict[str, Any]) -> ProcessorResult:
        """Process data using the configured transformations."""
        try:
            # Validate input
            if not self.validate_input(data):
                return ProcessorResult(data={}, metadata={}, error="Invalid input data")

            # Process data based on strategy
            if self.strategy in self.transform_methods:
                result = await self.transform_methods[self.strategy](data)
                return ProcessorResult(data=result, metadata={"strategy": self.strategy})
            else:
                error_msg = f"Unsupported transformation strategy: {self.strategy}"
                logger.error(error_msg)
                return ProcessorResult(data={}, metadata={}, error=error_msg)
            
        except Exception as e:
            error_msg = f"Transformation failed: {str(e)}"
            logger.error(error_msg)
            return ProcessorResult(data={}, metadata={}, error=error_msg)

    async def process_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Process input data and apply transformations."""
        if isinstance(data, list):
            return [self._apply_transformations(item) for item in data]
        return self._apply_transformations(data)

    def _apply_transformations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformations to data."""
        result = {}
        for output_field, expression in self.transformations.items():
            try:
                if isinstance(expression, str):
                    if expression.startswith("$input."):
                        # Handle dot notation for nested fields
                        path = expression.replace("$input.", "").split(".")
                        value = data
                        for key in path:
                            value = value.get(key)
                            if value is None:
                                break
                        result[output_field] = value
                    elif "input[" in expression:
                        # Handle Python-style expressions
                        input_data = data
                        result[output_field] = eval(expression, {"input": input_data})
                    else:
                        result[output_field] = expression
                else:
                    result[output_field] = expression
            except Exception as e:
                result[output_field] = None
        
        return result

    async def _feature_engineering(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply feature engineering transformations."""
        try:
            input_data = data.get("data")
            if not isinstance(input_data, np.ndarray):
                raise ValueError("Input data must be a numpy array")

            method = self.params.get("method", "standard")
            if method == "standard":
                scaler = StandardScaler(
                    with_mean=self.params.get("with_mean", True),
                    with_std=self.params.get("with_std", True)
                )
                transformed_data = scaler.fit_transform(input_data)
                return {"data": transformed_data}
            else:
                raise ValueError(f"Unsupported feature engineering method: {method}")
        except Exception as e:
            raise ValueError(f"Feature engineering failed: {str(e)}")

    async def _outlier_removal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply outlier removal transformations."""
        try:
            input_data = data.get("data")
            if not isinstance(input_data, np.ndarray):
                raise ValueError("Input data must be a numpy array")

            method = self.params.get("method", "isolation_forest")
            if method == "isolation_forest":
                detector = IsolationForest(
                    contamination=self.params.get("threshold", 0.1),
                    random_state=42
                )
                predictions = detector.fit_predict(input_data)
                filtered_data = input_data[predictions == 1]
                return {"data": filtered_data}
            else:
                raise ValueError(f"Unsupported outlier removal method: {method}")
        except Exception as e:
            raise ValueError(f"Outlier removal failed: {str(e)}")

    async def _filter_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter output data."""
        try:
            input_data = data.get("data")
            if not isinstance(input_data, np.ndarray):
                raise ValueError("Input data must be a numpy array")

            conditions = self.params.get("conditions", [])
            filtered_data = input_data
            for condition in conditions:
                field = condition.get("field")
                operator = condition.get("operator")
                value = condition.get("value")
                if field is not None and operator is not None and value is not None:
                    mask = self._apply_operator(operator, filtered_data[:, field], value)
                    filtered_data = filtered_data[mask]
            return {"data": filtered_data}
        except Exception as e:
            raise ValueError(f"Filter failed: {str(e)}")

    async def _map_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Map output data."""
        try:
            input_data = data.get("data")
            if not isinstance(input_data, np.ndarray):
                raise ValueError("Input data must be a numpy array")

            mapping = self.params.get("mapping", {})
            mapped_data = np.copy(input_data)
            for field, value_map in mapping.items():
                if isinstance(field, int) and field < input_data.shape[1]:
                    for old_val, new_val in value_map.items():
                        mapped_data[:, field][mapped_data[:, field] == old_val] = new_val
            return {"data": mapped_data}
        except Exception as e:
            raise ValueError(f"Mapping failed: {str(e)}")

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
                "data": {"type": "object"},
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
        self._groups: Dict[str, List[Dict[str, Any]]] = {}

    async def process(self, data: Dict[str, Any]) -> ProcessorResult:
        """Process data using the configured aggregations."""
        try:
            # Add data to groups
            group_key = str(data.get(self.group_by, "default"))
            if group_key not in self._groups:
                self._groups[group_key] = []
            self._groups[group_key].append(data)
            
            # Aggregate all groups
            result = {}
            for group_key, group_data in self._groups.items():
                result[group_key] = self._aggregate_group(group_data)
            
            # Create metadata
            metadata = {
                "group_count": str(len(self._groups)),
                "total_records": str(sum(len(group) for group in self._groups.values()))
            }
            
            return ProcessorResult(data=result, metadata=metadata)
        except Exception as e:
            return ProcessorResult(data={}, metadata={}, error=str(e))

    async def process_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Process input data and apply aggregations."""
        # Reset groups
        self._groups = {}
        
        # Process data
        if isinstance(data, list):
            for item in data:
                await self.process(item)
        else:
            await self.process(data)
        
        # Return aggregated results
        result = {}
        for group_key, group_data in self._groups.items():
            result[group_key] = self._aggregate_group(group_data)
        return result

    def _aggregate_group(self, group_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data within a group.
        
        Args:
            group_data: List of data points in the group
            
        Returns:
            Dict[str, Any]: Aggregated results
        """
        result = {}
        
        for field, agg_type in self.aggregations.items():
            try:
                # Extract values for the field
                values = [item.get(field) for item in group_data if item.get(field) is not None]
                
                if not values:
                    result[field] = None
                    continue
                
                # Apply aggregation
                if agg_type == "sum":
                    result[field] = sum(values)
                elif agg_type == "avg":
                    result[field] = sum(values) / len(values)
                elif agg_type == "min":
                    result[field] = min(values)
                elif agg_type == "max":
                    result[field] = max(values)
                elif agg_type == "count":
                    result[field] = len(values)
                elif agg_type == "first":
                    result[field] = values[0]
                elif agg_type == "last":
                    result[field] = values[-1]
                elif agg_type == "list":
                    result[field] = values
                elif agg_type == "set":
                    result[field] = list(set(values))
                elif agg_type == "concat":
                    result[field] = ",".join(str(v) for v in values)
                elif agg_type == "std":
                    result[field] = float(np.std(values))
                elif agg_type == "var":
                    result[field] = float(np.var(values))
                elif agg_type == "median":
                    result[field] = float(np.median(values))
                else:
                    result[field] = None
                    
            except Exception as e:
                result[field] = None
        
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
