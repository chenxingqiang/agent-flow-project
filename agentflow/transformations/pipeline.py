"""Transformation pipeline module for data processing."""
from typing import List, Any
import pandas as pd

class TransformationPipeline:
    """Pipeline for executing multiple transformation strategies."""
    
    def __init__(self):
        """Initialize transformation pipeline."""
        self.strategies = []
        
    def add_strategy(self, strategy: Any):
        """Add a transformation strategy to the pipeline.
        
        Args:
            strategy: Strategy object to add
        """
        self.strategies.append(strategy)
        
    def transform(self, data: Any) -> Any:
        """Execute all transformation strategies in sequence.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        result = data
        for strategy in self.strategies:
            transform_result = strategy.transform(result)
            
            # Handle different result formats
            if hasattr(transform_result, 'data'):
                if 'transformed_data' in transform_result.data:
                    result = transform_result.data['transformed_data']
                elif 'cleaned_data' in transform_result.data:
                    result = transform_result.data['cleaned_data']
                else:
                    result = transform_result.data
            else:
                result = transform_result
                
            # Convert result to DataFrame if it's a list
            if isinstance(result, list):
                if len(result) > 0:
                    if isinstance(result[0], (list, pd.Series, pd.DataFrame)):
                        result = pd.DataFrame(result)
                    else:
                        result = pd.DataFrame(result).T
                        
            # Add any additional columns from the transformation result
            if isinstance(result, pd.DataFrame) and hasattr(transform_result, 'data'):
                for key, value in transform_result.data.items():
                    if key not in ['transformed_data', 'cleaned_data'] and isinstance(value, list):
                        if len(value) == len(result):
                            result[key] = value
                            
        return result
