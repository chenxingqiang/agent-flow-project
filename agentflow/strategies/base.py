"""Base transformation strategy."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, Optional

class TransformationStrategy(ABC):
    """Base class for transformation strategies."""
    
    def __init__(self):
        """Initialize transformation strategy."""
        pass
        
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        pass
        
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if input is valid, False otherwise
        """
        return isinstance(data, pd.DataFrame) and not data.empty
        
    def validate_output(self, data: pd.DataFrame) -> bool:
        """Validate output data.
        
        Args:
            data: Output DataFrame
            
        Returns:
            True if output is valid, False otherwise
        """
        return isinstance(data, pd.DataFrame) and not data.empty
