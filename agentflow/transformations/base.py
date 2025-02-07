"""Base classes for transformations."""
from typing import Dict, Any, Optional
from dataclasses import dataclass
import abc
from abc import ABC, abstractmethod

@dataclass
class TransformationResult:
    """Result of a transformation operation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class TransformationBase(abc.ABC):
    """Base class for all transformations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize transformation."""
        self.config = config
        
    @abc.abstractmethod
    def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """Transform data."""
        pass

class TransformationStrategy(ABC):
    """Base class for transformation strategies."""
    
    @abstractmethod
    def transform(self, data):
        """Transform the input data.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        pass

    def __init__(self, config: Dict[str, Any]):
        """Initialize transformation strategy."""
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.description = config.get('description', '')
        self.version = config.get('version', '1.0.0')
        
    def get_name(self) -> str:
        """Get strategy name."""
        return self.name
        
    def get_description(self) -> str:
        """Get strategy description."""
        return self.description
        
    def get_version(self) -> str:
        """Get strategy version."""
        return self.version 