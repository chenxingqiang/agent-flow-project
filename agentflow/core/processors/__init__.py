"""
Processor nodes for AgentFlow
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class ProcessorResult(BaseModel):
    """Result from processor execution"""
    output: Dict[str, Any]
    metadata: Dict[str, str] = {}
    error: Optional[str] = None

class BaseProcessor(ABC):
    """Base class for all processors"""
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> ProcessorResult:
        """Process input data
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing result
        """
        pass
        
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
        
    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input data schema
        
        Returns:
            JSON schema for input data
        """
        pass
        
    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """Get output data schema
        
        Returns:
            JSON schema for output data
        """
        pass
