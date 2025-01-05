"""Base executor module."""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from .workflow_types import WorkflowConfig

class BaseExecutor(ABC):
    """Base executor class."""

    def __init__(self, config: WorkflowConfig):
        """Initialize base executor.
        
        Args:
            config: Workflow configuration.
        """
        self.config = config

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow step.
        
        Args:
            context: Execution context.
            
        Returns:
            Dict[str, Any]: Execution result.
        """
        pass
