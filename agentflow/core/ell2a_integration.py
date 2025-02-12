"""ELL2A integration module."""

from typing import Dict, Any, Optional, List, Callable
from functools import wraps
import time
from ..ell2a import ELL, Message, MessageRole, MessageType

class ELL2AIntegration:
    """ELL2A integration class."""
    
    _instance = None
    
    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize ELL2A integration."""
        if not self._initialized:
            self.enabled = True
            self.tracking_enabled = True
            self.config = {}
            self.metrics = {
                "function_calls": 0,
                "total_execution_time": 0.0,
                "errors": 0,
                "function_metrics": {}
            }
            self._messages = []
            self._workflows = {}
            self._initialized = True
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure ELL2A integration.
        
        Args:
            config: Configuration dictionary
        """
        self.enabled = config.get("enabled", True)
        self.tracking_enabled = config.get("tracking_enabled", True)
        self.config = config
    
    async def process_message(self, message: Message) -> Message:
        """Process a message.
        
        Args:
            message: Message to process
            
        Returns:
            Message: Response message
        """
        try:
            start_time = time.time()
            
            # If disabled, return a copy of the input message
            if not self.enabled:
                return Message(
                    role=message.role,  # Preserve the original role
                    content=message.content,
                    type=message.type,
                    metadata=message.metadata
                )
            
            # Update metrics
            if self.tracking_enabled:
                self.metrics["function_calls"] += 1
            
            # Store message
            self._messages.append(message)
            
            # Create response message
            model_name = "default"
            if isinstance(self.config, dict):
                model_config = self.config.get("model", {})
                if isinstance(model_config, dict):
                    model_name = model_config.get("name", "default")
            
            response = Message(
                role=MessageRole.ASSISTANT,
                content=message.content,
                type=MessageType.RESULT,  # Set type to RESULT
                metadata={
                    "model": model_name,
                    "timestamp": time.time(),
                    "type": MessageType.RESULT,
                    "status": "success"
                }
            )
            
            return response
            
        except Exception as e:
            if self.tracking_enabled:
                self.metrics["errors"] += 1
            raise
            
        finally:
            if self.tracking_enabled:
                self.metrics["total_execution_time"] += time.time() - start_time
    
    def cleanup(self) -> None:
        """Clean up ELL2A integration."""
        self.metrics = {
            "function_calls": 0,
            "total_execution_time": 0.0,
            "errors": 0,
            "function_metrics": {}
        }
        self.config = {}
        self._messages = []
        self._workflows = {}

    def register_workflow(self, workflow_id: str, workflow: Any) -> None:
        """Register a workflow.
        
        Args:
            workflow_id: Workflow ID
            workflow: Workflow instance
        """
        self._workflows[workflow_id] = workflow

    def unregister_workflow(self, workflow_id: str) -> None:
        """Unregister a workflow.
        
        Args:
            workflow_id: Workflow ID to unregister
        """
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]

    def list_workflows(self) -> List[str]:
        """List all registered workflows.
        
        Returns:
            List[str]: List of workflow IDs
        """
        return list(self._workflows.keys())

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics.
        
        Returns:
            Dict[str, Any]: Metrics dictionary
        """
        # Merge function metrics into top level
        metrics = {
            "function_calls": self.metrics["function_calls"],
            "total_execution_time": self.metrics["total_execution_time"],
            "errors": self.metrics["errors"]
        }
        # Add function-specific metrics at top level
        for func_name, func_metrics in self.metrics["function_metrics"].items():
            metrics[func_name] = func_metrics
        return metrics

    def get_mode_config(self, mode: str) -> Dict[str, Any]:
        """Get mode configuration.
        
        Args:
            mode: Mode name
            
        Returns:
            Dict[str, Any]: Mode configuration
        """
        return self.config.get(mode, {})

    def get_messages(self) -> List[Message]:
        """Get all messages.
        
        Returns:
            List[Message]: List of messages
        """
        return self._messages

    @staticmethod
    def track_function():
        """Decorator to track function execution metrics."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                instance = ell2a_integration
                if not instance.tracking_enabled:
                    return await func(*args, **kwargs)
                
                start_time = time.time()
                try:
                    # Update function call count
                    func_name = func.__name__
                    if func_name not in instance.metrics["function_metrics"]:
                        instance.metrics["function_metrics"][func_name] = {
                            "calls": 0,
                            "total_time": 0.0,
                            "errors": 0
                        }
                    instance.metrics["function_metrics"][func_name]["calls"] += 1
                    
                    # Execute function
                    result = await func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    # Update error count
                    instance.metrics["function_metrics"][func_name]["errors"] += 1
                    raise
                    
                finally:
                    # Update execution time
                    execution_time = time.time() - start_time
                    instance.metrics["function_metrics"][func_name]["total_time"] += execution_time
            
            return wrapper
        return decorator

    @staticmethod
    def with_ell2a(mode: str = "simple"):
        """Decorator to wrap function with ELL2A integration.
        
        Args:
            mode: Integration mode (simple or complex)
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                instance = ell2a_integration
                if not instance.enabled:
                    return await func(*args, **kwargs)
                
                # Get mode config
                mode_config = instance.config.get(mode, {})
                
                start_time = time.time()
                try:
                    # Update function call count
                    func_name = func.__name__
                    if func_name not in instance.metrics["function_metrics"]:
                        instance.metrics["function_metrics"][func_name] = {
                            "calls": 0,
                            "total_time": 0.0,
                            "errors": 0
                        }
                    instance.metrics["function_metrics"][func_name]["calls"] += 1
                    
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # Update metrics
                    if instance.tracking_enabled:
                        instance.metrics["function_calls"] += 1
                        instance.metrics["total_execution_time"] += time.time() - start_time
                    
                    return result
                    
                except Exception as e:
                    if instance.tracking_enabled:
                        instance.metrics["errors"] += 1
                        # Update function error count
                        instance.metrics["function_metrics"][func_name]["errors"] += 1
                    raise
            
            return wrapper
        return decorator

# Global instance
ell2a_integration = ELL2AIntegration() 