"""ELL2A integration module."""

from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from ..ell2a import ELL, Message, MessageRole

class ELL2AIntegration:
    """Singleton class for ELL2A integration."""
    
    _instance = None
    
    def __new__(cls):
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super(ELL2AIntegration, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize ELL2A integration."""
        self.config = {
            "default_model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "simple": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "timeout": 30.0
            },
            "complex": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "timeout": 60.0,
                "stream": True,
                "track_performance": True,
                "track_memory": True
            }
        }
        self.ell = ELL(model_name=self.config["default_model"], config=self.config)
        self._is_initialized = True
        self._workflows = {}
        self._messages = []
        self._metrics = {}
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure ELL2A integration.
        
        Args:
            config: Configuration parameters to update
        """
        if isinstance(config, dict):
            self.config.update(config)
            # Reinitialize ELL with new configuration
            self.ell = ELL(model_name=self.config.get("default_model", "gpt-4"), config=self.config)
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration.
        
        Args:
            new_config: New configuration parameters
        """
        self.configure(new_config)
    
    async def process_message(self, message: Message) -> Message:
        """Process a message using ELL2A.
        
        Args:
            message: Message to process
            
        Returns:
            Processed message
        """
        return await self.ell.process_message(message)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration.
        
        Returns:
            Current configuration
        """
        return self.config.copy()
        
    def get_mode_config(self, mode: str = "simple") -> Dict[str, Any]:
        """Get configuration for specified mode.
        
        Args:
            mode: Configuration mode (simple or complex)
            
        Returns:
            Mode-specific configuration
        """
        base_config = {
            "model": self.config["default_model"],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"]
        }
        mode_config = self.config.get(mode, {})
        return {**base_config, **mode_config}
    
    def create_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a message.
        
        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created message
        """
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        return message
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add message to context.
        
        Args:
            message: Message to add
        """
        self._messages.append(message)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages.
        
        Returns:
            List of messages
        """
        return self._messages.copy()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context.
        
        Returns:
            Current context
        """
        return {
            "messages": self._messages.copy()
        }
    
    def clear_context(self) -> None:
        """Clear current context."""
        self._messages.clear()
    
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
            workflow_id: Workflow ID
        """
        self._workflows.pop(workflow_id, None)
    
    def list_workflows(self) -> List[str]:
        """List registered workflows.
        
        Returns:
            List of workflow IDs
        """
        return list(self._workflows.keys())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics.
        
        Returns:
            Current metrics
        """
        return self._metrics.copy()
    
    @staticmethod
    def with_ell2a(mode: str = "simple"):
        """Decorator to enable ELL2A for any function.
        
        Args:
            mode: Processing mode (simple or complex)
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                ell2a = ELL2AIntegration()
                config = ell2a.get_mode_config(mode)
                
                # Create message from function call
                message = Message(
                    role=MessageRole.USER,
                    content=str(args[0]) if args else "",
                    metadata={
                        "mode": mode,
                        "function": func.__name__,
                        **kwargs
                    }
                )
                
                # Process with ELL2A
                result = await ell2a.process_message(message)
                return {"result": result.content}
                
            return wrapper
        return decorator
    
    @staticmethod
    def track_function():
        """Decorator to track function execution with ELL2A.
        
        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                ell2a = ELL2AIntegration()
                func_name = func.__name__
                
                # Initialize metrics for function if not exists
                if func_name not in ell2a._metrics:
                    ell2a._metrics[func_name] = {
                        "calls": 0,
                        "total_time": 0.0,
                        "errors": 0
                    }
                
                try:
                    # Track function execution
                    import time
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    ell2a._metrics[func_name]["calls"] += 1
                    ell2a._metrics[func_name]["total_time"] += time.time() - start_time
                    return result
                except Exception as e:
                    ell2a._metrics[func_name]["errors"] += 1
                    raise
                
            return wrapper
        return decorator

# Global instance
ell2a_integration = ELL2AIntegration() 