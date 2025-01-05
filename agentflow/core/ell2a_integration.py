"""ELL2A integration module."""

from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from ..ell2a import ELL, Message

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
                "track_performance": True
            }
        }
        self.ell = ELL(model_name=self.config["default_model"], config=self.config)
        self._is_initialized = True
    
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
                    role="user",
                    content=str(args[0]) if args else "",
                    metadata={
                        "mode": mode,
                        "function": func.__name__,
                        **kwargs
                    }
                )
                
                # Process with ELL2A
                result = await ell2a.process_message(message)
                return result
                
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
                
                # Track function execution
                message = Message(
                    role="function",
                    content=f"Executing {func.__name__}",
                    metadata={
                        "function": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                )
                
                try:
                    result = await func(*args, **kwargs)
                    message.metadata["status"] = "success"
                    message.metadata["result"] = str(result)
                except Exception as e:
                    message.metadata["status"] = "error"
                    message.metadata["error"] = str(e)
                    raise
                finally:
                    await ell2a.process_message(message)
                
                return result
                
            return wrapper
        return decorator

# Global instance
ell2a_integration = ELL2AIntegration() 