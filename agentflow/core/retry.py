import time
from typing import Callable, Any, Optional
from functools import wraps

class RetryConfig:
    def __init__(self, 
                 max_retries: int = 3,
                 delay: float = 1.0,
                 backoff_factor: float = 2.0,
                 exceptions: tuple = (Exception,)):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions

def with_retry(retry_config: Optional[RetryConfig] = None):
    """Retry decorator for workflow steps"""
    if retry_config is None:
        retry_config = RetryConfig()
        
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = retry_config.delay
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_config.exceptions as e:
                    last_exception = e
                    if attempt < retry_config.max_retries:
                        time.sleep(delay)
                        delay *= retry_config.backoff_factor
                    else:
                        raise last_exception
                        
        return wrapper
    return decorator 