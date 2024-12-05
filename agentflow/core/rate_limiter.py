import time
import logging
from typing import Optional, Callable, Any
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from agentflow.config import config

logger = logging.getLogger(__name__)

class RateLimitError(Exception):
    """Exception raised when rate limit is hit"""
    pass

def rate_limit_handler(details):
    """Handler for rate limit backoff"""
    logger.warning(f"Rate limit hit. Backing off for {details['wait']} seconds.")
    time.sleep(details['wait'])

@backoff.on_exception(
    backoff.expo,
    RateLimitError,
    max_tries=5,
    on_backoff=rate_limit_handler
)
def with_rate_limit(func: Callable, *args, **kwargs) -> Any:
    """Wrapper to handle rate limits with exponential backoff"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if "rate limit" in str(e).lower():
            raise RateLimitError(str(e))
        raise

class ModelRateLimiter:
    """Rate limiter for model API calls"""
    
    def __init__(self):
        rate_limits = config.get_rate_limits()
        self.max_retries = int(rate_limits.get('max_retries', 3))
        self.retry_delay = int(rate_limits.get('retry_delay', 1))
        self.logger = logging.getLogger(__name__)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=10),
        reraise=True,
        retry=retry_if_exception_type((ValueError, RateLimitError))
    )
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            raise
