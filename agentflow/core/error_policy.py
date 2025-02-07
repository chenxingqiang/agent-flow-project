"""Error policy configuration."""

from typing import Optional
from pydantic import BaseModel, Field

from .retry_policy import RetryPolicy


class ErrorPolicy(BaseModel):
    """Error policy configuration."""

    fail_fast: bool = True
    ignore_warnings: bool = False
    max_errors: int = Field(default=10, ge=1)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    ignore_validation_error: bool = False 