"""ELL module for language model interactions."""

from typing import Optional, Any, List, Union
from dataclasses import dataclass
import agentflow.ell2a as ell2a
from agentflow.ell2a.configurator import config
from agentflow.ell2a.types.message import Message, MessageRole, MessageType

def init(verbose: bool = False, autocommit_model: Optional[str] = None) -> None:
    """Initialize the ELL module.
    
    Args:
        verbose: Whether to enable verbose logging
        autocommit_model: The model to use for auto-commit messages
    """
    if autocommit_model:
        config.default_model = autocommit_model

def simple(model: str = "gpt-4o-mini", temperature: float = 0.7):
    """Decorator for simple language model functions.
    
    Args:
        model: The model to use
        temperature: The temperature parameter for generation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def system(message: str) -> Message:
    """Create a system message.
    
    Args:
        message: The message content
    """
    return Message(
        role=MessageRole.SYSTEM,
        content=message,
        type=MessageType.TEXT
    )

def user(message: str) -> Message:
    """Create a user message.
    
    Args:
        message: The message content
    """
    return Message(
        role=MessageRole.USER,
        content=message,
        type=MessageType.TEXT
    )

def assistant(message: str) -> Message:
    """Create an assistant message.
    
    Args:
        message: The message content
    """
    return Message(
        role=MessageRole.ASSISTANT,
        content=message,
        type=MessageType.TEXT
    ) 