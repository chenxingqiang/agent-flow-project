from enum import Enum

class NodeState(str, Enum):
    """Node state enum."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure" 