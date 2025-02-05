"""Instruction result module."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class InstructionStatus(str, Enum):
    """Instruction status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class InstructionResult:
    """Instruction result class."""
    
    id: str
    status: InstructionStatus = InstructionStatus.SUCCESS
    content: Optional[str] = None
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        """Post initialization."""
        # Set timestamps if not provided
        if not self.start_time:
            self.start_time = datetime.now()
        if not self.end_time and self.status in [InstructionStatus.COMPLETED, InstructionStatus.FAILED, InstructionStatus.CANCELLED]:
            self.end_time = datetime.now()
        
        # Calculate duration if possible
        if self.start_time and self.end_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
            
        # Initialize metrics if not provided
        if not self.metrics:
            self.metrics = {
                "tokens": 0,
                "cost": 0.0,
                "latency": self.duration or 0.0
            }
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "id": self.id,
            "status": self.status.value,
            "content": self.content,
            "error": self.error,
            "context": self.context,
            "metrics": self.metrics,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstructionResult':
        """Create result from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            InstructionResult: Created result
        """
        # Convert string status to enum
        if isinstance(data.get("status"), str):
            data["status"] = InstructionStatus(data["status"])
            
        # Convert ISO format strings to datetime
        if data.get("start_time"):
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            data["end_time"] = datetime.fromisoformat(data["end_time"])
            
        return cls(**data) 