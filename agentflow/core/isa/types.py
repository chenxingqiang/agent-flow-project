"""ISA types module."""
from enum import Enum
from typing import Dict, Any, List
from pydantic import BaseModel

class AgentType(str, Enum):
    """Agent type."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    REVIEW = "review"
    CUSTOM = "custom"

class InstructionType(str, Enum):
    """Instruction type."""
    RESEARCH = "research"
    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    CUSTOM = "custom"

class AnalysisType(str, Enum):
    """Analysis type."""
    BEHAVIOR = "behavior"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CUSTOM = "custom"

class PatternType(str, Enum):
    """Pattern type."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BEHAVIORAL = "behavioral"
    CUSTOM = "custom"

class AnalysisResult(BaseModel):
    """Analysis result."""
    type: AnalysisType
    metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    confidence: float
