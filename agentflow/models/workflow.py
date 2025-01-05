from enum import Enum
from pydantic import BaseModel
from typing import Dict, Any, Optional

class StepType(str, Enum):
    """Valid step types for workflow steps."""
    RESEARCH = "research"
    DATA_SCIENCE = "data_science"
    TEXT_ANALYSIS = "text_analysis"
    FEATURE_ENGINEERING = "feature_engineering"
    OUTLIER_REMOVAL = "outlier_removal"
    GENERIC = "generic"

class WorkflowStep(BaseModel):
    """Model for a workflow step."""
    id: str
    name: str
    type: StepType  # This will validate against the enum values
    config: Optional[Dict[str, Any]] = {} 