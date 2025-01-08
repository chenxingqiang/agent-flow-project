from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ProcessorResult:
    """Processor result."""
    data: Dict[str, Any]
    metadata: Dict[str, str] = None
    error: str = None
    output: Dict[str, Any] = None

    def __post_init__(self):
        """Ensure backward compatibility."""
        if self.output is None:
            self.output = self.data