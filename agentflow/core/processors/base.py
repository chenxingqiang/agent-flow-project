from typing import Any, Dict, Optional

class ProcessorResult:
    """Result from processor execution."""
    
    def __init__(
        self,
        output: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        error: Optional[str] = None
    ):
        self.output = output or {}
        self.metadata = metadata or {}
        self.error = error 