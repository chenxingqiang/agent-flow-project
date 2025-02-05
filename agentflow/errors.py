"""Error classes for agentflow."""

class AgentFlowError(Exception):
    """Base class for all agentflow errors."""
    pass

class WorkflowExecutionError(AgentFlowError):
    """Exception raised when workflow execution fails."""
    pass

class ConfigurationError(AgentFlowError):
    """Exception raised when configuration is invalid."""
    pass

class ValidationError(AgentFlowError):
    """Exception raised when validation fails."""
    pass

class InitializationError(AgentFlowError):
    """Exception raised when initialization fails."""
    pass

class ResourceError(AgentFlowError):
    """Exception raised when resource management fails."""
    pass

class CommunicationError(AgentFlowError):
    """Exception raised when communication fails."""
    pass

class TimeoutError(AgentFlowError):
    """Exception raised when operation times out."""
    pass

class AuthenticationError(AgentFlowError):
    """Exception raised when authentication fails."""
    pass

class PermissionError(AgentFlowError):
    """Exception raised when permission is denied."""
    pass

class NotFoundError(AgentFlowError):
    """Exception raised when resource is not found."""
    pass

class DuplicateError(AgentFlowError):
    """Exception raised when duplicate resource is detected."""
    pass

class StateError(AgentFlowError):
    """Exception raised when state is invalid."""
    pass

class DataError(AgentFlowError):
    """Exception raised when data is invalid."""
    pass

class IntegrationError(AgentFlowError):
    """Exception raised when integration fails."""
    pass

class VersionError(AgentFlowError):
    """Exception raised when version is incompatible."""
    pass

class EnvironmentError(AgentFlowError):
    """Exception raised when environment is invalid."""
    pass

class DependencyError(AgentFlowError):
    """Exception raised when dependency is missing or invalid."""
    pass

class PluginError(AgentFlowError):
    """Exception raised when plugin operation fails."""
    pass 