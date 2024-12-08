"""Exceptions for AgentFlow core."""

class AgentFlowError(Exception):
    """Base exception for AgentFlow errors."""
    pass

class ConfigurationError(AgentFlowError):
    """Raised when there is a configuration error."""
    pass

class ValidationError(AgentFlowError):
    """Raised when validation fails."""
    pass

class WorkflowError(AgentFlowError):
    """Raised when there is a workflow error."""
    pass

class ObjectiveError(AgentFlowError):
    """Raised when there is an objective error."""
    pass

class PersistenceError(AgentFlowError):
    """Raised when there is a persistence error."""
    pass

class MonitoringError(AgentFlowError):
    """Raised when there is a monitoring error."""
    pass

class IntegrationError(AgentFlowError):
    """Raised when there is an integration error."""
    pass

class DashboardError(AgentFlowError):
    """Raised when there is a dashboard error."""
    pass
