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

class WorkflowExecutionError(Exception):
    """Exception raised when workflow execution fails"""
    pass

class WorkflowValidationError(Exception):
    """Exception raised when workflow validation fails"""
    pass

class StepExecutionError(Exception):
    """Exception raised when step execution fails"""
    pass

class StepValidationError(Exception):
    """Exception raised when step validation fails"""
    pass

class StepTimeoutError(Exception):
    """Exception raised when step execution times out"""
    pass

class StepRetryError(Exception):
    """Exception raised when step retry mechanism fails"""
    pass

class WorkflowStateError(Exception):
    """Exception raised when workflow state is invalid"""
    pass

class WorkflowConfigError(Exception):
    """Exception raised when workflow configuration is invalid"""
    pass

class WorkflowInputError(Exception):
    """Exception raised when workflow input is invalid"""
    pass

class WorkflowOutputError(Exception):
    """Exception raised when workflow output is invalid"""
    pass
