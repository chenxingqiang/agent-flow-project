# Distributed Workflow API Module

from .workflow_server import app as workflow_app
from .app import app

__all__ = ['app', 'workflow_app']
