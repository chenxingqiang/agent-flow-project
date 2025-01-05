"""Test closure module."""

import pytest
from typing import Optional, List, Dict, Any

from agentflow.ell2a.util.closure import (
    closure,
    get_closure_for_function,
    get_closure_for_class,
    get_closure_for_module
)