"""Tests for ELL2A warnings utility."""

import pytest
import logging
from unittest.mock import MagicMock
from colorama import Fore, Style

# Import the module directly
from agentflow.ell2a.util._warnings import _no_api_key_warning, _warnings, _autocommit_warning

# Mock the config import
class MockConfig:
    registry = {}
    default_client = "OpenAIClient"
    get_client_for = MagicMock(return_value=(None, None))
    autocommit_model = "test_model"

# Patch the configuration and logger
@pytest.fixture(autouse=True)
def mock_config_and_logger(monkeypatch):
    monkeypatch.setattr('agentflow.ell2a.util._warnings.config', MockConfig)
    monkeypatch.setattr('agentflow.ell2a.util._warnings.logger', MagicMock())

def test_no_api_key_warning_default():
    """Test _no_api_key_warning with default parameters."""
    warning = _no_api_key_warning("gpt-3.5-turbo", None)
    
    # Verify basic warning structure
    assert "WARNING" in warning
    assert "No API key found" in warning
    assert "gpt-3.5-turbo" in warning
    assert "OpenAI" in warning
    assert Fore.LIGHTYELLOW_EX in warning
    assert Style.RESET_ALL in warning

def test_no_api_key_warning_with_name():
    """Test _no_api_key_warning with a specific LMP name."""
    warning = _no_api_key_warning("gpt-3.5-turbo", None, name="test_lmp")
    
    assert "used by LMP `test_lmp`" in warning

def test_no_api_key_warning_long_format():
    """Test _no_api_key_warning with long format."""
    warning = _no_api_key_warning("gpt-3.5-turbo", None, long=True)
    
    # Verify long format includes detailed instructions
    assert "To fix this:" in warning
    assert "Set your API key" in warning
    assert "specify a client explicitly" in warning

def test_no_api_key_warning_error_mode():
    """Test _no_api_key_warning in error mode."""
    try:
        _no_api_key_warning("gpt-3.5-turbo", None, error=True)
        pytest.fail("Expected ValueError to be raised")
    except ValueError as e:
        error_msg = str(e)
        assert "ERROR" in error_msg
        assert "No API key found" in error_msg
        assert "OpenAIClient" in error_msg
        assert "gpt-3.5-turbo" in error_msg

def test_no_api_key_warning_with_custom_client():
    """Test _no_api_key_warning with a custom client."""
    # Create a mock client
    mock_client = MagicMock()
    mock_client.__class__.__name__ = "MockClient"
    mock_client.__class__.__module__ = "test_module"
    
    warning = _no_api_key_warning("gpt-3.5-turbo", mock_client)
    
    # Verify client details are included
    assert "MockClient" in warning

def test_warnings_unregistered_model(mocker):
    """Test _warnings with an unregistered model."""
    # Create a mock function
    def mock_fn():
        pass
    mock_fn.__name__ = "test_function"
    
    # Mock the logger
    mock_logger = mocker.patch('agentflow.ell2a.util._warnings.logger')
    
    # Call _warnings
    decorated_fn = _warnings("gpt-3.5-turbo")(mock_fn)
    decorated_fn()
    
    # Verify warning was logged
    mock_logger.warning.assert_called_once()
    warning_msg = mock_logger.warning.call_args[0][0]
    
    assert "WARNING" in warning_msg
    assert "gpt-3.5-turbo" in warning_msg
    assert "not registered" in warning_msg
    assert Fore.LIGHTYELLOW_EX in warning_msg

def test_autocommit_warning(mocker):
    """Test _autocommit_warning."""
    # Mock the logger
    mock_logger = mocker.patch('agentflow.ell2a.util._warnings.logger')
    
    # Call _autocommit_warning
    result = _autocommit_warning()
    
    # Verify warning was logged
    mock_logger.warning.assert_called_once()
    warning_msg = mock_logger.warning.call_args[0][0]
    
    assert "WARNING" in warning_msg
    assert "Autocommit is enabled" in warning_msg
    assert "no client found" in warning_msg
    assert "test_model" in warning_msg
    assert Fore.LIGHTYELLOW_EX in warning_msg
    
    # Verify return value
    assert result is True
