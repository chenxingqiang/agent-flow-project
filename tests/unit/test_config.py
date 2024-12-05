import pytest
from pathlib import Path
from agentflow.core.config import ConfigManager

def test_config_loading(test_data_dir):
    """Test configuration loading"""
    config_manager = ConfigManager(str(test_data_dir / 'config.json'))
    assert config_manager.config is not None
    assert 'variables' in config_manager.config

def test_config_validation(test_data_dir):
    """Test configuration validation"""
    config_manager = ConfigManager(str(test_data_dir / 'config.json'))
    config_manager.validate_config(config_manager.config)

def test_variable_extraction(test_data_dir):
    """Test variable extraction"""
    config_manager = ConfigManager(str(test_data_dir / 'config.json'))
    variables = config_manager.extract_variables()
    assert isinstance(variables, dict)
    assert 'test_var' in variables

@pytest.mark.parametrize("config_update,expected_error", [
    ({"invalid_key": "value"}, ValueError),
    ({"variables": {"invalid_type": {"type": "invalid"}}}, ValueError),
])
def test_config_update_validation(test_data_dir, config_update, expected_error):
    """Test configuration update validation"""
    config_manager = ConfigManager(str(test_data_dir / 'config.json'))
    
    with pytest.raises(expected_error):
        config_manager.update_config(config_update)