import pytest
from datetime import date
from agentflow.core.config import (
    ResearchAgentConfig, 
    ConfigTypeConverter, 
    ConfigurationError
)
import os
from agentflow.core.base_types import AgentType

def test_agent_config_from_yaml():
    """Test loading agent configuration from YAML."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'agents')
    config = ResearchAgentConfig.from_yaml(
        config_path=config_path,
        config_name='academic_support_agent'
    )

    assert config.name == 'Academic_Research_Support_Agent'
    assert config.type == AgentType.RESEARCH

def test_research_agent_config_from_yaml():
    """Test loading research-specific agent configuration."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'agents')
    config = ResearchAgentConfig.from_yaml(
        config_path=config_path, 
        config_name='academic_support_agent'
    )
    
    assert config.research_context is not None
    assert config.publication_goals is not None
    
    # Validate research context
    assert config.research_context.get('topic') == '基于静态图的恶意软件分类方法研究'
    assert 'current_status' in config.research_context

def test_agent_config_to_dict():
    """Test converting agent config to dictionary."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'agents')
    config = ResearchAgentConfig.from_yaml(
        config_path=config_path,
        config_name='academic_support_agent'
    )
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict['name'] == 'Academic_Research_Support_Agent'
    assert config_dict['type'] == 'research'

def test_agent_config_from_dict():
    """Test creating agent config from dictionary."""
    config_dict = {
        'name': 'Test Agent',
        'type': 'research',
        'version': '0.1.0',
        'model': {
            'provider': 'openai',
            'name': 'gpt-4'
        }
    }
    
    config = ResearchAgentConfig.from_dict(config_dict)
    assert config.name == 'Test Agent'
    assert config.type == 'research'
    assert config.version == '0.1.0'

def test_research_agent_specific_attributes():
    """Test research agent specific configuration attributes."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'agents')
    config = ResearchAgentConfig.from_yaml(
        config_path=config_path, 
        config_name='academic_support_agent'
    )
    
    # Check publication goals
    assert config.publication_goals.get('target_journal') == '北大中文核心'
    assert config.publication_goals.get('acceptance_deadline') == date(2025, 3, 1)
    assert config.publication_goals.get('submission_timeline') == '3个月内'
    
    # Check research context
    assert config.research_context.get('topic') == '基于静态图的恶意软件分类方法研究'
    assert 'current_status' in config.research_context
    
    # Check research methods and tools
    assert isinstance(config.research_methods, list)
    assert isinstance(config.analysis_tools, list)

def test_config_type_converter():
    """Test the ConfigTypeConverter utility class."""
    # Test string to int conversion
    assert ConfigTypeConverter.convert_value('42', int) == 42
    
    # Test string to float conversion
    assert ConfigTypeConverter.convert_value('3.14', float) == 3.14
    
    # Test string to date conversion
    assert ConfigTypeConverter.convert_value('2025-03-01', date) == date(2025, 3, 1)
    
    # Test list conversion
    assert ConfigTypeConverter.convert_value('single', list) == ['single']
    
    # Test dict conversion for list
    assert ConfigTypeConverter.convert_value([1, 2, 3], dict) == {'0': 1, '1': 2, '2': 3}
    
    # Test dict conversion for non-list
    assert ConfigTypeConverter.convert_value('value', dict) == {'value': 'value'}

def test_research_agent_config_validation():
    """Test validation of research agent configuration."""
    # Test valid configuration
    config_dict = {
        'name': 'Valid Research Agent',
        'type': 'research',
        'version': '1.0.0',
        'model': {
            'provider': 'openai',
            'name': 'gpt-4'
        },
        'research_context': {
            'topic': 'Test Research',
            'current_status': ['Initial Phase']
        },
        'publication_goals': {
            'target_journal': 'Test Journal',
            'acceptance_deadline': '2025-03-01',
            'submission_timeline': '3 months'
        }
    }
    
    config = ResearchAgentConfig.model_validate(config_dict)
    assert config.name == 'Valid Research Agent'
    assert config.research_context['topic'] == 'Test Research'
    assert config.publication_goals['acceptance_deadline'] == date(2025, 3, 1)

def test_research_agent_name_validation():
    """Test name validation for research agent configuration."""
    # Test valid names
    valid_names = [
        'Research Agent',
        'Agent_123',
        'Advanced-Research-Agent'
    ]
    
    for name in valid_names:
        config = ResearchAgentConfig(name=name)
        assert config.name == name
    
    # Test invalid names
    invalid_names = [
        '',  # Empty name
        'AB',  # Too short
        'Invalid@Name',  # Contains special characters
    ]
    
    for name in invalid_names:
        with pytest.raises(ValueError):
            ResearchAgentConfig(name=name)

def test_research_agent_publication_goals_validation():
    """Test validation of publication goals."""
    # Test valid publication goals
    valid_goals = {
        'target_journal': 'Test Journal',
        'acceptance_deadline': '2025-03-01',
        'submission_timeline': '3 months'
    }
    
    config = ResearchAgentConfig(publication_goals=valid_goals)
    assert config.publication_goals['acceptance_deadline'] == date(2025, 3, 1)
    
    # Test invalid date format
    with pytest.raises(ValueError):
        ResearchAgentConfig(publication_goals={
            'target_journal': 'Test Journal',
            'acceptance_deadline': 'invalid-date',
        })

def test_config_validation_with_schema():
    """Test configuration validation with a type schema."""
    schema = {
        'name': str,
        'age': int,
        'active': bool,
        'deadline': date
    }
    
    config = {
        'name': 'Test Config',
        'age': '30',  # String that can be converted to int
        'active': 'true',  # String that can be converted to bool
        'deadline': '2025-03-01'  # String that can be converted to date
    }
    
    validated_config = ConfigTypeConverter.validate_config(config, schema)
    
    assert validated_config['name'] == 'Test Config'
    assert validated_config['age'] == 30
    assert validated_config['active'] is True
    assert validated_config['deadline'] == date(2025, 3, 1)

def test_config_validation_with_invalid_schema():
    """Test configuration validation with an invalid schema."""
    schema = {
        'name': str,
        'age': int
    }
    
    config = {
        'name': 'Test Config',
        'age': 'not an integer'
    }
    
    # This should log a warning but not raise an exception
    validated_config = ConfigTypeConverter.validate_config(config, schema)
    
    assert validated_config['name'] == 'Test Config'
    assert validated_config['age'] == 'not an integer'  # Original value preserved

import pytest
from datetime import date, datetime
from agentflow.core.config import (
    ResearchAgentConfig, 
    ConfigTypeConverter, 
    ConfigurationError,
    ConfigSecurityManager,
    ConfigurationInheritanceResolver,
    DynamicConfigGenerator
)
import os
import time
import yaml
from omegaconf import OmegaConf

def test_config_security_manager():
    """Test configuration security features."""
    # Test sensitive data hashing
    data = "my_secret_password"
    hashed_data = ConfigSecurityManager.hash_sensitive_data(data)
    assert hashed_data != data
    assert len(hashed_data) == 64  # SHA-256 hash length
    
    # Test sensitive field masking
    config = {
        'database': {
            'password': 'secret123',
            'host': 'localhost'
        },
        'api': {
            'key': 'api_secret'
        }
    }
    
    masked_config = ConfigSecurityManager.mask_sensitive_fields(config)
    
    # Check that sensitive fields are masked
    assert isinstance(masked_config, dict)
    assert masked_config.get('database') == '***MASKED***'
    assert masked_config.get('api') == '***MASKED***'

def test_configuration_inheritance():
    """Test configuration inheritance and merging."""
    # Create test configuration files
    base_config = {
        'agent': {
            'name': 'Base Agent',
            'type': 'base',
            'version': '1.0.0'
        },
        'research_context': {
            'base_topic': 'General Research'
        }
    }
    
    env_config = {
        'agent': {
            'name': 'Development Agent',
            'type': 'development'
        },
        'research_context': {
            'specific_topic': 'Development Research'
        }
    }
    
    # Simulate file-based configuration
    with open('/tmp/base_config.yaml', 'w') as f:
        yaml.dump(base_config, f)
    
    with open('/tmp/base_config.development.yaml', 'w') as f:
        yaml.dump(env_config, f)
    
    # Test inheritance resolver
    merged_config = ConfigurationInheritanceResolver.load_config_with_inheritance(
        '/tmp', 'base_config', 'development'
    )
    
    assert merged_config['agent']['name'] == 'Development Agent'
    assert merged_config['agent']['type'] == 'development'
    assert merged_config['agent']['version'] == '1.0.0'
    assert merged_config['research_context']['base_topic'] == 'General Research'
    assert merged_config['research_context']['specific_topic'] == 'Development Research'

def test_dynamic_config_generation():
    """Test dynamic configuration generation."""
    def config_generator():
        """Example configuration generator function."""
        return {
            'agent': {
                'name': f'Dynamic Agent {time.time()}',
                'type': 'dynamic'
            }
        }
    
    # Test dynamic config generation
    dynamic_config = DynamicConfigGenerator.generate_config_from_function(config_generator)
    
    assert 'agent' in dynamic_config
    assert dynamic_config['agent']['type'] == 'dynamic'
    
    # Test timestamped config generation
    base_config = {'agent': {'name': 'Test Agent'}}
    timestamped_config = DynamicConfigGenerator.generate_timestamped_config(base_config)
    
    assert '_metadata' in timestamped_config
    assert 'generated_at' in timestamped_config['_metadata']
    assert 'version' in timestamped_config['_metadata']
    assert len(timestamped_config['_metadata']['version']) == 8

def test_advanced_type_conversion():
    """Test advanced type conversion with strict and default modes."""
    # Test default conversion
    assert ConfigTypeConverter.convert_value('42', int) == 42
    assert ConfigTypeConverter.convert_value('3.14', float) == 3.14
    
    # Test strict mode
    with pytest.raises(ValueError):
        ConfigTypeConverter.convert_value('not a number', int, strict=True)
    
    # Test non-strict mode returns default values
    assert ConfigTypeConverter.convert_value('not a number', int) == 0
    
    # Test complex type conversion
    assert ConfigTypeConverter.convert_value('2025-03-01', date) == date(2025, 3, 1)
    assert ConfigTypeConverter.convert_value('single', list) == ['single']

def test_config_environment_loading():
    """Test loading configurations for different environments."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'agents')
    
    # Test default environment
    default_config = ResearchAgentConfig.from_yaml(
        config_path=config_path, 
        config_name='academic_support_agent', 
        environment='default'
    )
    assert default_config.name == 'Academic_Research_Support_Agent'
    
    # Note: This assumes you have a development-specific configuration
    # If not, you might want to create one or modify the test
    try:
        dev_config = ResearchAgentConfig.from_yaml(
            config_path=config_path, 
            config_name='academic_support_agent', 
            environment='development'
        )
        # Add specific assertions for development environment
    except FileNotFoundError:
        # Skip if development config doesn't exist
        pass

def test_config_validation_with_complex_schema():
    """Test configuration validation with a complex type schema."""
    schema = {
        'name': str,
        'age': int,
        'active': bool,
        'deadline': date,
        'skills': list,
        'metadata': dict
    }

    config = {
        'name': 'Test Config',
        'age': '30',
        'active': 'true',
        'deadline': '2025-03-01',
        'skills': 'Python',
        'metadata': 'extra info'
    }

    validated_config = ConfigTypeConverter.validate_config(config, schema)

    assert validated_config['name'] == 'Test Config'
    assert validated_config['age'] == 30
    assert validated_config['active'] is True
    assert validated_config['deadline'] == date(2025, 3, 1)
    assert validated_config['skills'] == ['Python']
    assert validated_config['metadata'] == {'value': 'extra info'}
