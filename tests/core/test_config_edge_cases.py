import pytest
import os
import yaml
from typing import Dict, Any
from datetime import date, datetime
from agentflow.core.config import (
    ConfigTypeConverterHydra as ConfigTypeConverter, 
    ConfigurationInheritanceResolverHydra as ConfigurationInheritanceResolver, 
    ConfigSecurityManagerHydra as ConfigSecurityManager,
    ResearchAgentConfigHydra as ResearchAgentConfig,
    ConfigurationError
)
import hydra
from omegaconf import OmegaConf

def test_type_conversion_edge_cases():
    """Test complex and edge cases for type conversion."""
    test_cases = [
        # Numeric conversions
        ('42', int, 42),
        ('3.14', float, 3.14),
        ('0', int, 0),
        ('-42', int, -42),
        
        # Boolean conversions
        ('true', bool, True),
        ('false', bool, False),
        ('1', bool, True),
        ('0', bool, False),
        
        # Date conversions
        ('2025-01-06', date, date(2025, 1, 6)),
        
        # List conversions
        ('single', list, ['single']),
        ([1, 2, 3], list, [1, 2, 3]),
        
        # Dict conversions
        ('extra', dict, {'value': 'extra'}),
        ({'key': 'value'}, dict, {'key': 'value'}),
    ]
    
    for value, target_type, expected in test_cases:
        result = ConfigTypeConverter.convert_value(value, target_type)
        assert result == expected, f"Failed conversion: {value} to {target_type}"

def test_type_conversion_failure_modes():
    """Test failure modes and error handling in type conversion."""
    failure_cases = [
        # Strict mode conversions
        ('not_a_number', int, True),  # Strict mode should raise
        ('not_a_bool', bool, True),   # Strict mode should raise
        
        # Boundary and edge cases
        (None, int, False),  # None handling
        ('', str, False),    # Empty string
    ]
    
    for value, target_type, strict in failure_cases:
        if strict:
            with pytest.raises((ValueError, ConfigurationError)):
                ConfigTypeConverter.convert_value(value, target_type, strict=True)
        else:
            result = ConfigTypeConverter.convert_value(value, target_type)
            assert result is None or result == value

def test_configuration_inheritance_complex_scenarios():
    """Test complex configuration inheritance scenarios."""
    base_config = {
        'agent': {
            'name': 'Base Agent',
            'version': '1.0.0',
            'capabilities': ['base_capability'],
            'nested': {
                'deep_key': 'base_value'
            }
        }
    }
    
    override_configs = [
        # Simple override
        {
            'agent': {
                'name': 'Overridden Agent'
            }
        },
        # Deep merge
        {
            'agent': {
                'capabilities': ['new_capability'],
                'nested': {
                    'additional_key': 'override_value'
                }
            }
        },
        # Complete replacement
        {
            'agent': {
                'name': 'Completely New Agent',
                'version': '2.0.0'
            }
        }
    ]
    
    for override_config in override_configs:
        merged_config = ConfigurationInheritanceResolver.resolve_inheritance(
            OmegaConf.create(base_config),
            OmegaConf.create(override_config)
        )
    
        # Validate merge behavior
        assert merged_config.agent.name == override_config['agent'].get('name', base_config['agent']['name'])
    
        # Ensure deep merge preserves base values when not overridden
        if 'nested' in base_config['agent'] and 'nested' not in override_config['agent']:
            # Check that the nested configuration is preserved
            merged_dict = OmegaConf.to_container(merged_config, resolve=True)
            base_dict = base_config
            
            # Verify nested configuration is preserved
            assert 'nested' in merged_dict['agent']
            assert 'deep_key' in merged_dict['agent']['nested']
            assert merged_dict['agent']['nested']['deep_key'] == base_dict['agent']['nested']['deep_key']

def test_security_masking_comprehensive():
    """Comprehensive test for security field masking."""
    test_configs = [
        # Nested sensitive configurations
        {
            'database': {
                'password': 'secret123',
                'host': 'localhost'
            },
            'api': {
                'key': 'api_secret',
                'endpoint': 'https://api.example.com'
            },
            'safe_field': 'public_value'
        },
        # Deeply nested configurations
        {
            'services': {
                'authentication': {
                    'credentials': {
                        'username': 'admin',
                        'password': 'super_secret'
                    }
                }
            }
        }
    ]

    for config in test_configs:
        print(f"Original config: {config}")
        masked_config = ConfigSecurityManager.mask_sensitive_fields(OmegaConf.create(config))
        print(f"Masked config: {masked_config}")

        def check_masking(original, masked):
            def is_sensitive_key(key):
                return any(sensitive in str(key).lower() for sensitive in ['password', 'secret', 'key', 'credentials'])

            def check_masked_value(key, value, masked_value):
                # If the key is sensitive, it should be masked
                if is_sensitive_key(key):
                    # If the entire dictionary is sensitive, all values should be masked
                    if isinstance(value, dict):
                        assert masked_value == '***MASKED***', f"Failed to mask dictionary for {key}"
                    else:
                        assert masked_value == '***MASKED***', f"Failed to mask {key}"

            def recursive_check(orig_dict, masked_dict):
                # Handle case where the entire config is masked
                if masked_dict == '***MASKED***':
                    return

                # If masked_dict is a string, it means the parent was masked
                if isinstance(masked_dict, str):
                    return

                for key, value in orig_dict.items():
                    print(f"Checking key: {key}, value: {value}")

                    # Check if the current key or its parent is sensitive
                    if is_sensitive_key(key):
                        check_masked_value(key, value, masked_dict[key])

                    # Recurse into nested dictionaries
                    if isinstance(value, dict):
                        # If the parent was sensitive, it would have been masked entirely
                        if isinstance(masked_dict[key], str):
                            continue
                        recursive_check(value, masked_dict[key])

            recursive_check(config, masked_config)

        check_masking(config, masked_config)

def test_configuration_validation_complex_schemas():
    """Test configuration validation with complex, nested schemas."""
    schemas = [
        # Nested type validation
        {
            'user': {
                'name': str,
                'age': int,
                'preferences': {
                    'theme': str,
                    'notifications': bool
                }
            }
        },
        # Mixed type validation
        {
            'project': {
                'name': str,
                'start_date': date,
                'team_members': list,
                'budget': float
            }
        }
    ]

    test_configs = [
        {
            'user': {
                'name': 'Test User',
                'age': '30',
                'preferences': {
                    'theme': 'dark',
                    'notifications': 'true'
                }
            }
        },
        {
            'project': {
                'name': 'Research Project',
                'start_date': '2025-01-06',
                'team_members': 'John',
                'budget': '1000.50'
            }
        }
    ]

    for schema, config in zip(schemas, test_configs):
        validated_config = ConfigTypeConverter.convert_value(
            OmegaConf.create(config),
            dict,
            schema=schema
        )

        # Validate type conversions
        for key, expected_type in schema.items():
            if isinstance(expected_type, dict):
                for nested_key, nested_type in expected_type.items():
                    converted_value = validated_config[key][nested_key]
                    # Special handling for type comparison
                    if nested_type == bool:
                        assert isinstance(converted_value, bool), f"Failed to convert {nested_key} to bool"
                    elif nested_type == int:
                        assert isinstance(converted_value, int), f"Failed to convert {nested_key} to int"
                    elif nested_type == float:
                        assert isinstance(converted_value, float), f"Failed to convert {nested_key} to float"
                    elif nested_type == str:
                        assert isinstance(converted_value, str), f"Failed to convert {nested_key} to str"
                    elif nested_type == date:
                        assert isinstance(converted_value, date), f"Failed to convert {nested_key} to date"

def test_dynamic_config_generation_time_sensitivity():
    """Test dynamic configuration generation with time sensitivity."""
    def config_generator():
        return {
            'agent': {
                'name': f'Dynamic Agent {datetime.now().isoformat()}',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    # Generate multiple configs and ensure uniqueness
    configs = [config_generator() for _ in range(3)]
    
    # Verify timestamp uniqueness
    timestamps = [config['agent']['timestamp'] for config in configs]
    assert len(set(timestamps)) == 3, "Timestamps should be unique"

def test_configuration_loading_error_handling():
    """Test comprehensive error handling during configuration loading."""
    # Test non-existent path
    with pytest.raises(ConfigurationError, match="Configuration path .* does not exist"):
        ResearchAgentConfig.from_yaml(
            config_path='/non_existent_path',
            config_name='non_existent_config'
        )

    # Test malformed configuration
    malformed_config = {
        'invalid_key': {'nested_key': lambda x: x}  # Use a non-serializable object
    }

    with pytest.raises(ConfigurationError, match="Invalid configuration.*not a supported primitive type"):
        ResearchAgentConfig.from_dict(malformed_config)
