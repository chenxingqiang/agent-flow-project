import unittest
import json
import os
import tempfile
from typing import Dict, Any

from agentflow.core.agent import (
    ConfigurationSchema, 
    ConfigurationType, 
    FlexibleConfigurationManager
)

class TestConfigurationFlexibility(unittest.TestCase):
    def setUp(self):
        """Set up test environment with sample configurations."""
        self.base_config = {
            "AGENT": {
                "type": "RESEARCH",
                "name": "TestResearchAgent",
                "version": "1.2.0",
                "description": "Advanced research agent for scientific workflows"
            },
            "WORKFLOW": {
                "max_iterations": 15,
                "timeout": 600,
                "retry_strategy": {
                    "max_retries": 5,
                    "backoff_factor": 2
                }
            },
            "TRANSFORMATIONS": {
                "input": [
                    {
                        "type": "outlier_removal",
                        "params": {
                            "method": "z_score",
                            "threshold": 3.0
                        }
                    }
                ],
                "preprocessing": [
                    {
                        "type": "feature_engineering",
                        "params": {
                            "strategy": "polynomial",
                            "degree": 2
                        }
                    }
                ]
            },
            "DOMAIN_CONFIG": {
                "research_domains": ["machine learning", "data science"]
            }
        }
        
        self.config_manager = FlexibleConfigurationManager()
    
    def test_configuration_schema_creation(self):
        """Test creating configuration schema from dictionary."""
        config_schema = ConfigurationSchema.from_dict(self.base_config)
        
        # Validate basic properties
        self.assertEqual(config_schema.name, "TestResearchAgent")
        self.assertEqual(config_schema.type, ConfigurationType.RESEARCH)
        self.assertEqual(config_schema.max_iterations, 15)
        
        # Validate transformations
        self.assertEqual(len(config_schema.transformations['input']), 1)
        self.assertEqual(len(config_schema.transformations['preprocessing']), 1)
    
    def test_configuration_validation(self):
        """Test comprehensive configuration validation."""
        # Valid configuration
        valid_config = ConfigurationSchema.from_dict(self.base_config)
        self.assertTrue(valid_config.validate())
        
        # Invalid configurations
        invalid_configs = [
            # Negative max iterations
            {**self.base_config, "WORKFLOW": {"max_iterations": -1}},
            
            # Invalid transformation
            {
                **self.base_config, 
                "TRANSFORMATIONS": {
                    "input": [
                        {
                            "type": "outlier_removal",
                            "params": {
                                "method": "invalid_method",
                                "threshold": -1
                            }
                        }
                    ]
                }
            }
        ]
        
        for invalid_config in invalid_configs:
            with self.subTest(config=invalid_config):
                config_schema = ConfigurationSchema.from_dict(invalid_config)
                self.assertFalse(config_schema.validate())
    
    def test_configuration_merging(self):
        """Test configuration merging with priority handling."""
        base_config = ConfigurationSchema.from_dict(self.base_config)
        
        # Partial configuration to merge
        partial_config = ConfigurationSchema(
            name="MergedAgent",
            max_iterations=20,
            transformations={
                "output": [
                    {
                        "type": "text_transformation",
                        "params": {
                            "strategy": "lemmatize"
                        }
                    }
                ]
            }
        )
        
        # Merge configurations
        merged_config = base_config.merge(partial_config)
        
        # Validate merged configuration
        self.assertEqual(merged_config.name, "TestResearchAgent")  # Base config takes priority
        self.assertEqual(merged_config.max_iterations, 15)
        
        # Validate transformation merging
        self.assertEqual(len(merged_config.transformations['output']), 1)
        self.assertEqual(
            merged_config.transformations['output'][0]['type'], 
            'text_transformation'
        )
    
    def test_configuration_export_and_import(self):
        """Test configuration export and import functionality."""
        config_schema = ConfigurationSchema.from_dict(self.base_config)
        
        # Test JSON export
        json_config = config_schema.export(format='json')
        self.assertIsInstance(json_config, str)
        
        # Test dictionary export
        dict_config = config_schema.export(format='dict')
        self.assertIsInstance(dict_config, dict)
        
        # Test file-based configuration saving and loading
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            json.dump(self.base_config, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Load configuration from file
            loaded_config = self.config_manager.load_configuration(temp_file_path)
            self.assertIsInstance(loaded_config, ConfigurationSchema)
            self.assertEqual(loaded_config.name, "TestResearchAgent")
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def test_flexible_configuration_loading(self):
        """Test loading configurations from various sources."""
        test_scenarios = [
            # Dictionary configuration
            self.base_config,
            
            # JSON string configuration
            json.dumps(self.base_config),
            
            # Partial configuration
            {
                "AGENT": {"name": "PartialAgent"},
                "WORKFLOW": {"max_iterations": 5}
            }
        ]
        
        for scenario in test_scenarios:
            with self.subTest(scenario=scenario):
                try:
                    loaded_config = self.config_manager.load_configuration(scenario)
                    self.assertIsInstance(loaded_config, ConfigurationSchema)
                except Exception as e:
                    self.fail(f"Failed to load configuration: {e}")
    
    def test_domain_specific_configuration(self):
        """Test domain-specific configuration support."""
        # Create configuration with domain-specific settings
        domain_config = {
            **self.base_config,
            "DOMAIN_CONFIG": {
                "research_domains": ["quantum computing", "artificial intelligence"],
                "citation_styles": ["APA", "IEEE"],
                "publication_targets": {
                    "conferences": ["NeurIPS", "ICML"],
                    "journals": ["Nature", "Science"]
                }
            }
        }
        
        config_schema = ConfigurationSchema.from_dict(domain_config)
        
        # Validate domain-specific configuration
        self.assertEqual(
            config_schema.domain_config['research_domains'], 
            ["quantum computing", "artificial intelligence"]
        )
        self.assertIn("publication_targets", config_schema.domain_config)
    
    def test_advanced_transformation_configuration(self):
        """Test advanced transformation configuration scenarios."""
        advanced_config = {
            **self.base_config,
            "TRANSFORMATIONS": {
                "input": [
                    {
                        "type": "outlier_removal",
                        "params": {
                            "method": "modified_z_score",
                            "threshold": 3.5,
                            "handling_strategy": "replace"
                        }
                    }
                ],
                "preprocessing": [
                    {
                        "type": "feature_engineering",
                        "params": {
                            "strategy": "polynomial",
                            "degree": 3,
                            "interaction_terms": True
                        }
                    }
                ],
                "output": [
                    {
                        "type": "text_transformation",
                        "params": {
                            "strategy": "lemmatize",
                            "language": "en"
                        }
                    }
                ]
            }
        }
        
        config_schema = ConfigurationSchema.from_dict(advanced_config)
        
        # Validate advanced transformation configuration
        self.assertEqual(
            config_schema.transformations['input'][0]['params']['method'], 
            'modified_z_score'
        )
        self.assertTrue(
            config_schema.transformations['preprocessing'][0]['params'].get('interaction_terms')
        )
        self.assertEqual(
            config_schema.transformations['output'][0]['params']['strategy'], 
            'lemmatize'
        )

    def test_research_agent_configuration(self):
        """
        Test specific Research Agent configuration with quantum computing domain.
        
        Validates:
        - Agent type configuration
        - Outlier removal transformation
        - Domain-specific configuration
        """
        research_config = {
            "AGENT": {
                "type": "RESEARCH",
                "name": "QuantumResearchAgent"
            },
            "TRANSFORMATIONS": {
                "input": [
                    {
                        "type": "outlier_removal",
                        "params": {
                            "method": "z_score",
                            "threshold": 3.0
                        }
                    }
                ]
            },
            "DOMAIN_CONFIG": {
                "research_domains": ["quantum computing"]
            }
        }
        
        # Load and validate configuration
        config_schema = ConfigurationSchema.from_dict(research_config)
        
        # Validate agent configuration
        self.assertEqual(config_schema.name, "QuantumResearchAgent")
        self.assertEqual(config_schema.type, ConfigurationType.RESEARCH)
        
        # Validate transformations
        self.assertEqual(len(config_schema.transformations['input']), 1)
        input_transformation = config_schema.transformations['input'][0]
        self.assertEqual(input_transformation['type'], 'outlier_removal')
        self.assertEqual(input_transformation['params']['method'], 'z_score')
        self.assertEqual(input_transformation['params']['threshold'], 3.0)
        
        # Validate domain configuration
        self.assertEqual(
            config_schema.domain_config['research_domains'], 
            ["quantum computing"]
        )
        
        # Validate configuration
        self.assertTrue(config_schema.validate())
    
    def test_data_science_agent_configuration(self):
        """
        Test specific Data Science Agent configuration with feature engineering.
        
        Validates:
        - Agent type configuration
        - Feature engineering transformation
        - Preprocessing configuration
        """
        data_science_config = {
            "AGENT": {
                "type": "DATA_SCIENCE",
                "name": "MachineLearningAgent"
            },
            "TRANSFORMATIONS": {
                "preprocessing": [
                    {
                        "type": "feature_engineering",
                        "params": {
                            "strategy": "polynomial",
                            "degree": 2
                        }
                    }
                ]
            }
        }
        
        # Load and validate configuration
        config_schema = ConfigurationSchema.from_dict(data_science_config)
        
        # Validate agent configuration
        self.assertEqual(config_schema.name, "MachineLearningAgent")
        self.assertEqual(config_schema.type, ConfigurationType.DATA_SCIENCE)
        
        # Validate transformations
        self.assertEqual(len(config_schema.transformations['preprocessing']), 1)
        preprocessing_transformation = config_schema.transformations['preprocessing'][0]
        self.assertEqual(preprocessing_transformation['type'], 'feature_engineering')
        self.assertEqual(preprocessing_transformation['params']['strategy'], 'polynomial')
        self.assertEqual(preprocessing_transformation['params']['degree'], 2)
        
        # Validate configuration
        self.assertTrue(config_schema.validate())
    
    def test_combined_agent_configuration(self):
        """
        Test combining Research and Data Science agent configurations.
        
        Validates:
        - Merging configurations
        - Preserving unique transformation strategies
        """
        research_config = {
            "AGENT": {
                "type": "RESEARCH",
                "name": "QuantumResearchAgent"
            },
            "TRANSFORMATIONS": {
                "input": [
                    {
                        "type": "outlier_removal",
                        "params": {
                            "method": "z_score",
                            "threshold": 3.0
                        }
                    }
                ]
            },
            "DOMAIN_CONFIG": {
                "research_domains": ["quantum computing"]
            }
        }
        
        data_science_config = {
            "AGENT": {
                "type": "DATA_SCIENCE",
                "name": "MachineLearningAgent"
            },
            "TRANSFORMATIONS": {
                "preprocessing": [
                    {
                        "type": "feature_engineering",
                        "params": {
                            "strategy": "polynomial",
                            "degree": 2
                        }
                    }
                ]
            }
        }
        
        # Convert to configuration schemas
        research_schema = ConfigurationSchema.from_dict(research_config)
        data_science_schema = ConfigurationSchema.from_dict(data_science_config)
        
        # Merge configurations
        merged_config = research_schema.merge(data_science_schema)
        
        # Validate merged configuration
        self.assertEqual(merged_config.name, "QuantumResearchAgent")
        self.assertEqual(merged_config.type, ConfigurationType.RESEARCH)
        
        # Validate merged transformations
        self.assertEqual(len(merged_config.transformations['input']), 1)
        self.assertEqual(len(merged_config.transformations['preprocessing']), 1)
        
        # Validate specific transformation details
        input_transformation = merged_config.transformations['input'][0]
        preprocessing_transformation = merged_config.transformations['preprocessing'][0]
        
        self.assertEqual(input_transformation['type'], 'outlier_removal')
        self.assertEqual(preprocessing_transformation['type'], 'feature_engineering')
        
        # Validate configuration
        self.assertTrue(merged_config.validate())
    
if __name__ == '__main__':
    unittest.main()
