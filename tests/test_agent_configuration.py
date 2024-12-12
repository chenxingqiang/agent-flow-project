import unittest
import json
import os
import tempfile
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

from agentflow.core.agent import (
    Agent,
    ResearchAgent,
    DataScienceAgent,
    AgentFactory,
    TransformationPipeline,
    AgentTransformationMixin
)
from agentflow.transformations.advanced_strategies import (
    OutlierRemovalStrategy,
    FeatureEngineeringStrategy,
    TextTransformationStrategy
)

class TestAgentConfiguration(unittest.TestCase):
    def setUp(self):
        """Set up test environment with sample configurations."""
        # Research Agent Configuration
        self.research_config = {
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
        
        # Data Science Agent Configuration
        self.data_science_config = {
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
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def test_research_agent_creation(self):
        """
        Test creating a Research Agent with specific configuration.
        
        Validates:
        - Agent creation
        - Configuration parsing
        - Transformation pipeline setup
        """
        # Create Research Agent
        research_agent = ResearchAgent(self.research_config)
        
        # Validate basic agent properties
        self.assertEqual(research_agent.name, "QuantumResearchAgent")
        self.assertEqual(research_agent.agent_type, "RESEARCH")
        
        # Validate input transformation pipeline
        self.assertEqual(
            len(research_agent.input_transformation_pipeline.strategies), 
            1
        )
        
        # Validate domain configuration
        self.assertEqual(
            research_agent.config.domain_config.get('research_domains'), 
            ["quantum computing"]
        )
    
    def test_data_science_agent_creation(self):
        """
        Test creating a Data Science Agent with specific configuration.
        
        Validates:
        - Agent creation
        - Configuration parsing
        - Transformation pipeline setup
        """
        # Create Data Science Agent
        data_science_agent = DataScienceAgent(self.data_science_config)
        
        # Validate basic agent properties
        self.assertEqual(data_science_agent.name, "MachineLearningAgent")
        self.assertEqual(data_science_agent.agent_type, "DATA_SCIENCE")
        
        # Validate preprocessing transformation pipeline
        self.assertEqual(
            len(data_science_agent.preprocessing_transformation_pipeline.strategies), 
            1
        )
        
        preprocessing_strategy = data_science_agent.preprocessing_transformation_pipeline.strategies[0]
        self.assertIsInstance(preprocessing_strategy, FeatureEngineeringStrategy)
        self.assertEqual(preprocessing_strategy.strategy, "polynomial")
        self.assertEqual(preprocessing_strategy.degree, 2)
    
    def test_agent_factory_creation(self):
        """
        Test Agent Factory for creating different agent types.
        
        Validates:
        - Agent Factory functionality
        - Dynamic agent creation
        - Configuration-based agent instantiation
        """
        # Test Research Agent creation
        research_agent = AgentFactory.create_agent(self.research_config)
        self.assertIsInstance(research_agent, ResearchAgent)
        
        # Test Data Science Agent creation
        data_science_agent = AgentFactory.create_agent(self.data_science_config)
        self.assertIsInstance(data_science_agent, DataScienceAgent)
    
    def test_transformation_pipeline_integration(self):
        """
        Test comprehensive transformation pipeline integration.
        
        Validates:
        - Multiple transformation strategy chaining
        - Complex data transformation
        - Strategy configuration
        """
        # Prepare test data
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 100, 4, 5, 6],
            'B': [10, 20, 30, 400, 50, 60, 70]
        })
        
        # Create transformation pipeline
        pipeline = TransformationPipeline()
        
        # Add multiple strategies
        pipeline.add_strategy(
            OutlierRemovalStrategy(method='z_score', threshold=2.0)
        )
        pipeline.add_strategy(
            FeatureEngineeringStrategy(strategy='polynomial', degree=2)
        )
        
        # Apply transformation
        transformed_data = pipeline.transform(test_data)
        
        # Validate transformation
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertLess(len(transformed_data), len(test_data))  # Outlier removal
        self.assertGreater(transformed_data.shape[1], test_data.shape[1])  # Feature engineering
    
    def test_agent_transformation_mixin(self):
        """
        Test AgentTransformationMixin with custom transformation strategies.
        
        Validates:
        - Mixin transformation capabilities
        - Dynamic strategy configuration
        - Multi-stage transformation
        """
        class CustomAgent(AgentTransformationMixin, Agent):
            def __init__(self, config):
                super().__init__(config)
                
                # Configure custom transformations
                self.configure_input_transformation([
                    {
                        'type': 'outlier_removal',
                        'params': {
                            'method': 'modified_z_score',
                            'threshold': 3.5
                        }
                    }
                ])
                
                self.configure_preprocessing_transformation([
                    {
                        'type': 'feature_engineering',
                        'params': {
                            'strategy': 'log'
                        }
                    }
                ])
                
                self.configure_output_transformation([
                    {
                        'type': 'text_transformation',
                        'params': {
                            'strategy': 'lemmatize'
                        }
                    }
                ])
        
        # Create custom agent
        custom_agent = CustomAgent(self.research_config)
        
        # Prepare test data
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 100, 4, 5, 6],
            'B': [10, 20, 30, 400, 50, 60, 70]
        })
        
        # Test input transformation
        transformed_input = custom_agent.transform_input(test_data)
        self.assertIsInstance(transformed_input, pd.DataFrame)
        
        # Test preprocessing
        preprocessed_data = custom_agent.preprocess_data(transformed_input)
        self.assertIsInstance(preprocessed_data, pd.DataFrame)
    
    def test_configuration_file_loading(self):
        """
        Test loading agent configuration from a file.
        
        Validates:
        - Configuration file parsing
        - Dynamic configuration loading
        - File-based agent instantiation
        """
        # Create a temporary configuration file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_config_file:
            json.dump(self.research_config, temp_config_file)
            temp_config_path = temp_config_file.name
        
        try:
            # Load agent from configuration file
            agent = Agent.from_config(temp_config_path)
            
            # Validate agent properties
            self.assertEqual(agent.name, "QuantumResearchAgent")
            self.assertEqual(agent.agent_type, "RESEARCH")
        finally:
            # Clean up temporary file
            os.unlink(temp_config_path)
    
    def test_advanced_configuration_scenarios(self):
        """
        Test advanced configuration scenarios with complex settings.
        
        Validates:
        - Multiple transformation strategies
        - Domain-specific configurations
        - Complex workflow settings
        """
        advanced_config = {
            "AGENT": {
                "type": "RESEARCH",
                "name": "AdvancedResearchAgent"
            },
            "WORKFLOW": {
                "max_iterations": 15,
                "timeout": 600,
                "retry_strategy": {
                    "max_retries": 3,
                    "backoff_factor": 2
                }
            },
            "TRANSFORMATIONS": {
                "input": [
                    {
                        "type": "outlier_removal",
                        "params": {
                            "method": "modified_z_score",
                            "threshold": 3.5
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
                ]
            },
            "DOMAIN_CONFIG": {
                "research_domains": ["quantum computing", "AI ethics"],
                "publication_targets": {
                    "conferences": ["NeurIPS", "ICML"],
                    "journals": ["Nature", "Science"]
                }
            }
        }
        
        # Create agent with advanced configuration
        advanced_agent = ResearchAgent(advanced_config)
        
        # Validate workflow configuration
        self.assertEqual(advanced_agent.workflow.max_iterations, 15)
        self.assertEqual(advanced_agent.workflow.timeout, 600)
        
        # Validate domain configuration
        domain_config = advanced_agent.config.domain_config
        self.assertEqual(
            domain_config.get('research_domains'), 
            ["quantum computing", "AI ethics"]
        )
        self.assertIn("publication_targets", domain_config)
        
        # Validate transformation strategies
        self.assertEqual(
            len(advanced_agent.input_transformation_pipeline.strategies), 
            1
        )
        self.assertEqual(
            len(advanced_agent.preprocessing_transformation_pipeline.strategies), 
            1
        )

if __name__ == '__main__':
    unittest.main()
