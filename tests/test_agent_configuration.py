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
                "name": "FinancialDataAgent"
            },
            "TRANSFORMATIONS": {
                "input": [
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
                "model_type": "regression",
                "metrics": ["r2_score", "mse"]
            }
        }
    
    def test_research_agent_configuration(self):
        """Test research agent configuration and initialization."""
        # Create agent using factory
        research_agent = AgentFactory.create_agent(self.research_config)
        
        # Verify agent configuration
        self.assertIsInstance(research_agent, Agent)
        self.assertEqual(research_agent.name, "QuantumResearchAgent")
        self.assertEqual(research_agent.agent_type, "RESEARCH")
        
        # Verify input transformations
        input_transformations = research_agent.input_transformation_pipeline
        self.assertIsNotNone(input_transformations)
        self.assertEqual(len(input_transformations.strategies), 1)
        
        # Verify domain-specific configuration
        self.assertTrue(hasattr(research_agent, 'research_domains'))
        self.assertEqual(research_agent.research_domains, ["quantum computing"])
    
    def test_data_science_agent_configuration(self):
        """Test data science agent configuration and initialization."""
        # Create agent using factory
        data_science_agent = AgentFactory.create_agent(self.data_science_config)
        
        # Verify agent configuration
        self.assertIsInstance(data_science_agent, Agent)
        self.assertEqual(data_science_agent.name, "FinancialDataAgent")
        self.assertEqual(data_science_agent.agent_type, "DATA_SCIENCE")
        
        # Verify input transformations
        input_transformations = data_science_agent.input_transformation_pipeline
        self.assertIsNotNone(input_transformations)
        self.assertEqual(len(input_transformations.strategies), 1)
        
        # Verify domain-specific configuration
        self.assertTrue(hasattr(data_science_agent, 'model_type'))
        self.assertEqual(data_science_agent.model_type, "regression")
        self.assertEqual(data_science_agent.metrics, ["r2_score", "mse"])
    
    def test_agent_factory_registration(self):
        """Test agent factory registration and agent creation."""
        # Register a custom agent type
        class CustomAgent(Agent):
            def __init__(self, config):
                super().__init__(config)
                self.custom_attribute = "custom_value"
        
        AgentFactory.register_agent("CUSTOM", CustomAgent)
        
        # Create a configuration for the custom agent
        custom_config = {
            "AGENT": {
                "type": "CUSTOM",
                "name": "CustomTestAgent"
            }
        }
        
        # Create custom agent
        custom_agent = AgentFactory.create_agent(custom_config)
        
        # Verify custom agent creation
        self.assertIsInstance(custom_agent, CustomAgent)
        self.assertEqual(custom_agent.name, "CustomTestAgent")
        self.assertEqual(custom_agent.custom_attribute, "custom_value")
    
    def test_transformation_pipeline(self):
        """Test transformation pipeline configuration and execution."""
        # Create a sample transformation pipeline
        pipeline = TransformationPipeline()
        
        # Add strategies
        outlier_strategy = OutlierRemovalStrategy(method="z_score", threshold=3.0)
        feature_strategy = FeatureEngineeringStrategy(strategy="polynomial", degree=2)
        text_strategy = TextTransformationStrategy(method="normalize", remove_stopwords=True)
        
        pipeline.add_strategy(outlier_strategy)
        pipeline.add_strategy(feature_strategy)
        pipeline.add_strategy(text_strategy)
        
        # Verify pipeline configuration
        self.assertEqual(len(pipeline.strategies), 3)
        
        # Sample data for transformation
        sample_data = pd.DataFrame({
            'A': [1, 2, 100, 4, 5],
            'B': ['hello world', 'test data', 'another example', 'text processing', 'nlp']
        })
        
        # Transform data
        transformed_data = pipeline.transform(sample_data)
        
        # Verify transformation results
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertNotEqual(transformed_data.shape, sample_data.shape)

if __name__ == '__main__':
    unittest.main()
