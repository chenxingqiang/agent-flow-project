import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import pytest

from agentflow.core.agent import (
    Agent,
    ResearchAgent,
    DataScienceAgent
)
from agentflow.core.config_manager import AgentConfig
from agentflow.transformations.pipeline import TransformationPipeline
from agentflow.transformations.strategies import (
    OutlierRemovalStrategy,
    FeatureEngineeringStrategy,
    TextTransformationStrategy,
    AnomalyDetectionStrategy
)

class TestAdvancedAgents(unittest.TestCase):
    """Test cases for advanced agent functionality."""

    def setUp(self):
        """Set up test cases."""
        self.research_config = {
            "AGENT": {
                "name": "TestResearchAgent",
                "type": "research",
                "version": "1.0.0"
            },
            "MODEL": {
                "provider": "openai",
                "name": "gpt-4"
            },
            "WORKFLOW": {
                "max_iterations": 5
            },
            "TRANSFORMATIONS": {
                "input": [
                    {
                        "type": "text",
                        "params": {
                            "method": "clean"
                        }
                    }
                ]
            }
        }

        self.data_science_config = {
            "AGENT": {
                "name": "TestDataScienceAgent",
                "type": "data_science",
                "version": "1.0.0"
            },
            "MODEL": {
                "provider": "openai",
                "name": "gpt-4"
            },
            "WORKFLOW": {
                "max_iterations": 5
            },
            "TRANSFORMATIONS": {
                "input": [
                    {
                        "type": "feature_engineering",
                        "params": {
                            "strategy": "standard"
                        }
                    }
                ],
                "preprocessing": [
                    {
                        "type": "outlier_removal",
                        "params": {
                            "method": "z_score",
                            "threshold": 3.0
                        }
                    }
                ]
            }
        }

    def test_research_agent_creation(self):
        """Test creation of a research agent."""
        agent = ResearchAgent(self.research_config)
        self.assertIsInstance(agent, ResearchAgent)
        self.assertEqual(agent.citation_style, None)

    def test_data_science_agent_creation(self):
        """Test creation of a data science agent."""
        agent = DataScienceAgent(self.data_science_config)
        self.assertIsInstance(agent, DataScienceAgent)
        self.assertEqual(agent.model_type, None)

    def test_feature_engineering_strategy(self):
        """Test feature engineering transformation strategy."""
        strategy = FeatureEngineeringStrategy(strategy='standard')
        test_data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        transformed_data = strategy.transform(test_data)
        self.assertIsInstance(transformed_data, pd.DataFrame)

    def test_outlier_removal_strategy(self):
        """Test outlier removal transformation strategy."""
        strategy = OutlierRemovalStrategy(method='z_score')
        test_data = pd.DataFrame({
            'value': [1, 2, 3, 100, 4, 5]
        })
        transformed_data = strategy.transform(test_data)
        self.assertIsInstance(transformed_data, pd.DataFrame)

    def test_text_transformation_strategy(self):
        """Test text transformation strategy."""
        strategy = TextTransformationStrategy(strategy='clean')
        test_data = ["This is a test", "Another test string"]
        transformed_data = strategy.transform(test_data)
        self.assertIsInstance(transformed_data, list)

    def test_agent_workflow_execution(self):
        """Test basic workflow execution for research and data science agents."""
        research_agent = ResearchAgent(self.research_config)
        data_science_agent = DataScienceAgent(self.data_science_config)

        # Test research agent workflow
        text_data = ["Test research text", "Another research text"]
        research_result = research_agent.transform(text_data)
        self.assertIsInstance(research_result, list)

        # Test data science agent workflow
        numeric_data = pd.DataFrame({
            'value': [1, 2, 3, 100, 4, 5]
        })
        ds_result = data_science_agent.transform(numeric_data)
        self.assertIsInstance(ds_result, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
