import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any

from agentflow.core.agent import (
    ResearchAgent, 
    DataScienceAgent, 
    AgentFactory, 
    AgentConfiguration
)
from agentflow.transformations.advanced_strategies import (
    OutlierRemovalStrategy,
    FeatureEngineeringStrategy,
    TextTransformationStrategy
)

class TestAdvancedAgents(unittest.TestCase):
    def setUp(self):
        # Sample configurations for different agent types
        self.research_config = AgentConfiguration(
            name="Test_Research_Agent",
            version="1.0.0",
            agent_type="research",
            input_specification={
                "MODES": ["CONTEXT_INJECTION"],
                "VALIDATION": {
                    "STRICT_MODE": True,
                    "SCHEMA_VALIDATION": True
                }
            }
        )
        
        self.data_science_config = AgentConfiguration(
            name="Test_Data_Science_Agent",
            version="1.0.0",
            agent_type="data_science",
            input_specification={
                "MODES": ["DIRECT_INPUT"],
                "VALIDATION": {
                    "STRICT_MODE": True,
                    "SCHEMA_VALIDATION": True
                }
            }
        )
    
    def test_research_agent_creation(self):
        """Test creation of a research agent."""
        agent = AgentFactory.create_agent(self.research_config)
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Test_Research_Agent")
        self.assertEqual(agent.version, "1.0.0")
        self.assertEqual(agent.agent_type, "research")
    
    def test_data_science_agent_creation(self):
        """Test creation of a data science agent."""
        agent = AgentFactory.create_agent(self.data_science_config)
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Test_Data_Science_Agent")
        self.assertEqual(agent.version, "1.0.0")
        self.assertEqual(agent.agent_type, "data_science")
    
    def test_outlier_removal_strategy(self):
        """Test outlier removal strategy."""
        data = pd.DataFrame({
            'A': [1, 2, 3, 100, 4, 5, 6],
            'B': [10, 20, 30, 400, 50, 60, 70]
        })
        
        # Z-score method
        z_score_remover = OutlierRemovalStrategy(method='z_score', threshold=2.0)
        cleaned_data_z = z_score_remover.transform(data)
        
        self.assertLess(len(cleaned_data_z), len(data))
        self.assertTrue(100 not in cleaned_data_z['A'].values)
        self.assertTrue(400 not in cleaned_data_z['B'].values)
        
        # IQR method
        iqr_remover = OutlierRemovalStrategy(method='iqr', threshold=1.5)
        cleaned_data_iqr = iqr_remover.transform(data)
        
        self.assertLess(len(cleaned_data_iqr), len(data))
    
    def test_feature_engineering_strategy(self):
        """Test feature engineering strategy."""
        data = pd.DataFrame({
            'X': [1, 2, 3, 4, 5],
            'Y': [10, 20, 30, 40, 50]
        })
        
        # Polynomial features
        poly_engineer = FeatureEngineeringStrategy(strategy='polynomial', degree=2)
        poly_features = poly_engineer.transform(data)
        
        self.assertIsInstance(poly_features, pd.DataFrame)
        self.assertGreater(poly_features.shape[1], data.shape[1])
        
        # Logarithmic transformation
        log_engineer = FeatureEngineeringStrategy(strategy='log')
        log_transformed = log_engineer.transform(data)
        
        self.assertTrue(np.all(log_transformed.values >= 0))
    
    def test_text_transformation_strategy(self):
        """Test text transformation strategy."""
        text_data = [
            "Natural language processing is fascinating",
            "Machine learning transforms data"
        ]
        
        # Tokenization
        tokenizer = TextTransformationStrategy(strategy='tokenize')
        tokenized_text = tokenizer.transform(text_data)
        
        self.assertEqual(len(tokenized_text), len(text_data))
        self.assertTrue(all(isinstance(tokens, list) for tokens in tokenized_text))
        
        # Lemmatization
        lemmatizer = TextTransformationStrategy(strategy='lemmatize')
        lemmatized_text = lemmatizer.transform(text_data)
        
        self.assertEqual(len(lemmatized_text), len(text_data))
        self.assertTrue(all(isinstance(tokens, list) for tokens in lemmatized_text))
    
    def test_agent_workflow_execution(self):
        """Test basic workflow execution for research and data science agents."""
        research_agent = AgentFactory.create_agent(self.research_config)
        data_science_agent = AgentFactory.create_agent(self.data_science_config)
        
        # Simulate workflow steps
        research_workflow = {
            'step_1': {
                'type': 'literature_review',
                'title': 'Initial Research Review'
            },
            'step_2': {
                'type': 'data_analysis',
                'title': 'Preliminary Data Analysis'
            }
        }
        
        data_science_workflow = {
            'step_1': {
                'type': 'data_preprocessing',
                'title': 'Data Cleaning'
            },
            'step_2': {
                'type': 'model_training',
                'title': 'Model Development'
            }
        }
        
        # Mock input data
        research_input = {"research_topic": "AI in Healthcare"}
        data_science_input = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        # Test workflow execution (mocked for demonstration)
        for step_key, step_config in research_workflow.items():
            result = research_agent._execute_step(step_config, research_input)
            self.assertIsNotNone(result)
        
        for step_key, step_config in data_science_workflow.items():
            result = data_science_agent._execute_step(step_config, data_science_input)
            self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
