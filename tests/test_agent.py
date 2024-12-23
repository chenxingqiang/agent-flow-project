"""Test cases for the agent framework."""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from agentflow.core.agent import (
    Agent,
    DataScienceAgent,
    ResearchAgent,
    GenericAgent,
    TransformationPipeline
)
from agentflow.transformations.advanced_strategies import (
    OutlierRemovalStrategy,
    FeatureEngineeringStrategy,
    TextTransformationStrategy,
    AnomalyDetectionStrategy,
    TimeSeriesTransformationStrategy
)

class TestAgentFramework(unittest.TestCase):
    """Test cases for Agent Framework core functionality."""

    def setUp(self):
        """Set up test cases."""
        self.base_config = {
            "AGENT": {
                "name": "TestAgent",
                "type": "generic",
                "version": "1.0.0"
            },
            "MODEL": {
                "provider": "openai",
                "name": "gpt-4",
                "temperature": 0.7
            },
            "WORKFLOW": {
                "max_iterations": 5,
                "logging_level": "INFO"
            }
        }

    def test_base_agent_initialization(self):
        """Test basic agent initialization."""
        agent = Agent(self.base_config)
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.version, "1.0.0")
        self.assertIsNotNone(agent.model)
        self.assertIsNotNone(agent.workflow)

    def test_agent_configuration_validation(self):
        """Test agent configuration validation and error handling."""
        invalid_configs = [
            # Missing required configuration keys
            {},
            # Invalid transformation strategy
            {
                "AGENT": {"name": "InvalidAgent"},
                "TRANSFORMATIONS": {
                    "input": [{"type": "non_existent_strategy"}]
                }
            }
        ]

        for invalid_config in invalid_configs:
            with self.assertRaises(ValueError):
                Agent(invalid_config)

    def test_research_agent_workflow(self):
        """Test Research Agent workflow with transformation strategies."""
        research_config = {
            **self.base_config,
            "DOMAIN_CONFIG": {
                "research_domains": ["machine learning", "data science"],
                "citation_style": "APA"
            },
            "TRANSFORMATIONS": {
                "input": [
                    {
                        "type": "outlier_removal",
                        "params": {
                            "method": "z_score",
                            "threshold": 2.5
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
            }
        }

        research_agent = ResearchAgent(research_config)

        # Verify transformation pipeline configuration
        self.assertTrue(hasattr(research_agent, 'transformation_pipeline'))
        self.assertIsInstance(
            research_agent.transformation_pipeline,
            TransformationPipeline
        )

        # Simulate research workflow
        test_data = pd.DataFrame({
            'citations': [10, 20, 300, 40, 50],
            'impact_factor': [0.5, 1.0, 5.0, 2.0, 1.5]
        })

        transformed_data = research_agent.transform(test_data)

        # Validate transformation
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertEqual(len(transformed_data), len(test_data))

    def test_data_science_agent_advanced_transformations(self):
        """Test Data Science Agent with advanced transformation strategies."""
        ds_config = {
            **self.base_config,
            "DOMAIN_CONFIG": {
                "model_type": "regression",
                "metrics": ["accuracy", "f1_score"]
            },
            "TRANSFORMATIONS": {
                "input": [
                    {
                        "type": "anomaly_detection",
                        "params": {
                            "strategy": "isolation_forest",
                            "contamination": 0.1
                        }
                    }
                ],
                "preprocessing": [
                    {
                        "type": "time_series",
                        "params": {
                            "strategy": "rolling_features",
                            "window": 14
                        }
                    }
                ]
            }
        }

        data_science_agent = DataScienceAgent(ds_config)
        self.assertIsNotNone(data_science_agent.transformation_pipeline)

        # Test transformation
        test_data = pd.DataFrame({
            'value': np.random.randn(100),
            'timestamp': pd.date_range(start='2023-01-01', periods=100)
        })
        transformed_data = data_science_agent.transform(test_data)
        self.assertIsInstance(transformed_data, pd.DataFrame)

    def test_transformation_pipeline_integration(self):
        """Test integration of transformation pipeline with agents."""
        pipeline = TransformationPipeline()
        pipeline.add_strategy(OutlierRemovalStrategy(method='z_score'))
        pipeline.add_strategy(FeatureEngineeringStrategy(strategy='standard'))

        test_data = pd.DataFrame({
            'A': [1, 2, 100, 4, 5],
            'B': [0.1, 0.2, 0.3, 0.4, 10.0]
        })

        transformed_data = pipeline.transform(test_data)
        self.assertIsInstance(transformed_data, pd.DataFrame)

    def test_generic_agent_flexible_transformations(self):
        """Test generic agent with flexible transformation configurations."""
        config = {
            **self.base_config,
            "TRANSFORMATIONS": {
                "input": [
                    {"type": "feature_engineering", "params": {"strategy": "standard"}}
                ]
            }
        }

        agent = Agent(config)
        test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        transformed_data = agent.transform(test_data)
        self.assertIsInstance(transformed_data, pd.DataFrame)

    def test_custom_transformation_strategy(self):
        """Test implementation of custom transformation strategy."""
        class CustomStrategy:
            def transform(self, data):
                if isinstance(data, pd.DataFrame):
                    return data * 2
                return data

        pipeline = TransformationPipeline()
        pipeline.add_strategy(CustomStrategy())

        test_data = pd.DataFrame({'A': [1, 2, 3]})
        result = pipeline.transform(test_data)
        self.assertTrue((result == test_data * 2).all().all())

class TestAgentFrameworkExtended(unittest.TestCase):
    def setUp(self):
        """Set up comprehensive test environment with advanced configurations."""
        self.advanced_config = {
            "AGENT": {
                "name": "AdvancedTestAgent",
                "version": "2.0.0",
                "description": "Comprehensive agent testing framework"
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
                            "degree": 3
                        }
                    }
                ],
                "output": [
                    {
                        "type": "text_transformation",
                        "params": {
                            "strategy": "lemmatize"
                        }
                    }
                ]
            }
        }
    
    def test_advanced_transformation_strategy_composition(self):
        """
        Test complex composition of transformation strategies with multiple layers.
        Validates strategy chaining, parameter flexibility, and interdependence.
        """
        # Create a multi-stage transformation pipeline
        pipeline = TransformationPipeline()

        # Add sophisticated strategies with complex configurations
        pipeline.add_strategy(
            OutlierRemovalStrategy(
                method='modified_z_score',
                threshold=3.5,
                handling_strategy='replace'  # New advanced parameter
            )
        )
        pipeline.add_strategy(
            FeatureEngineeringStrategy(
                strategy='polynomial',
                degree=3,
                interaction_terms=True  # Advanced feature engineering
            )
        )
        pipeline.add_strategy(
            AnomalyDetectionStrategy(
                strategy='ensemble',
                contamination=0.05,
                detection_methods=['isolation_forest', 'local_outlier_factor']
            )
        )

        # Create test data
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })

        # Apply transformations
        transformed_data = pipeline.transform(data)

        # Verify transformations were applied
        self.assertIn('anomaly_score', transformed_data.columns)
        self.assertEqual(len(transformed_data), len(data))
        self.assertTrue(all(transformed_data['anomaly_score'].between(-1, 1)))

    def test_flexible_configuration_scenarios(self):
        """
        Test multiple configuration scenarios to validate flexibility.
        Covers different agent types, transformation strategies, and edge cases.
        """
        test_scenarios = [
            # Minimal configuration
            {},
            
            # Research agent with specific domain constraints
            {
                "AGENT": {"type": "research"},
                "DOMAIN_CONFIG": {
                    "research_domains": ["quantum computing"],
                    "citation_style": "IEEE"
                },
                "TRANSFORMATIONS": {
                    "input": [
                        {"type": "text_transformation", "params": {"strategy": "tokenize"}}
                    ]
                }
            },
            
            # Data science agent with complex model configuration
            {
                "AGENT": {"type": "data_science"},
                "DATA_SCIENCE": {
                    "model_types": ["deep_learning"],
                    "neural_network": {
                        "layers": [64, 32],
                        "activation": "relu"
                    }
                },
                "TRANSFORMATIONS": {
                    "preprocessing": [
                        {
                            "type": "time_series",
                            "params": {
                                "strategy": "seasonal_decomposition",
                                "period": 12
                            }
                        }
                    ]
                }
            }
        ]
        
        for scenario in test_scenarios:
            with self.subTest(scenario=scenario):
                try:
                    # Attempt to create agent with flexible configuration
                    agent = Agent(scenario)
                    self.assertIsNotNone(agent)
                except (ValueError, KeyError) as e:
                    # Some scenarios might be intentionally incomplete
                    print(f"Configuration scenario validation: {e}")
    
    def test_robust_error_handling(self):
        """
        Comprehensive error handling test suite.
        Validates graceful handling of various error scenarios.
        """
        error_test_cases = [
            # Invalid transformation strategy
            {
                "config": {
                    "TRANSFORMATIONS": {
                        "input": [{"type": "non_existent_strategy"}]
                    }
                },
                "expected_error": ValueError
            },
            
            # Incompatible data type for transformation
            {
                "config": {},
                "data": "invalid_data_type",
                "expected_error": ValueError
            },
            
            # Workflow execution failure
            {
                "config": {
                    "WORKFLOW": {
                        "max_iterations": -1  # Invalid configuration
                    }
                },
                "expected_error": ValueError
            }
        ]
        
        for case in error_test_cases:
            with self.subTest(case=case):
                with self.assertRaises(case.get('expected_error', Exception)):
                    agent = Agent(case.get('config', {}))
                    if 'data' in case:
                        agent.transform(case['data'])
    
    def test_custom_transformation_extensibility(self):
        """
        Advanced test for custom transformation strategy extensibility.
        Demonstrates creating complex, context-aware transformation strategies.
        """
        class ContextAwareTransformationStrategy:
            def __init__(self, context: Dict[str, Any] = None):
                self.context = context or {}

            def transform(self, data):
                """
                Context-aware transformation with dynamic strategy selection.
                
                Args:
                    data (pd.DataFrame): Input data to transform
                
                Returns:
                    pd.DataFrame: Transformed data
                """
                if not isinstance(data, pd.DataFrame):
                    raise ValueError("Input must be a pandas DataFrame")
                
                # Dynamic transformation based on context
                strategy = self.context.get('strategy', 'default')
                
                if strategy == 'scaling':
                    scale_factor = self.context.get('scale_factor', 1.0)
                    return data * scale_factor
                
                elif strategy == 'normalization':
                    return (data - data.mean()) / data.std()
                
                elif strategy == 'categorical_encoding':
                    # Example of categorical encoding
                    return pd.get_dummies(data)
                
                # Default passthrough
                return data
        
        # Create pipeline with custom strategy
        pipeline = TransformationPipeline()

        # Test different context-aware transformations
        test_scenarios = [
            {
                "context": {"strategy": "categorical_encoding"},
                "data": pd.DataFrame({"category": ["A", "B", "A", "C", "B"]})
            },
            {
                "context": {"strategy": "scaling", "scale_factor": 2.0},
                "data": pd.DataFrame({"value": [1, 2, 3, 4, 5]})
            },
            {
                "context": {"strategy": "normalization"},
                "data": pd.DataFrame({"value": [10, 20, 30, 40, 50]})
            }
        ]

        for scenario in test_scenarios:
            with self.subTest(scenario=scenario):
                custom_strategy = ContextAwareTransformationStrategy(
                    context=scenario['context']
                )
                pipeline = TransformationPipeline()  # Create new pipeline for each scenario
                pipeline.add_strategy(custom_strategy)

                transformed_data = pipeline.transform(scenario['data'])
                self.assertIsInstance(transformed_data, pd.DataFrame)

                # For categorical encoding, check if shape changed due to one-hot encoding
                if scenario['context']['strategy'] == 'categorical_encoding':
                    self.assertGreater(transformed_data.shape[1], scenario['data'].shape[1])

if __name__ == '__main__':
    unittest.main()
