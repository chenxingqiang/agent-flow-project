import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from agentflow.core.agent import (
    Agent, 
    ResearchAgent, 
    DataScienceAgent, 
    GenericAgent,
    TransformationPipeline
)
from agentflow.transformations.advanced_strategies import (
    OutlierRemovalStrategy,
    FeatureEngineeringStrategy,
    TextTransformationStrategy
)
from agentflow.transformations.specialized_strategies import (
    TimeSeriesTransformationStrategy,
    AnomalyDetectionStrategy
)

class TestAgentFramework(unittest.TestCase):
    def setUp(self):
        """Set up common test data and configurations."""
        self.base_config = {
            "AGENT": {
                "name": "TestAgent",
                "version": "1.0.0",
                "description": "Test agent for comprehensive validation"
            },
            "WORKFLOW": {
                "max_iterations": 10,
                "timeout": 300
            }
        }
    
    def test_base_agent_initialization(self):
        """Test basic agent initialization and configuration."""
        agent = Agent(self.base_config)
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.config["AGENT"]["name"], "TestAgent")
        self.assertEqual(agent.config["WORKFLOW"]["max_iterations"], 10)
    
    def test_research_agent_workflow(self):
        """Test Research Agent workflow with transformation strategies."""
        research_config = {
            **self.base_config,
            "RESEARCH": {
                "domains": ["machine learning", "data science"],
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
        self.assertLess(len(transformed_data), len(test_data))
    
    def test_data_science_agent_advanced_transformations(self):
        """Test Data Science Agent with advanced transformation strategies."""
        ds_config = {
            **self.base_config,
            "DATA_SCIENCE": {
                "model_types": ["regression", "classification"],
                "evaluation_metrics": ["accuracy", "f1_score"]
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
        
        # Test time series data transformation
        time_series_data = pd.DataFrame({
            'value': np.random.randn(100).cumsum(),
            'timestamp': pd.date_range(start='2023-01-01', periods=100)
        }).set_index('timestamp')
        
        transformed_data = data_science_agent.transform(time_series_data)
        
        # Validate transformation
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertGreater(transformed_data.shape[1], time_series_data.shape[1])
    
    def test_generic_agent_flexible_transformations(self):
        """Test Generic Agent with flexible transformation configuration."""
        generic_config = {
            **self.base_config,
            "TRANSFORMATIONS": {
                "input": [
                    {
                        "type": "text_transformation",
                        "params": {
                            "strategy": "lemmatize"
                        }
                    }
                ],
                "output": [
                    {
                        "type": "feature_engineering",
                        "params": {
                            "strategy": "binning",
                            "bins": 5
                        }
                    }
                ]
            }
        }
        
        generic_agent = GenericAgent(generic_config)
        
        # Test text transformation
        text_data = [
            "Natural language processing is fascinating",
            "Machine learning transforms data"
        ]
        
        transformed_text = generic_agent.transform(text_data)
        
        self.assertIsNotNone(transformed_text)
        self.assertEqual(len(transformed_text), len(text_data))
    
    def test_transformation_pipeline_integration(self):
        """Test comprehensive transformation pipeline integration."""
        pipeline = TransformationPipeline()
        
        # Add multiple strategies
        pipeline.add_strategy(
            OutlierRemovalStrategy(method='z_score', threshold=2.0)
        )
        pipeline.add_strategy(
            FeatureEngineeringStrategy(strategy='polynomial', degree=2)
        )
        pipeline.add_strategy(
            AnomalyDetectionStrategy(strategy='isolation_forest', contamination=0.1)
        )
        
        # Test data
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 100, 4, 5, 6],
            'B': [10, 20, 30, 400, 50, 60, 70]
        })
        
        # Apply transformation pipeline
        transformed_data = pipeline.transform(test_data)
        
        # Validate transformation
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertGreater(transformed_data.shape[1], test_data.shape[1])
    
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
            with self.assertRaises((ValueError, KeyError)):
                Agent(invalid_config)
    
    def test_custom_transformation_strategy(self):
        """Test ability to create and use custom transformation strategies."""
        class CustomTransformationStrategy:
            def __init__(self, scaling_factor=1.0):
                self.scaling_factor = scaling_factor
            
            def transform(self, data):
                """Apply custom scaling transformation."""
                return data * self.scaling_factor
        
        # Create a transformation pipeline with custom strategy
        pipeline = TransformationPipeline()
        pipeline.add_strategy(CustomTransformationStrategy(scaling_factor=2.0))
        
        # Test data
        test_data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        
        # Apply transformation
        transformed_data = pipeline.transform(test_data)
        
        # Validate transformation
        self.assertTrue(np.all(transformed_data['value'] == test_data['value'] * 2.0))

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
        
        # Prepare test data with complex characteristics
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 100, 4, 5, 6, 200],
            'B': [10, 20, 30, 400, 50, 60, 70, 500],
            'C': [0.1, 0.2, 0.3, 5.0, 0.4, 0.5, 0.6, 10.0]
        })
        
        # Apply transformation pipeline
        transformed_data = pipeline.transform(test_data)
        
        # Advanced validation checks
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertLess(len(transformed_data), len(test_data))  # Outlier removal
        self.assertGreater(transformed_data.shape[1], test_data.shape[1])  # Feature engineering
        
        # Check for anomaly column
        self.assertTrue('anomaly' in transformed_data.columns)
    
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
                "RESEARCH": {
                    "domains": ["quantum computing"],
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
                "context": {"strategy": "scaling", "scale_factor": 2.0},
                "data": pd.DataFrame({"value": [1, 2, 3, 4, 5]})
            },
            {
                "context": {"strategy": "normalization"},
                "data": pd.DataFrame({"value": [10, 20, 30, 40, 50]})
            },
            {
                "context": {"strategy": "categorical_encoding"},
                "data": pd.DataFrame({"category": ["A", "B", "A", "C", "B"]})
            }
        ]
        
        for scenario in test_scenarios:
            with self.subTest(scenario=scenario):
                custom_strategy = ContextAwareTransformationStrategy(
                    context=scenario['context']
                )
                pipeline.add_strategy(custom_strategy)
                
                transformed_data = pipeline.transform(scenario['data'])
                
                self.assertIsInstance(transformed_data, pd.DataFrame)
                self.assertNotEqual(transformed_data.shape, scenario['data'].shape)

if __name__ == '__main__':
    unittest.main()
