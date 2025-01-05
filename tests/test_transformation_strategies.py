import unittest
import numpy as np
import pandas as pd
import logging

from agentflow.transformations.advanced_strategies import (
    OutlierRemovalStrategy,
    FeatureEngineeringStrategy,
    TextTransformationStrategy
)
from agentflow.transformations.specialized_strategies import (
    TimeSeriesTransformationStrategy,
    AnomalyDetectionStrategy
)

class TestTransformationStrategies(unittest.TestCase):
    def setUp(self):
        """Set up logging and test data."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Numeric test data
        self.numeric_data = pd.DataFrame({
            'A': [1, 2, 3, 100, 4, 5, 6],
            'B': [10, 20, 30, 400, 50, 60, 70]
        })
        
        # Time series test data
        self.time_series_data = pd.DataFrame({
            'value': np.random.randn(100).cumsum(),
            'timestamp': pd.date_range(start='2023-01-01', periods=100)
        }).set_index('timestamp')
        
        # Text test data
        self.text_data = [
            "Natural language processing is fascinating",
            "Machine learning transforms data"
        ]
    
    def test_outlier_removal_strategies(self):
        """Test different outlier removal strategies."""
        strategies = [
            ('z_score', 2.0),
            ('iqr', 1.5),
            ('modified_z_score', 3.0)
        ]
        
        for method, threshold in strategies:
            with self.subTest(method=method):
                strategy = OutlierRemovalStrategy(method=method, threshold=threshold)
                cleaned_data = strategy.transform(self.numeric_data)
                
                self.assertIsInstance(cleaned_data, pd.DataFrame)
                self.assertLess(len(cleaned_data), len(self.numeric_data))
                
                # Verify no extreme values remain
                for column in cleaned_data.columns:
                    self.assertFalse(any(np.abs(cleaned_data[column]) > 100))
    
    def test_feature_engineering_strategies(self):
        """Test various feature engineering strategies."""
        strategies = [
            ('polynomial', {'degree': 2}),
            ('standard', {'with_mean': True, 'with_std': True})
        ]
    
        for strategy, params in strategies:
            with self.subTest(strategy=strategy):
                feature_engineer = FeatureEngineeringStrategy(
                    strategy=strategy,
                    **params
                )
                engineered_features = feature_engineer.transform(self.numeric_data)
                
                self.assertIsNotNone(engineered_features)
                if strategy == 'polynomial':
                    # For polynomial features, we should have more columns than original
                    self.assertTrue(engineered_features.shape[1] > self.numeric_data.shape[1])
                elif strategy == 'standard':
                    # For standardization, we should have same number of columns
                    self.assertEqual(engineered_features.shape[1], self.numeric_data.shape[1])
    
    def test_text_transformation_strategies(self):
        """Test text transformation techniques."""
        # Convert text data to DataFrame
        text_df = pd.DataFrame({
            'text': self.text_data
        })
        
        method = 'normalize'  # Only supported method
        params = {
            'remove_stopwords': True,
            'lemmatize': True,
            'vectorize': True,
            'max_features': 10
        }
        
        text_transformer = TextTransformationStrategy(
            method=method,
            **params
        )
        transformed_text = text_transformer.transform(text_df)
        
        self.assertIsNotNone(transformed_text)
        self.assertIsInstance(transformed_text, pd.DataFrame)
        self.assertTrue('text_word_count' in transformed_text.columns)
        # Check for TF-IDF features if vectorization is enabled
        if params['vectorize']:
            tfidf_cols = [col for col in transformed_text.columns if 'tfidf' in col]
            # Since we have a small text dataset, we expect fewer features than max_features
            self.assertLessEqual(len(tfidf_cols), params['max_features'])
    
    def test_time_series_transformation(self):
        """Test time series transformation techniques."""
        strategies = [
            ('decomposition', {'period': 7}),
            ('rolling_features', {'window': 14}),
            ('lag_features', {'lags': [1, 7, 14]}),
            ('difference', {'order': 1})
        ]
        
        for strategy, params in strategies:
            with self.subTest(strategy=strategy):
                time_series_transformer = TimeSeriesTransformationStrategy(
                    strategy=strategy, 
                    **params
                )
                transformed_data = time_series_transformer.transform(self.time_series_data)
                
                self.assertIsInstance(transformed_data, pd.DataFrame)
                self.assertGreater(transformed_data.shape[1], 1)
    
    def test_anomaly_detection_strategies(self):
        """Test anomaly detection techniques."""
        strategies = [
            ('isolation_forest', {'contamination': 0.1}),
            ('local_outlier_factor', {'contamination': 0.1}),
            ('statistical', {}),
            ('ensemble', {'contamination': 0.1})
        ]
    
        # Create test data with some anomalies
        anomaly_data = np.random.randn(100, 3)
        anomaly_data[10:15] *= 10  # Introduce some anomalies
    
        for strategy, params in strategies:
            with self.subTest(strategy=strategy):
                anomaly_detector = AnomalyDetectionStrategy(
                    strategy=strategy,
                    **params
                )
                anomaly_results = anomaly_detector.transform(anomaly_data)
    
                self.assertIsNotNone(anomaly_results)
    
                # Verify anomaly columns are added (predictions, anomalies, anomaly_scores)
                if isinstance(anomaly_results, np.ndarray):
                    self.assertEqual(anomaly_results.shape[1], anomaly_data.shape[1] + 3)
                else:
                    self.assertTrue('predictions' in anomaly_results.columns)
                    self.assertTrue('anomalies' in anomaly_results.columns)
                    self.assertTrue('anomaly_scores' in anomaly_results.columns)
    
    def test_transformation_pipeline_integration(self):
        """Test integration of multiple transformation strategies."""
        from agentflow.agents.agent import TransformationPipeline
        
        # Create a transformation pipeline
        pipeline = TransformationPipeline()
        
        # Add strategies
        pipeline.add_strategy(
            OutlierRemovalStrategy(method='z_score', threshold=2.0)
        )
        pipeline.add_strategy(
            FeatureEngineeringStrategy(strategy='polynomial', degree=2)
        )
        
        # Apply transformation
        transformed_data = pipeline.fit_transform(self.numeric_data)
        
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertGreater(transformed_data.shape[1], self.numeric_data.shape[1])
    
    def test_error_handling(self):
        """Test error handling in transformation strategies."""
        # Test with invalid input
        with self.assertRaises(ValueError):
            strategy = OutlierRemovalStrategy(method='invalid_method')
            strategy.transform(self.numeric_data)
        
        with self.assertRaises(ValueError):
            strategy = FeatureEngineeringStrategy(strategy='invalid_strategy')
            strategy.transform(self.numeric_data)
    
    def test_logging_and_monitoring(self):
        """Test logging capabilities of transformation strategies."""
        import io
        import logging
        
        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger('test_logger')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Create strategy with custom logger
        strategy = OutlierRemovalStrategy(
            method='z_score', 
            threshold=2.0, 
            logger=logger
        )
        
        # Apply transformation
        strategy.transform(self.numeric_data)
        
        # Check log output
        log_contents = log_capture.getvalue()
        self.assertIn("Applying strategy", log_contents)

if __name__ == '__main__':
    unittest.main()
