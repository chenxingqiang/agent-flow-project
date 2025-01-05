"""Specialized transformation strategies for specific data types."""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import seasonal_decompose
from agentflow.transformations.advanced_strategies import AdvancedTransformationStrategy

class TimeSeriesTransformationStrategy(AdvancedTransformationStrategy):
    """Strategy for time series transformations."""
    
    def __init__(self, strategy: str = 'decomposition', **kwargs):
        """Initialize time series transformation strategy.
        
        Args:
            strategy: Strategy to use ('decomposition', 'rolling_features', 'lag_features', 'difference')
            **kwargs: Additional strategy-specific parameters
        """
        super().__init__(**kwargs)
        self.strategy = strategy
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform time series data.
        
        Args:
            data: Input DataFrame with time series data
            
        Returns:
            DataFrame with transformed time series
        """
        self.logger.info(f"Applying {self.strategy} time series transformation")
        
        # Ensure data is 1-dimensional
        if isinstance(data, pd.DataFrame):
            if data.shape[1] > 1:
                raise ValueError("Time series data must be 1-dimensional")
            series = data.iloc[:, 0]
        else:
            series = pd.Series(data)
        
        result = pd.DataFrame(index=series.index)
        result['original'] = series  # Always include original series
        
        if self.strategy == 'decomposition':
            period = self.params.get('period', 7)
            decomposition = seasonal_decompose(series, period=period)
            result['trend'] = decomposition.trend
            result['seasonal'] = decomposition.seasonal
            result['residual'] = decomposition.resid
        
        elif self.strategy == 'rolling_features':
            window = self.params.get('window', 14)
            result['rolling_mean'] = series.rolling(window=window).mean()
            result['rolling_std'] = series.rolling(window=window).std()
            result['rolling_min'] = series.rolling(window=window).min()
            result['rolling_max'] = series.rolling(window=window).max()
        
        elif self.strategy == 'lag_features':
            lags = self.params.get('lags', [1, 7, 14])
            for lag in lags:
                result[f'lag_{lag}'] = series.shift(lag)
        
        elif self.strategy == 'difference':
            order = self.params.get('order', 1)
            result[f'diff_{order}'] = series.diff(order)
            # Add some additional features for difference strategy
            result['diff_abs'] = result[f'diff_{order}'].abs()
            result['diff_sign'] = result[f'diff_{order}'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        else:
            raise ValueError(f"Unknown time series transformation strategy: {self.strategy}")
        
        return result

class AnomalyDetectionStrategy(AdvancedTransformationStrategy):
    """Strategy for anomaly detection."""
    
    def __init__(self, strategy: str = 'isolation_forest', **kwargs):
        """Initialize anomaly detection strategy.
        
        Args:
            strategy: Strategy to use ('isolation_forest', 'local_outlier_factor', 'statistical', 'ensemble')
            **kwargs: Additional strategy-specific parameters
        """
        super().__init__(**kwargs)
        self.strategy = strategy
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Detect anomalies in data.
        
        Args:
            data: Input data
            
        Returns:
            Data with anomaly scores/labels
        """
        self.logger.info(f"Applying {self.strategy} anomaly detection")
        
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        if self.strategy == 'isolation_forest':
            detector = IsolationForest(**self.params)
            scores = detector.fit_predict(data_array)
            anomaly_scores = detector.score_samples(data_array)  # Get anomaly scores
            
        elif self.strategy == 'local_outlier_factor':
            detector = LocalOutlierFactor(**self.params)
            scores = detector.fit_predict(data_array)
            anomaly_scores = detector.negative_outlier_factor_  # Get anomaly scores
            
        elif self.strategy == 'statistical':
            mean = np.mean(data_array, axis=0)
            std = np.std(data_array, axis=0)
            z_scores = np.abs((data_array - mean) / std)
            scores = np.where(np.any(z_scores > 3, axis=1), -1, 1)
            anomaly_scores = -np.max(z_scores, axis=1)  # Use negative max z-score as anomaly score
            
        elif self.strategy == 'ensemble':
            # Combine multiple detectors
            iforest = IsolationForest(**self.params)
            lof = LocalOutlierFactor(**self.params)
            
            iforest_scores = iforest.fit_predict(data_array)
            lof_scores = lof.fit_predict(data_array)
            
            # Majority voting
            scores = np.where((iforest_scores + lof_scores) < 0, -1, 1)
            
            # Average anomaly scores
            iforest_anomaly_scores = iforest.score_samples(data_array)
            lof_anomaly_scores = lof.negative_outlier_factor_
            anomaly_scores = (iforest_anomaly_scores + lof_anomaly_scores) / 2
            
        else:
            raise ValueError(f"Unknown anomaly detection strategy: {self.strategy}")
        
        # Add anomaly column to output
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            result['predictions'] = scores
            result['anomalies'] = scores == -1  # True for anomalies (-1), False for normal points (1)
            result['anomaly_scores'] = anomaly_scores
            return result
        else:
            return np.column_stack([data_array, scores, scores == -1, anomaly_scores])
