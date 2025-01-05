"""Data science transformation strategies."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from .base import TransformationStrategy

class OutlierRemovalStrategy(TransformationStrategy):
    """Strategy for removing outliers from data."""
    
    def __init__(self, method: str = 'z_score', threshold: float = 3.0):
        """Initialize outlier removal strategy.
        
        Args:
            method: Method to use for outlier detection ('z_score' or 'iqr')
            threshold: Threshold for outlier detection
        """
        super().__init__()
        self.method = method
        self.threshold = threshold
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with outliers removed
        """
        if self.method == 'z_score':
            z_scores = np.abs((data - data.mean()) / data.std())
            mask = z_scores.apply(lambda x: all(x < self.threshold), axis=1)
            return data[mask]
        elif self.method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((data < (Q1 - self.threshold * IQR)) | 
                    (data > (Q3 + self.threshold * IQR))).any(axis=1)
            return data[mask]
        else:
            raise ValueError(f"Invalid method: {self.method}")

class FeatureEngineeringStrategy(TransformationStrategy):
    """Strategy for feature engineering."""
    
    def __init__(self, method: str = 'standard'):
        """Initialize feature engineering strategy.
        
        Args:
            method: Method to use for feature engineering
                   ('standard', 'minmax', 'robust', 'polynomial')
        """
        super().__init__()
        self.method = method
        self._validate_method()
        
    def _validate_method(self):
        """Validate feature engineering method."""
        valid_methods = ['standard', 'minmax', 'robust', 'polynomial']
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method: {self.method}. Must be one of {valid_methods}"
            )
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformation.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if self.method == 'standard':
            return (data - data.mean()) / data.std()
        elif self.method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif self.method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        elif self.method == 'polynomial':
            result = data.copy()
            for col in data.columns:
                result[f"{col}_squared"] = data[col] ** 2
                result[f"{col}_cubed"] = data[col] ** 3
            return result
        else:
            raise ValueError(f"Invalid method: {self.method}")

class AnomalyDetectionStrategy(TransformationStrategy):
    """Strategy for anomaly detection."""
    
    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.1):
        """Initialize anomaly detection strategy.
        
        Args:
            method: Method to use for anomaly detection
                   ('isolation_forest', 'local_outlier_factor', 'one_class_svm')
            contamination: Expected proportion of outliers in the data
        """
        super().__init__()
        self.method = method
        self.contamination = contamination
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with anomaly scores
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.svm import OneClassSVM
        
        if self.method == 'isolation_forest':
            model = IsolationForest(contamination=self.contamination)
        elif self.method == 'local_outlier_factor':
            model = LocalOutlierFactor(contamination=self.contamination)
        elif self.method == 'one_class_svm':
            model = OneClassSVM(nu=self.contamination)
        else:
            raise ValueError(f"Invalid method: {self.method}")
            
        scores = model.fit_predict(data)
        result = data.copy()
        result['anomaly_score'] = scores
        return result
