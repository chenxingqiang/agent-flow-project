"""Transformation strategies for data processing."""

from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import logging

class TransformationStrategy:
    """Base class for all transformation strategies."""

    def transform(self, data: Any) -> Any:
        """Transform input data."""
        raise NotImplementedError


class TextTransformationStrategy(TransformationStrategy):
    """Strategy for text data transformations."""

    def __init__(self, strategy: str = 'clean', **kwargs):
        """Initialize text transformation strategy."""
        self.strategy = strategy
        self.kwargs = kwargs

    def transform(self, data: Any) -> Any:
        """Transform text data."""
        if self.strategy == 'clean':
            return self._clean_text(data)
        elif self.strategy == 'tokenize':
            return self._tokenize_text(data)
        elif self.strategy == 'lemmatize':
            return self._lemmatize_text(data)
        return data

    def _clean_text(self, text: str) -> str:
        """Clean text by removing special characters."""
        if not isinstance(text, str):
            return text
        return ' '.join(text.split())

    def _tokenize_text(self, text: str) -> list:
        """Split text into tokens."""
        if not isinstance(text, str):
            return text
        return text.split()

    def _lemmatize_text(self, text: str) -> str:
        """Lemmatize text (placeholder implementation)."""
        return text


class FeatureEngineeringStrategy(TransformationStrategy):
    """Strategy for feature engineering."""

    def __init__(self, strategy: str = 'standard', **kwargs):
        """Initialize feature engineering strategy."""
        self.strategy = strategy
        self.kwargs = kwargs
        self.scaler = StandardScaler() if strategy == 'standard' else None

    def transform(self, data: Any) -> Any:
        """Transform features."""
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            return data

        if self.strategy == 'standard':
            return self._standardize(data)
        elif self.strategy == 'polynomial':
            return self._polynomial_features(data)
        elif self.strategy == 'binning':
            return self._binning(data)
        return data

    def _standardize(self, data: Any) -> Any:
        """Standardize numerical features."""
        try:
            if isinstance(data, pd.DataFrame):
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
                return data
            return self.scaler.fit_transform(data)
        except Exception:
            return data

    def _polynomial_features(self, data: Any) -> Any:
        """Generate polynomial features."""
        degree = self.kwargs.get('degree', 2)
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                data[f"{col}_squared"] = data[col] ** 2
                if degree > 2:
                    data[f"{col}_cubed"] = data[col] ** 3
        return data

    def _binning(self, data: Any) -> Any:
        """Bin numerical features."""
        bins = self.kwargs.get('bins', 5)
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                data[f"{col}_binned"] = pd.qcut(data[col], bins, labels=False, duplicates='drop')
        return data


class OutlierRemovalStrategy(TransformationStrategy):
    """Strategy for outlier removal."""

    def __init__(self, method: str = 'z_score', **kwargs):
        """Initialize outlier removal strategy."""
        self.method = method
        self.threshold = kwargs.get('threshold', 3)
        self.handling_strategy = kwargs.get('handling_strategy', 'remove')

    def transform(self, data: Any) -> Any:
        """Remove outliers from data."""
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            return data

        if self.method == 'z_score':
            return self._z_score_removal(data)
        elif self.method == 'iqr':
            return self._iqr_removal(data)
        elif self.method == 'isolation_forest':
            return self._isolation_forest_removal(data)
        return data

    def _z_score_removal(self, data: Any) -> Any:
        """Remove outliers using z-score method."""
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std())
            mask = (z_scores < self.threshold).all(axis=1)
            return data[mask] if self.handling_strategy == 'remove' else data

        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        mask = z_scores < self.threshold
        return data[mask] if self.handling_strategy == 'remove' else data

    def _iqr_removal(self, data: Any) -> Any:
        """Remove outliers using IQR method."""
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            Q1 = data[numeric_cols].quantile(0.25)
            Q3 = data[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            return data[mask] if self.handling_strategy == 'remove' else data
        return data

    def _isolation_forest_removal(self, data: Any) -> Any:
        """Remove outliers using Isolation Forest."""
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            iso_forest = IsolationForest(contamination=self.threshold, random_state=42)
            mask = iso_forest.fit_predict(data[numeric_cols]) == 1
            return data[mask] if self.handling_strategy == 'remove' else data
        return data


class AnomalyDetectionStrategy:
    """Strategy for detecting anomalies in data."""
    
    def __init__(self, strategy: str = 'isolation_forest', contamination: float = 0.1, **kwargs):
        """
        Initialize anomaly detection strategy.
        
        Args:
            strategy: Detection method ('isolation_forest', 'local_outlier_factor', 'ensemble')
            contamination: Expected proportion of outliers in the dataset
            **kwargs: Additional parameters for specific strategies
        """
        self.strategy = strategy
        self.contamination = contamination
        self.kwargs = kwargs
        self.detectors = []
        
        if strategy == 'ensemble':
            # Initialize ensemble of detectors
            self.detectors = [
                IsolationForest(contamination=contamination, **kwargs),
                LocalOutlierFactor(contamination=contamination, **kwargs)
            ]
        else:
            # Initialize single detector
            if strategy == 'isolation_forest':
                self.detectors = [IsolationForest(contamination=contamination, **kwargs)]
            elif strategy == 'local_outlier_factor':
                self.detectors = [LocalOutlierFactor(contamination=contamination, **kwargs)]
            else:
                raise ValueError(f"Unknown anomaly detection strategy: {strategy}")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply anomaly detection to the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with anomalies detected
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        try:
            # Get predictions from all detectors
            predictions = []
            for detector in self.detectors:
                if isinstance(detector, LocalOutlierFactor):
                    pred = detector.fit_predict(data)
                else:
                    pred = detector.fit(data).predict(data)
                predictions.append(pred)
            
            # Combine predictions (majority voting for ensemble)
            if len(predictions) > 1:
                final_pred = np.mean([pred == -1 for pred in predictions], axis=0) >= 0.5
            else:
                final_pred = predictions[0] == -1
                
            # Add anomaly flag to DataFrame
            data['is_anomaly'] = final_pred
            
            return data
            
        except Exception as e:
            logging.error(f"Anomaly detection failed: {str(e)}")
            return data
