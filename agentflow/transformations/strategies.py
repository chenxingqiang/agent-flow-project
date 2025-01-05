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

    def __init__(self, method: str = 'z_score', threshold: float = 3.0, handling_strategy: str = 'remove'):
        """Initialize outlier removal strategy."""
        self.method = method
        self.threshold = threshold
        self.handling_strategy = handling_strategy

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from data."""
        if not isinstance(data, pd.DataFrame):
            return data

        if self.method == 'z_score':
            return self._z_score_removal(data)
        elif self.method == 'iqr':
            return self._iqr_removal(data)
        elif self.method == 'isolation_forest':
            return self._isolation_forest_removal(data)
        return data

    def _z_score_removal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(data[numeric_cols].apply(lambda x: (x - x.mean()) / x.std()))
        mask = (z_scores <= self.threshold).all(axis=1)
        return data[mask]

    def _iqr_removal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        Q1 = data[numeric_cols].quantile(0.25)
        Q3 = data[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((data[numeric_cols] < (Q1 - self.threshold * IQR)) | 
                (data[numeric_cols] > (Q3 + self.threshold * IQR))).any(axis=1)
        return data[mask]

    def _isolation_forest_removal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Isolation Forest."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return data
            
        clf = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = clf.fit_predict(data[numeric_cols])
        return data[outlier_labels == 1]


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


class BaseStrategy:
    """Base class for transformation strategies."""
    
    def transform(self, data: Any) -> Any:
        """Transform input data."""
        raise NotImplementedError("Subclasses must implement transform method")


class AnomalyDetectionStrategy(BaseStrategy):
    """Strategy for anomaly detection."""
    
    def __init__(self, strategy: str = 'isolation_forest', contamination: float = 0.1, **kwargs):
        self.strategy = strategy
        self.contamination = contamination
        self.kwargs = kwargs
        self.detector = None
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if self.strategy == 'isolation_forest':
            self.detector = IsolationForest(contamination=self.contamination, **self.kwargs)
            anomaly_labels = self.detector.fit_predict(data)
            data['anomaly_score'] = anomaly_labels
            return data
        else:
            raise ValueError(f"Unknown anomaly detection strategy: {self.strategy}")


class FeatureEngineeringStrategy(BaseStrategy):
    """Strategy for feature engineering."""
    
    def __init__(self, strategy: str = 'standard', **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs
        self.scaler = None
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if self.strategy == 'standard':
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(data)
            return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        elif self.strategy == 'polynomial':
            degree = self.kwargs.get('degree', 2)
            interaction_terms = self.kwargs.get('interaction_terms', False)
            
            result = data.copy()
            for col in data.columns:
                for i in range(2, degree + 1):
                    result[f"{col}_power_{i}"] = data[col] ** i
                    
            if interaction_terms:
                for i, col1 in enumerate(data.columns):
                    for col2 in data.columns[i+1:]:
                        result[f"{col1}_{col2}_interaction"] = data[col1] * data[col2]
                        
            return result
        else:
            raise ValueError(f"Unknown feature engineering strategy: {self.strategy}")


class OutlierRemovalStrategy(BaseStrategy):
    """Strategy for outlier removal."""
    
    def __init__(self, method: str = 'z_score', threshold: float = 3.0, handling_strategy: str = 'remove'):
        self.method = method
        self.threshold = threshold
        self.handling_strategy = handling_strategy
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if self.method == 'z_score':
            z_scores = np.abs((data - data.mean()) / data.std())
            if self.handling_strategy == 'remove':
                return data[~(z_scores > self.threshold).any(axis=1)]
            elif self.handling_strategy == 'replace':
                mask = z_scores > self.threshold
                data[mask] = data.mean()[mask.columns]
                return data
        elif self.method == 'modified_z_score':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * np.abs(data - median) / mad
            if self.handling_strategy == 'remove':
                return data[~(modified_z_scores > self.threshold).any(axis=1)]
            elif self.handling_strategy == 'replace':
                mask = modified_z_scores > self.threshold
                data[mask] = median[mask.columns]
                return data
        else:
            raise ValueError(f"Unknown outlier removal method: {self.method}")


class TimeSeriesStrategy(BaseStrategy):
    """Strategy for time series transformations."""
    
    def __init__(self, strategy: str = 'rolling_features', window: int = 3):
        self.strategy = strategy
        self.window = window
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if self.strategy == 'rolling_features':
            result = data.copy()
            for col in data.columns:
                result[f"{col}_rolling_mean"] = data[col].rolling(window=self.window).mean()
                result[f"{col}_rolling_std"] = data[col].rolling(window=self.window).std()
                result[f"{col}_rolling_min"] = data[col].rolling(window=self.window).min()
                result[f"{col}_rolling_max"] = data[col].rolling(window=self.window).max()
            return result.fillna(method='bfill')
        else:
            raise ValueError(f"Unknown time series strategy: {self.strategy}")
