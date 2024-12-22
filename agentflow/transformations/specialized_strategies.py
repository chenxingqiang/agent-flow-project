import numpy as np
import pandas as pd
import scipy.stats as stats
import logging
from typing import Any, List, Optional, Union, Callable

from agentflow.transformations.advanced_strategies import AdvancedTransformationStrategy

class TimeSeriesTransformationStrategy(AdvancedTransformationStrategy):
    """
    Specialized transformation strategy for time series data.
    
    Provides advanced techniques for time series preprocessing and feature extraction.
    """
    
    def __init__(
        self, 
        strategy: str = 'decomposition', 
        period: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize time series transformation strategy.
        
        Args:
            strategy: Transformation strategy
            period: Periodicity for decomposition or resampling
            logger: Optional custom logger
        """
        super().__init__(logger)
        self.strategy = strategy
        self.period = period
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply time series transformation.
        
        Args:
            data: Input time series data
        
        Returns:
            Transformed time series data
        """
        try:
            if self.strategy == 'decomposition':
                return self._seasonal_decomposition(data)
            elif self.strategy == 'rolling_features':
                return self._rolling_features(data)
            elif self.strategy == 'lag_features':
                return self._lag_features(data)
            elif self.strategy == 'difference':
                return self._difference_transform(data)
            else:
                return data
        except Exception as e:
            self.logger.error(f"Error in time series transformation: {e}")
            return data

    def _seasonal_decomposition(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform seasonal decomposition using statsmodels.
        
        Args:
            data: Time series DataFrame
        
        Returns:
            DataFrame with decomposition components
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        if self.period is None:
            # Auto-detect periodicity
            self.period = self._detect_periodicity(data)
        
        decomposition = seasonal_decompose(data, period=self.period)
        
        # Combine decomposition results
        decomposed_df = pd.DataFrame({
            'original': data,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        })
        
        return decomposed_df
    
    def _rolling_features(self, data: pd.DataFrame, window: Optional[int] = None) -> pd.DataFrame:
        """
        Generate rolling window features.
        
        Args:
            data: Time series DataFrame
            window: Size of rolling window
        
        Returns:
            DataFrame with rolling features
        """
        if window is None:
            window = self.period or 7  # Default to weekly if no period specified
        
        rolling_features = data.rolling(window=window).agg([
            'mean', 'std', 'min', 'max', 
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75)
        ])
        
        return rolling_features
    
    def _lag_features(self, data: pd.DataFrame, lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Generate lag features.
        
        Args:
            data: Time series DataFrame
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features
        """
        if lags is None:
            lags = [1, 7, 14, 30]  # Default lag periods
        
        lag_features = data.copy()
        for lag in lags:
            lag_features[f'lag_{lag}'] = data.shift(lag)
        
        return lag_features.dropna()
    
    def _difference_transform(self, data: pd.DataFrame, order: int = 1) -> pd.DataFrame:
        """
        Apply differencing transformation to remove trend.
        
        Args:
            data: Time series DataFrame
            order: Order of differencing
        
        Returns:
            Differenced DataFrame
        """
        return data.diff(periods=order).dropna()
    
    def _detect_periodicity(self, data: pd.DataFrame) -> int:
        """
        Automatically detect time series periodicity.
        
        Args:
            data: Time series DataFrame
        
        Returns:
            Estimated periodicity
        """
        from scipy.signal import find_peaks
        
        # Compute autocorrelation
        autocorr = data.autocorr()
        peaks, _ = find_peaks(autocorr)
        
        # Return first significant peak as periodicity
        return peaks[0] if len(peaks) > 0 else 7

class AnomalyDetectionStrategy(AdvancedTransformationStrategy):
    """
    Advanced anomaly detection strategy with multiple techniques.
    
    Supports:
    - Statistical methods
    - Machine learning-based detection
    - Ensemble anomaly detection
    """
    
    def __init__(
        self, 
        strategy: str = 'isolation_forest', 
        contamination: Optional[float] = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize anomaly detection strategy.
        
        Args:
            strategy: Anomaly detection method
            contamination: Expected proportion of anomalies
            logger: Optional custom logger
        """
        super().__init__(logger)
        self.strategy = strategy
        self.contamination = contamination

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Detect and handle anomalies in the data.
        
        Args:
            data: Input data
        
        Returns:
            Data with anomalies identified or removed
        """
        try:
            if self.strategy == 'isolation_forest':
                return self._isolation_forest_detection(data)
            elif self.strategy == 'local_outlier_factor':
                return self._local_outlier_factor_detection(data)
            elif self.strategy == 'statistical':
                return self._statistical_detection(data)
            elif self.strategy == 'ensemble':
                return self._ensemble_detection(data)
            else:
                return data
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return data

    def _isolation_forest_detection(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            data: Input data
        
        Returns:
            Data with anomaly labels
        """
        from sklearn.ensemble import IsolationForest
        
        # Flatten multi-dimensional data
        if isinstance(data, pd.DataFrame):
            X = data.select_dtypes(include=[np.number]).values
        else:
            X = data
        
        clf = IsolationForest(
            contamination=self.contamination, 
            random_state=42
        )
        
        anomaly_labels = clf.fit_predict(X)
        
        # Add anomaly labels to original data
        if isinstance(data, pd.DataFrame):
            data['anomaly'] = anomaly_labels
            return data
        else:
            return np.column_stack([data, anomaly_labels])
    
    def _local_outlier_factor_detection(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Detect anomalies using Local Outlier Factor.
        
        Args:
            data: Input data
        
        Returns:
            Data with anomaly labels
        """
        from sklearn.neighbors import LocalOutlierFactor
        
        # Flatten multi-dimensional data
        if isinstance(data, pd.DataFrame):
            X = data.select_dtypes(include=[np.number]).values
        else:
            X = data
        
        clf = LocalOutlierFactor(
            n_neighbors=20, 
            contamination=self.contamination
        )
        
        anomaly_labels = clf.fit_predict(X)
        
        # Add anomaly labels to original data
        if isinstance(data, pd.DataFrame):
            data['anomaly'] = anomaly_labels
            return data
        else:
            return np.column_stack([data, anomaly_labels])
    
    def _statistical_detection(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Detect anomalies using statistical methods.
        
        Args:
            data: Input data
        
        Returns:
            Data with anomaly labels
        """
        if isinstance(data, pd.DataFrame):
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            anomalies = pd.DataFrame(index=data.index)
            
            for col in numeric_columns:
                z_scores = np.abs(stats.zscore(data[col]))
                anomalies[f'{col}_anomaly'] = z_scores > 3
            
            return pd.concat([data, anomalies], axis=1)
        else:
            z_scores = np.abs(stats.zscore(data))
            return np.column_stack([data, z_scores > 3])
    
    def _ensemble_detection(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Perform ensemble anomaly detection.
        
        Args:
            data: Input data
        
        Returns:
            Data with anomaly labels
        """
        # Combine multiple anomaly detection methods
        methods = [
            self._isolation_forest_detection,
            self._local_outlier_factor_detection,
            self._statistical_detection
        ]
        
        anomaly_results = [method(data) for method in methods]
        
        # Combine anomaly detection results
        if isinstance(data, pd.DataFrame):
            anomaly_columns = [
                col for result in anomaly_results 
                for col in result.columns if 'anomaly' in col
            ]
            
            # Majority voting for anomaly detection
            data['ensemble_anomaly'] = (
                sum(result[col] for result in anomaly_results for col in anomaly_columns) 
                >= len(methods) / 2
            )
            
            return data
        else:
            # For numpy arrays, use majority voting
            anomaly_results_array = [
                result[:, -1] if result.ndim > 1 else result 
                for result in anomaly_results
            ]
            
            ensemble_anomalies = (
                sum(anomaly_results_array) >= len(methods) / 2
            )
            
            return np.column_stack([data, ensemble_anomalies])

# Demonstration function
def demonstrate_specialized_strategies():
    """Demonstrate specialized transformation strategies."""
    # Time Series Transformation
    time_series_data = pd.DataFrame({
        'value': np.random.randn(100).cumsum(),
        'timestamp': pd.date_range(start='2023-01-01', periods=100)
    }).set_index('timestamp')
    
    time_series_transformer = TimeSeriesTransformationStrategy(
        strategy='decomposition', 
        period=7
    )
    decomposed_data = time_series_transformer.transform(time_series_data)
    print("Time Series Decomposition:\n", decomposed_data)
    
    # Anomaly Detection
    data = np.random.randn(100, 3)
    data[10:15] *= 10  # Introduce some anomalies
    
    anomaly_detector = AnomalyDetectionStrategy(
        strategy='ensemble', 
        contamination=0.1
    )
    anomaly_results = anomaly_detector.transform(data)
    print("\nAnomaly Detection Results:\n", anomaly_results)

if __name__ == "__main__":
    demonstrate_specialized_strategies()
