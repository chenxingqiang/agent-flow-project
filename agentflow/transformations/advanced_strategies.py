"""Advanced transformation strategies for data processing."""

import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

class AdvancedTransformationStrategy(abc.ABC):
    """
    Advanced base class for sophisticated data transformation strategies.
    
    This abstract base class provides a framework for implementing complex
    data transformation techniques with robust error handling and logging.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, **kwargs):
        """
        Initialize the transformation strategy.
        
        Args:
            logger: Optional custom logger. If not provided, a default logger is created.
            **kwargs: Additional configuration parameters
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = kwargs
    
    @abc.abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Abstract method for transforming input data.
        
        Args:
            data: Input data to transform
        
        Returns:
            Transformed data
        
        Raises:
            ValueError: If transformation fails
        """
        pass

class OutlierRemovalStrategy(AdvancedTransformationStrategy):
    """Advanced outlier removal strategy using multiple statistical techniques."""
    
    def __init__(self, method: str = 'z_score', threshold: float = 3.0, **kwargs):
        """Initialize outlier removal strategy."""
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Remove outliers from the input data."""
        try:
            if isinstance(data, pd.DataFrame):
                return self._remove_outliers_dataframe(data)
            elif isinstance(data, np.ndarray):
                return self._remove_outliers_array(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        except Exception as e:
            self.logger.error(f"Outlier removal failed: {e}")
            raise ValueError(f"Outlier removal failed: {e}")
    
    def _remove_outliers_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from a pandas DataFrame."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            df = df[~self._is_outlier(df[column])]
        return df
    
    def _remove_outliers_array(self, arr: np.ndarray) -> np.ndarray:
        """Remove outliers from a NumPy array."""
        mask = ~self._is_outlier(arr)
        return arr[mask]
    
    def _is_outlier(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Identify outliers using the specified method."""
        if self.method == 'z_score':
            z_scores = np.abs(stats.zscore(data))
            return z_scores > self.threshold
        elif self.method == 'iqr':
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            return (data < lower_bound) | (data > upper_bound)
        elif self.method == 'modified_z_score':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.abs(modified_z_scores) > self.threshold
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")

class FeatureEngineeringStrategy(AdvancedTransformationStrategy):
    """Advanced feature engineering strategy with multiple transformation techniques."""
    
    def __init__(self, strategy: str = 'polynomial', degree: int = 2, **kwargs):
        """Initialize feature engineering strategy."""
        super().__init__(**kwargs)
        self.strategy = strategy
        self.degree = degree
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply feature engineering transformation."""
        try:
            if self.strategy == 'polynomial':
                return self._polynomial_features(data)
            elif self.strategy == 'log':
                return self._log_transform(data)
            elif self.strategy == 'exp':
                return self._exp_transform(data)
            elif self.strategy == 'binning':
                return self._binning_transform(data, bins=self.degree)
            elif self.strategy == 'standard':
                return self._standard_transform(data)
            else:
                raise ValueError(f"Unknown feature engineering strategy: {self.strategy}")
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            raise ValueError(f"Feature engineering failed: {e}")
    
    def _polynomial_features(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Generate polynomial features."""
        try:
            # If input is a DataFrame, handle mixed types
            if isinstance(data, pd.DataFrame):
                # Separate numeric and non-numeric columns
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
                
                # Apply polynomial features only to numeric columns
                if len(numeric_columns) > 0:
                    from sklearn.preprocessing import PolynomialFeatures
                    poly_features = PolynomialFeatures(
                        degree=self.degree, 
                        include_bias=False
                    )
                    
                    # Transform numeric columns
                    numeric_poly = poly_features.fit_transform(data[numeric_columns])
                    numeric_poly_df = pd.DataFrame(
                        numeric_poly, 
                        columns=poly_features.get_feature_names_out(numeric_columns),
                        index=data.index
                    )
                    
                    # Combine with non-numeric columns
                    if len(non_numeric_columns) > 0:
                        result = pd.concat([numeric_poly_df, data[non_numeric_columns]], axis=1)
                    else:
                        result = numeric_poly_df
                    
                    return result
                
                # If no numeric columns, return original data
                return data
            
            # If input is a numpy array, use standard polynomial features
            elif isinstance(data, np.ndarray):
                from sklearn.preprocessing import PolynomialFeatures
                poly_features = PolynomialFeatures(
                    degree=self.degree, 
                    include_bias=False
                )
                return poly_features.fit_transform(data)
            
            else:
                raise TypeError(f"Unsupported data type: {type(data)}")
        
        except Exception as e:
            self.logger.error(f"Polynomial feature generation failed: {e}")
            raise ValueError(f"Polynomial feature generation failed: {e}")
    
    def _log_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply logarithmic transformation."""
        return np.log1p(data)
    
    def _exp_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply exponential transformation."""
        return np.exp(data) - 1
    
    def _binning_transform(self, data: Union[pd.DataFrame, np.ndarray], bins: int = 5) -> Union[pd.DataFrame, np.ndarray]:
        """Discretize continuous features into categorical bins."""
        if isinstance(data, pd.DataFrame):
            return pd.qcut(data, q=bins, labels=False, duplicates='drop')
        else:
            return pd.qcut(pd.Series(data.flatten()), q=bins, labels=False, duplicates='drop').values
    
    def _standard_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply standard scaling transformation."""
        if isinstance(data, pd.DataFrame):
            return (data - data.mean()) / data.std()
        elif isinstance(data, np.ndarray):
            return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        else:
            raise ValueError("Input must be a pandas DataFrame or numpy array")

class TextTransformationStrategy(AdvancedTransformationStrategy):
    """Text transformation strategy for text data preprocessing."""
    
    def __init__(self, strategy: str = 'tokenize', **kwargs):
        """
        Initialize text transformation strategy.

        Args:
            strategy: Type of text transformation ('tokenize', 'lemmatize', 'stem')
            **kwargs: Additional parameters for specific strategies
        """
        self.strategy = strategy
        self.kwargs = kwargs
        
        # Initialize NLTK components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            
            self.tokenizer = nltk.word_tokenize
            self.lemmatizer = WordNetLemmatizer() if strategy == 'lemmatize' else None
            self.stemmer = PorterStemmer() if strategy == 'stem' else None
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logging.warning(f"Failed to initialize NLTK components: {str(e)}")
            self.tokenizer = str.split
            self.lemmatizer = None
            self.stemmer = None
            self.stop_words = set()

    def transform(self, data: Union[str, List[str], pd.Series, pd.DataFrame]) -> Union[List[str], List[List[str]], pd.Series, pd.DataFrame]:
        """
        Transform text data based on the selected strategy.

        Args:
            data: Input text data (string, list of strings, pandas Series, or DataFrame)

        Returns:
            Transformed text data
        """
        if isinstance(data, str):
            return self._transform_single(data)
        elif isinstance(data, list):
            return [self._transform_single(text) for text in data]
        elif isinstance(data, pd.Series):
            return data.apply(self._transform_single)
        elif isinstance(data, pd.DataFrame):
            # Process only string/object columns
            text_columns = data.select_dtypes(include=['object']).columns
            result = data.copy()
            for col in text_columns:
                result[col] = result[col].apply(self._transform_single)
            return result
        else:
            raise ValueError(f"Unsupported data type for text transformation: {type(data)}")

    def _transform_single(self, text: str) -> str:
        """
        Transform a single text string.

        Args:
            text: Input text string

        Returns:
            Transformed text
        """
        if not isinstance(text, str):
            return text

        # Tokenize
        tokens = self.tokenizer(text.lower())
        
        # Remove stopwords if specified
        if self.kwargs.get('remove_stopwords', False):
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Apply transformation strategy
        if self.strategy == 'lemmatize' and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        elif self.strategy == 'stem' and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        # Join tokens if not tokenizing
        if self.strategy != 'tokenize':
            return ' '.join(tokens)
        
        return tokens

class AnomalyDetectionStrategy(AdvancedTransformationStrategy):
    """Strategy for detecting anomalies in data."""

    def __init__(self, strategy: str = 'isolation_forest', contamination: float = 0.1, detection_methods: Optional[List[str]] = None, **kwargs):
        """Initialize anomaly detection strategy.
        
        Args:
            strategy: Base detection method ('isolation_forest', 'local_outlier_factor', 'ensemble')
            contamination: Expected proportion of outliers in the dataset
            detection_methods: List of detection methods for ensemble strategy
            **kwargs: Additional parameters for the detection method
        """
        super().__init__(**kwargs)  # Pass kwargs to parent class
        self.strategy = strategy
        self.contamination = contamination
        self.detection_methods = detection_methods or ['isolation_forest', 'local_outlier_factor']
        self.kwargs = kwargs
        
        # Initialize detectors
        self.detectors = {}
        if strategy == 'ensemble':
            for method in self.detection_methods:
                self.detectors[method] = self._create_detector(method)
        else:
            self.detectors[strategy] = self._create_detector(strategy)

    def _create_detector(self, method: str) -> Any:
        """Create an anomaly detector based on the specified method."""
        detector_kwargs = {k: v for k, v in self.kwargs.items() if k != 'detection_methods'}
        
        if method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            return IsolationForest(contamination=self.contamination, **detector_kwargs)
        elif method == 'local_outlier_factor':
            from sklearn.neighbors import LocalOutlierFactor
            return LocalOutlierFactor(contamination=self.contamination, **detector_kwargs)
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply anomaly detection to the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with anomaly scores
        """
        try:
            if self.strategy == 'ensemble':
                # Run all detectors and combine results
                scores = pd.DataFrame()
                for method, detector in self.detectors.items():
                    scores[f'anomaly_score_{method}'] = detector.fit_predict(data)
                
                # Combine scores (majority voting)
                data['anomaly_score'] = scores.mean(axis=1)
            else:
                # Single detector
                detector = self.detectors[self.strategy]
                data['anomaly_score'] = detector.fit_predict(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {str(e)}")
            return data

class TimeSeriesTransformationStrategy(AdvancedTransformationStrategy):
    """Time series transformation strategy."""

    def __init__(self, strategy: str = 'rolling_features', window: int = 14, **kwargs):
        """Initialize time series transformation strategy."""
        super().__init__()
        self.strategy = strategy
        self.window = window
        self.kwargs = kwargs

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply time series transformation."""
        try:
            if self.strategy == 'rolling_features':
                return self._rolling_features(data)
            elif self.strategy == 'lag_features':
                return self._lag_features(data)
            elif self.strategy == 'diff_features':
                return self._diff_features(data)
            else:
                raise ValueError(f"Unknown time series strategy: {self.strategy}")
        except Exception as e:
            self.logger.error(f"Time series transformation failed: {e}")
            raise ValueError(f"Time series transformation failed: {e}")

    def _rolling_features(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Create rolling window features."""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        result = data.copy()
        for col in data.columns:
            result[f'{col}_rolling_mean'] = data[col].rolling(window=self.window).mean()
            result[f'{col}_rolling_std'] = data[col].rolling(window=self.window).std()
            result[f'{col}_rolling_min'] = data[col].rolling(window=self.window).min()
            result[f'{col}_rolling_max'] = data[col].rolling(window=self.window).max()
        
        return result.fillna(method='bfill')

    def _lag_features(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Create lagged features."""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        result = data.copy()
        for col in data.columns:
            for lag in range(1, self.window + 1):
                result[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return result.fillna(method='bfill')

    def _diff_features(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Create difference features."""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        result = data.copy()
        for col in data.columns:
            result[f'{col}_diff_1'] = data[col].diff()
            result[f'{col}_diff_{self.window}'] = data[col].diff(self.window)
            result[f'{col}_pct_change'] = data[col].pct_change(self.window)
        
        return result.fillna(method='bfill')

# Example usage and demonstration
def demonstrate_transformations():
    """Demonstrate advanced transformation strategies."""
    # Create sample data
    data = pd.DataFrame({
        'numeric': np.random.randn(100),
        'text': ['Sample text ' * 3] * 100
    })
    
    # Demonstrate outlier removal
    outlier_strategy = OutlierRemovalStrategy(method='z_score', threshold=3.0)
    cleaned_data = outlier_strategy.transform(data['numeric'])
    
    # Demonstrate feature engineering
    feature_strategy = FeatureEngineeringStrategy(strategy='polynomial', degree=2)
    engineered_features = feature_strategy.transform(data[['numeric']])
    
    # Demonstrate text transformation
    text_strategy = TextTransformationStrategy(method='tokenize', remove_stopwords=True)
    processed_text = text_strategy.transform(data['text'])
    
    # Demonstrate anomaly detection
    anomaly_strategy = AnomalyDetectionStrategy(strategy='isolation_forest', contamination=0.1)
    detected_anomalies = anomaly_strategy.transform(data[['numeric']])
    
    return {
        'cleaned_data': cleaned_data,
        'engineered_features': engineered_features,
        'processed_text': processed_text,
        'detected_anomalies': detected_anomalies
    }

if __name__ == "__main__":
    demonstrate_transformations()
