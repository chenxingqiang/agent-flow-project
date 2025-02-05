"""Advanced transformation strategies for data processing."""
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.covariance import EllipticEnvelope
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

class AdvancedTransformationStrategy:
    """Base class for advanced transformation strategies."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, **kwargs):
        """Initialize transformation strategy.
        
        Args:
            logger: Optional logger instance
            **kwargs: Additional strategy-specific parameters
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.params = kwargs
    
    def transform(self, data: Any) -> Any:
        """Transform input data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        raise NotImplementedError("Subclasses must implement transform method")

class OutlierRemovalStrategy(AdvancedTransformationStrategy):
    """Strategy for removing outliers from data."""
    
    def __init__(self, method: str = 'z_score', threshold: float = 3.0, **kwargs):
        """Initialize outlier removal strategy.
        
        Args:
            method: Method to use for outlier detection ('z_score', 'iqr', 'modified_z_score', 'isolation_forest')
            threshold: Threshold for outlier detection
        """
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold
        self.type = 'outlier_removal'
        self.params = {
            'method': method,
            'threshold': threshold,
            **kwargs
        }
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Remove outliers from data.
        
        Args:
            data: Input data (DataFrame or ndarray)
            
        Returns:
            Data with outliers removed
        """
        self.logger.info(f"Applying strategy: {self.method} outlier removal with threshold {self.threshold}")
        
        # Convert numpy array to DataFrame if needed
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = pd.DataFrame(data)
        
        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return data.values if is_numpy else data
        
        if self.method == 'z_score':
            z_scores = np.abs(StandardScaler().fit_transform(numeric_data))
            mask = (z_scores < self.threshold).all(axis=1)
        elif self.method == 'iqr':
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((numeric_data < (Q1 - self.threshold * IQR)) | (numeric_data > (Q3 + self.threshold * IQR))).any(axis=1)
        elif self.method == 'modified_z_score':
            median = numeric_data.median()
            mad = (numeric_data - median).abs().median()
            modified_z_scores = 0.6745 * (numeric_data - median) / mad
            mask = (np.abs(modified_z_scores) < self.threshold).all(axis=1)
        elif self.method == 'isolation_forest':
            contamination = self.params.get('contamination', 0.1)
            random_state = self.params.get('random_state', None)
            detector = IsolationForest(contamination=contamination, random_state=random_state)
            predictions = detector.fit_predict(numeric_data)
            mask = predictions == 1  # 1 for inliers, -1 for outliers
        else:
            raise ValueError(f"Unknown outlier removal method: {self.method}")
            
        result = data[mask]
        return result.values if is_numpy else result

class FeatureEngineeringStrategy(AdvancedTransformationStrategy):
    """Strategy for feature engineering."""
    
    def __init__(self, strategy: str = 'polynomial', **kwargs):
        """Initialize feature engineering strategy."""
        super().__init__(**kwargs)
        self.strategy = strategy
        self.degree = kwargs.get('degree', 2)
        self.with_mean = kwargs.get('with_mean', True)
        self.with_std = kwargs.get('with_std', True)
        self.type = 'feature_engineering'
        self.params = {
            'strategy': strategy,
            'degree': self.degree,
            'with_mean': self.with_mean,
            'with_std': self.with_std,
            **kwargs
        }
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply feature engineering transformations.
        
        Args:
            data: Input data (DataFrame, ndarray, or dictionary)
            
        Returns:
            Transformed data with engineered features
        """
        self.logger.info(f"Applying {self.strategy} feature engineering")
        
        # Convert dictionary to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Convert numpy array to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            # If no numeric columns, return original data
            return data.values if isinstance(data, pd.DataFrame) else data
        
        if self.strategy == 'standard':
            scaler = StandardScaler()
            transformed_data = scaler.fit_transform(numeric_data)
            transformed_df = pd.DataFrame(transformed_data, columns=numeric_data.columns, index=data.index)
            
            # Replace numeric columns with transformed data
            result = data.copy()
            for col in numeric_data.columns:
                result[col] = transformed_df[col]
            return result.values if isinstance(data, np.ndarray) else result
            
        elif self.strategy == 'polynomial':
            poly = PolynomialFeatures(degree=self.params.get('degree', 2))
            transformed_data = poly.fit_transform(numeric_data)
            feature_names = [f'poly_{i}' for i in range(transformed_data.shape[1])]
            transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=data.index)
            
            # Combine original non-numeric columns with transformed data
            non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
            if not non_numeric_cols.empty:
                result = pd.concat([data[non_numeric_cols], transformed_df], axis=1)
            else:
                result = transformed_df
            return result.values if isinstance(data, np.ndarray) else result
            
        else:
            raise ValueError(f"Unknown feature engineering strategy: {self.strategy}")

class TextTransformationStrategy(AdvancedTransformationStrategy):
    """Strategy for text data transformations."""
    
    def __init__(self, method: str = 'normalize', **kwargs):
        """Initialize text transformation strategy."""
        super().__init__(**kwargs)
        self.method = method
        self.remove_stopwords = kwargs.get('remove_stopwords', True)
        self.lemmatize = kwargs.get('lemmatize', False)
        self.vectorize = kwargs.get('vectorize', False)
        self.max_features = kwargs.get('max_features', 100)
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Transform text data.
        
        Args:
            data: Input data (DataFrame or ndarray)
            
        Returns:
            Transformed data with processed text
        """
        self.logger.info(f"Applying {self.method} text transformation")
        
        # Convert numpy array to DataFrame if needed
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = pd.DataFrame(data)
        
        # Get text columns only
        text_cols = data.select_dtypes(include=['object']).columns
        if len(text_cols) == 0:
            return data if not is_numpy else data.values
        
        result = data.copy()
        
        for col in text_cols:
            if self.method == 'normalize':
                # Basic text cleaning
                result[col] = result[col].str.lower()
                result[col] = result[col].str.replace(r'[^\w\s]', '', regex=True)
                
                if self.remove_stopwords:
                    stop_words = set(stopwords.words('english'))
                    result[col] = result[col].apply(lambda x: ' '.join([word for word in str(x).split() if word not in stop_words]))
                
                if self.lemmatize:
                    lemmatizer = WordNetLemmatizer()
                    result[col] = result[col].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in str(x).split()]))
                
                # Add word count feature
                result[f'{col}_word_count'] = result[col].str.split().str.len()
                
                if self.vectorize:
                    # Add TF-IDF features
                    vectorizer = TfidfVectorizer(max_features=self.max_features)
                    tfidf_matrix = vectorizer.fit_transform(result[col].fillna(''))
                    feature_names = [f'{col}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=data.index)
                    result = pd.concat([result, tfidf_df], axis=1)
            
            else:
                raise ValueError(f"Unknown text transformation method: {self.method}")
        
        return result if not is_numpy else result.values

class AnomalyDetectionStrategy(AdvancedTransformationStrategy):
    """Strategy for detecting anomalies in data."""
    
    def __init__(self, strategy: str = 'isolation_forest', **kwargs):
        """Initialize anomaly detection strategy.
        
        Args:
            strategy: Strategy to use ('isolation_forest', 'local_outlier_factor', 'one_class_svm')
            **kwargs: Additional strategy-specific parameters
        """
        super().__init__(**kwargs)
        self.strategy = strategy
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with anomaly detection results
        """
        self.logger.info(f"Applying {self.strategy} anomaly detection")
        
        result = data.copy()
        
        if self.strategy == 'isolation_forest':
            contamination = self.params.get('contamination', 0.1)
            random_state = self.params.get('random_state', None)
            detector = IsolationForest(contamination=contamination, random_state=random_state)
            predictions = detector.fit_predict(data)
            result['predictions'] = predictions
            return result
        
        elif self.strategy == 'local_outlier_factor':
            n_neighbors = self.params.get('n_neighbors', 20)
            contamination = self.params.get('contamination', 0.1)
            detector = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            predictions = detector.fit_predict(data)
            result['predictions'] = predictions
            return result
        
        elif self.strategy == 'one_class_svm':
            kernel = self.params.get('kernel', 'rbf')
            nu = self.params.get('nu', 0.1)
            detector = OneClassSVM(kernel=kernel, nu=nu)
            predictions = detector.fit_predict(data)
            result['predictions'] = predictions
            return result
        
        else:
            raise ValueError(f"Unknown anomaly detection strategy: {self.strategy}")

class TimeSeriesTransformationStrategy(AdvancedTransformationStrategy):
    """Strategy for transforming time series data."""
    
    def __init__(self, strategy: str = 'moving_average', **kwargs):
        """Initialize time series transformation strategy.
        
        Args:
            strategy: Strategy to use ('moving_average', 'exponential_smoothing', 'differencing')
            **kwargs: Additional strategy-specific parameters
        """
        super().__init__(**kwargs)
        self.strategy = strategy
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform time series data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with transformed time series
        """
        self.logger.info(f"Applying {self.strategy} time series transformation")
        
        if self.strategy == 'moving_average':
            window = self.params.get('window', 3)
            return data.rolling(window=window).mean()
        
        elif self.strategy == 'exponential_smoothing':
            alpha = self.params.get('alpha', 0.3)
            return data.ewm(alpha=alpha).mean()
        
        elif self.strategy == 'differencing':
            periods = self.params.get('periods', 1)
            return data.diff(periods=periods)
        
        else:
            raise ValueError(f"Unknown time series transformation strategy: {self.strategy}") 