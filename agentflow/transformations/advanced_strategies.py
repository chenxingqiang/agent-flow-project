import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

class AdvancedTransformationStrategy(abc.ABC):
    """
    Advanced base class for sophisticated data transformation strategies.
    
    This abstract base class provides a framework for implementing complex
    data transformation techniques with robust error handling and logging.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the transformation strategy.
        
        Args:
            logger: Optional custom logger. If not provided, a default logger is created.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
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
    """
    Advanced outlier removal strategy using multiple statistical techniques.
    
    Supports different outlier detection methods:
    - Z-score method
    - Interquartile Range (IQR) method
    - Modified Z-score method
    """
    
    def __init__(
        self, 
        method: str = 'z_score', 
        threshold: float = 3.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize outlier removal strategy.
        
        Args:
            method: Outlier detection method ('z_score', 'iqr', 'modified_z_score')
            threshold: Number of standard deviations or IQR multiples to consider as outlier
            logger: Optional custom logger
        """
        super().__init__(logger)
        self.method = method
        self.threshold = threshold
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Remove outliers from the input data using the specified method.
        
        Args:
            data: Input data (DataFrame or NumPy array)
        
        Returns:
            Data with outliers removed
        """
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
        return arr[~self._is_outlier(arr)]
    
    def _is_outlier(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Detect outliers using the specified method.
        
        Returns:
            Boolean mask of outliers
        """
        if self.method == 'z_score':
            return np.abs(stats.zscore(data)) > self.threshold
        elif self.method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        elif self.method == 'modified_z_score':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.abs(modified_z_scores) > self.threshold
        else:
            raise ValueError(f"Unsupported outlier detection method: {self.method}")

class FeatureEngineeringStrategy(AdvancedTransformationStrategy):
    """
    Advanced feature engineering strategy with multiple transformation techniques.
    
    Supports:
    - Polynomial feature generation
    - Logarithmic transformation
    - Exponential transformation
    - Binning/discretization
    """
    
    def __init__(
        self, 
        strategy: str = 'polynomial', 
        degree: int = 2,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize feature engineering strategy.
        
        Args:
            strategy: Transformation strategy
            degree: Degree of polynomial features (for polynomial strategy)
            logger: Optional custom logger
        """
        super().__init__(logger)
        self.strategy = strategy
        self.degree = degree
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply feature engineering transformation.
        
        Args:
            data: Input data
        
        Returns:
            Transformed data with engineered features
        """
        try:
            if self.strategy == 'polynomial':
                return self._polynomial_features(data)
            elif self.strategy == 'log':
                return self._log_transform(data)
            elif self.strategy == 'exp':
                return self._exp_transform(data)
            elif self.strategy == 'binning':
                return self._binning_transform(data)
            else:
                raise ValueError(f"Unsupported feature engineering strategy: {self.strategy}")
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            raise ValueError(f"Feature engineering failed: {e}")
    
    def _polynomial_features(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Generate polynomial features."""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        
        if isinstance(data, pd.DataFrame):
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            poly_features = poly.fit_transform(data[numeric_columns])
            
            # Create new column names
            feature_names = poly.get_feature_names_out(numeric_columns)
            return pd.DataFrame(poly_features, columns=feature_names, index=data.index)
        else:
            return poly.fit_transform(data)
    
    def _log_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply logarithmic transformation."""
        if isinstance(data, pd.DataFrame):
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = np.log1p(data[numeric_columns])
            return data
        else:
            return np.log1p(data)
    
    def _exp_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply exponential transformation."""
        if isinstance(data, pd.DataFrame):
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = np.exp(data[numeric_columns])
            return data
        else:
            return np.exp(data)
    
    def _binning_transform(self, data: Union[pd.DataFrame, np.ndarray], bins: int = 5) -> Union[pd.DataFrame, np.ndarray]:
        """
        Discretize continuous features into categorical bins.
        
        Args:
            data: Input data
            bins: Number of bins to create
        
        Returns:
            Binned data
        """
        if isinstance(data, pd.DataFrame):
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                data[f'{column}_binned'] = pd.qcut(data[column], q=bins, labels=False, duplicates='drop')
            return data
        else:
            return np.digitize(data, bins=np.linspace(data.min(), data.max(), bins + 1)[1:-1])

class TextTransformationStrategy(AdvancedTransformationStrategy):
    """
    Advanced text transformation strategy with NLP-based techniques.
    
    Supports:
    - Tokenization
    - Stop word removal
    - Lemmatization
    - TF-IDF vectorization
    """
    
    def __init__(
        self, 
        strategy: str = 'tokenize', 
        language: str = 'english',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize text transformation strategy.
        
        Args:
            strategy: Text transformation strategy
            language: Language for NLP processing
            logger: Optional custom logger
        """
        super().__init__(logger)
        self.strategy = strategy
        self.language = language
    
    def transform(self, text: Union[str, List[str], pd.Series]) -> Any:
        """
        Apply text transformation.
        
        Args:
            text: Input text data
        
        Returns:
            Transformed text
        """
        try:
            if self.strategy == 'tokenize':
                return self._tokenize(text)
            elif self.strategy == 'remove_stopwords':
                return self._remove_stopwords(text)
            elif self.strategy == 'lemmatize':
                return self._lemmatize(text)
            elif self.strategy == 'tfidf':
                return self._tfidf_vectorize(text)
            else:
                raise ValueError(f"Unsupported text transformation strategy: {self.strategy}")
        except Exception as e:
            self.logger.error(f"Text transformation failed: {e}")
            raise ValueError(f"Text transformation failed: {e}")
    
    def _tokenize(self, text: Union[str, List[str], pd.Series]) -> List[List[str]]:
        """Tokenize text using NLTK."""
        import nltk
        nltk.download('punkt', quiet=True)
        
        if isinstance(text, str):
            return [nltk.word_tokenize(text)]
        elif isinstance(text, list):
            return [nltk.word_tokenize(t) for t in text]
        elif isinstance(text, pd.Series):
            return text.apply(nltk.word_tokenize).tolist()
    
    def _remove_stopwords(self, text: Union[str, List[str], pd.Series]) -> List[List[str]]:
        """Remove stop words using NLTK."""
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        
        stop_words = set(stopwords.words(self.language))
        
        def _remove_stops(tokens):
            return [word for word in tokens if word.lower() not in stop_words]
        
        tokenized = self._tokenize(text)
        return [_remove_stops(tokens) for tokens in tokenized]
    
    def _lemmatize(self, text: Union[str, List[str], pd.Series]) -> List[List[str]]:
        """Lemmatize text using NLTK."""
        import nltk
        nltk.download('wordnet', quiet=True)
        from nltk.stem import WordNetLemmatizer
        
        lemmatizer = WordNetLemmatizer()
        
        def _lemmatize_tokens(tokens):
            return [lemmatizer.lemmatize(word) for word in tokens]
        
        tokenized = self._remove_stopwords(text)
        return [_lemmatize_tokens(tokens) for tokens in tokenized]
    
    def _tfidf_vectorize(self, text: Union[str, List[str], pd.Series]) -> Any:
        """Convert text to TF-IDF vectors."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer()
        
        if isinstance(text, str):
            return vectorizer.fit_transform([text])
        elif isinstance(text, list):
            return vectorizer.fit_transform(text)
        elif isinstance(text, pd.Series):
            return vectorizer.fit_transform(text.tolist())

# Example usage and demonstration
def demonstrate_transformations():
    """Demonstrate advanced transformation strategies."""
    # Outlier Removal
    data = pd.DataFrame({
        'A': [1, 2, 3, 100, 4, 5, 6],
        'B': [10, 20, 30, 400, 50, 60, 70]
    })
    
    outlier_remover = OutlierRemovalStrategy(method='z_score', threshold=2.0)
    cleaned_data = outlier_remover.transform(data)
    print("Outlier Removal Result:\n", cleaned_data)
    
    # Feature Engineering
    feature_engineer = FeatureEngineeringStrategy(strategy='polynomial', degree=2)
    engineered_features = feature_engineer.transform(data)
    print("\nFeature Engineering Result:\n", engineered_features)
    
    # Text Transformation
    text_data = ["Natural language processing is fascinating", 
                 "Machine learning transforms data"]
    text_transformer = TextTransformationStrategy(strategy='lemmatize')
    transformed_text = text_transformer.transform(text_data)
    print("\nText Transformation Result:\n", transformed_text)

if __name__ == "__main__":
    demonstrate_transformations()
