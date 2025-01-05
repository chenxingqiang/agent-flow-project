from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime

class BaseTransformationStrategy(ABC):
    """Base class for transformation strategies."""
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transform the input data."""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate the input data."""
        pass

class TimeSeriesStrategy(BaseTransformationStrategy):
    """Strategy for time series data transformations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.window_size = self.config.get('window_size', 5)
        self.method = self.config.get('method', 'moving_average')
    
    def transform(self, data: List[float]) -> List[float]:
        """Transform time series data."""
        if not self.validate(data):
            raise ValueError("Invalid time series data")
        
        if self.method == 'moving_average':
            return self._moving_average(data)
        elif self.method == 'exponential_smoothing':
            return self._exponential_smoothing(data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def validate(self, data: Any) -> bool:
        """Validate time series data."""
        return isinstance(data, list) and all(isinstance(x, (int, float)) for x in data)
    
    def _moving_average(self, data: List[float]) -> List[float]:
        """Calculate moving average."""
        result = []
        for i in range(len(data)):
            start = max(0, i - self.window_size + 1)
            window = data[start:i + 1]
            result.append(sum(window) / len(window))
        return result
    
    def _exponential_smoothing(self, data: List[float], alpha: float = 0.3) -> List[float]:
        """Apply exponential smoothing."""
        result = [data[0]]
        for n in range(1, len(data)):
            result.append(alpha * data[n] + (1 - alpha) * result[n-1])
        return result

class AnomalyDetectionStrategy(BaseTransformationStrategy):
    """Strategy for anomaly detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', 2.0)
        self.method = self.config.get('method', 'zscore')
    
    def transform(self, data: List[float]) -> List[bool]:
        """Detect anomalies in the data."""
        if not self.validate(data):
            raise ValueError("Invalid data for anomaly detection")
        
        if self.method == 'zscore':
            return self._zscore_detection(data)
        elif self.method == 'iqr':
            return self._iqr_detection(data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def validate(self, data: Any) -> bool:
        """Validate input data."""
        return isinstance(data, list) and all(isinstance(x, (int, float)) for x in data)
    
    def _zscore_detection(self, data: List[float]) -> List[bool]:
        """Detect anomalies using Z-score method."""
        mean = np.mean(data)
        std = np.std(data)
        z_scores = [(x - mean) / std for x in data]
        return [abs(z) > self.threshold for z in z_scores]
    
    def _iqr_detection(self, data: List[float]) -> List[bool]:
        """Detect anomalies using IQR method."""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr
        return [x < lower_bound or x > upper_bound for x in data]

class FeatureEngineeringStrategy(BaseTransformationStrategy):
    """Strategy for feature engineering."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.features = self.config.get('features', ['mean', 'std'])
    
    def transform(self, data: List[float]) -> Dict[str, float]:
        """Engineer features from the data."""
        if not self.validate(data):
            raise ValueError("Invalid data for feature engineering")
        
        features = {}
        for feature in self.features:
            if feature == 'mean':
                features['mean'] = np.mean(data)
            elif feature == 'std':
                features['std'] = np.std(data)
            elif feature == 'min':
                features['min'] = min(data)
            elif feature == 'max':
                features['max'] = max(data)
            elif feature == 'range':
                features['range'] = max(data) - min(data)
        return features
    
    def validate(self, data: Any) -> bool:
        """Validate input data."""
        return isinstance(data, list) and all(isinstance(x, (int, float)) for x in data)

class OutlierRemovalStrategy(BaseTransformationStrategy):
    """Strategy for outlier removal."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', 2.0)
        self.method = self.config.get('method', 'zscore')
    
    def transform(self, data: List[float]) -> List[float]:
        """Remove outliers from the data."""
        if not self.validate(data):
            raise ValueError("Invalid data for outlier removal")
        
        if self.method == 'zscore':
            return self._remove_zscore_outliers(data)
        elif self.method == 'iqr':
            return self._remove_iqr_outliers(data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def validate(self, data: Any) -> bool:
        """Validate input data."""
        return isinstance(data, list) and all(isinstance(x, (int, float)) for x in data)
    
    def _remove_zscore_outliers(self, data: List[float]) -> List[float]:
        """Remove outliers using Z-score method."""
        mean = np.mean(data)
        std = np.std(data)
        return [x for x in data if abs((x - mean) / std) <= self.threshold]
    
    def _remove_iqr_outliers(self, data: List[float]) -> List[float]:
        """Remove outliers using IQR method."""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr
        return [x for x in data if lower_bound <= x <= upper_bound]

class TextTransformationStrategy(BaseTransformationStrategy):
    """Strategy for text transformations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.method = self.config.get('method', 'clean')
    
    def transform(self, text: str) -> str:
        """Transform text data."""
        if not self.validate(text):
            raise ValueError("Invalid text data")
        
        if self.method == 'clean':
            return self._clean_text(text)
        elif self.method == 'normalize':
            return self._normalize_text(text)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def validate(self, data: Any) -> bool:
        """Validate input data."""
        return isinstance(data, str)
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing special characters and extra whitespace."""
        import re
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by converting to lowercase and removing accents."""
        import unicodedata
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return text.lower()

class ResearchTransformationStrategy(BaseTransformationStrategy):
    """Strategy for research data transformations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.method = self.config.get('method', 'summarize')
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform research data."""
        if not self.validate(data):
            raise ValueError("Invalid research data")
        
        if self.method == 'summarize':
            return self._summarize_research(data)
        elif self.method == 'analyze':
            return self._analyze_research(data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def validate(self, data: Any) -> bool:
        """Validate input data."""
        return isinstance(data, dict)
    
    def _summarize_research(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize research data."""
        summary = {
            'title': data.get('title', ''),
            'abstract': data.get('abstract', ''),
            'key_findings': data.get('findings', []),
            'timestamp': datetime.now().isoformat()
        }
        return summary
    
    def _analyze_research(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze research data."""
        analysis = {
            'title': data.get('title', ''),
            'methodology': data.get('methodology', ''),
            'results': data.get('results', []),
            'conclusions': data.get('conclusions', []),
            'timestamp': datetime.now().isoformat()
        }
        return analysis

class DataScienceTransformationStrategy(BaseTransformationStrategy):
    """Strategy for data science transformations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.method = self.config.get('method', 'preprocess')
    
    def transform(self, data: Any) -> Any:
        """Transform data science data."""
        if not self.validate(data):
            raise ValueError("Invalid data science data")
        
        if self.method == 'preprocess':
            return self._preprocess_data(data)
        elif self.method == 'feature_selection':
            return self._select_features(data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def validate(self, data: Any) -> bool:
        """Validate input data."""
        return True  # Implement specific validation logic
    
    def _preprocess_data(self, data: Any) -> Any:
        """Preprocess data science data."""
        # Implement preprocessing logic
        return data
    
    def _select_features(self, data: Any) -> Any:
        """Select features from data science data."""
        # Implement feature selection logic
        return data

class DefaultTransformationStrategy(BaseTransformationStrategy):
    """Default transformation strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def transform(self, data: Any) -> Any:
        """Pass through data without transformation."""
        if not self.validate(data):
            raise ValueError("Invalid data")
        return data
    
    def validate(self, data: Any) -> bool:
        """Always validate as true."""
        return True

class AdvancedTransformationStrategy(BaseTransformationStrategy):
    """Advanced transformation strategy combining multiple strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.strategies = []
        self._setup_strategies()
    
    def transform(self, data: Any) -> Any:
        """Apply multiple transformations in sequence."""
        if not self.validate(data):
            raise ValueError("Invalid data")
        
        result = data
        for strategy in self.strategies:
            result = strategy.transform(result)
        return result
    
    def validate(self, data: Any) -> bool:
        """Validate using all strategies."""
        return all(strategy.validate(data) for strategy in self.strategies)
    
    def _setup_strategies(self) -> None:
        """Setup transformation strategies based on configuration."""
        strategy_configs = self.config.get('strategies', [])
        for config in strategy_configs:
            strategy_type = config.get('type', 'default')
            strategy_config = config.get('config', {})
            
            if strategy_type == 'time_series':
                self.strategies.append(TimeSeriesStrategy(strategy_config))
            elif strategy_type == 'anomaly_detection':
                self.strategies.append(AnomalyDetectionStrategy(strategy_config))
            elif strategy_type == 'feature_engineering':
                self.strategies.append(FeatureEngineeringStrategy(strategy_config))
            elif strategy_type == 'outlier_removal':
                self.strategies.append(OutlierRemovalStrategy(strategy_config))
            elif strategy_type == 'text':
                self.strategies.append(TextTransformationStrategy(strategy_config))
            elif strategy_type == 'research':
                self.strategies.append(ResearchTransformationStrategy(strategy_config))
            elif strategy_type == 'data_science':
                self.strategies.append(DataScienceTransformationStrategy(strategy_config))
            else:
                self.strategies.append(DefaultTransformationStrategy(strategy_config)) 