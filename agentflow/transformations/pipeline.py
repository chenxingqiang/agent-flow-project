"""Transformation pipeline module for data processing."""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from .advanced_strategies import (
    TextTransformationStrategy,
    FeatureEngineeringStrategy,
    OutlierRemovalStrategy,
    AnomalyDetectionStrategy
)

class TransformationPipeline:
    """Pipeline for applying multiple transformation strategies."""

    def __init__(self, strategies: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize transformation pipeline.

        Args:
            strategies: List of strategy configurations
        """
        self.strategies = []
        if strategies:
            self._configure_strategies(strategies)

    def _configure_strategies(self, strategies: List[Dict[str, Any]]) -> None:
        """
        Configure transformation strategies.

        Args:
            strategies: List of strategy configurations
        """
        for config in strategies:
            strategy_type = config.pop('type', '').lower()
            
            try:
                if strategy_type == 'text':
                    self.strategies.append(TextTransformationStrategy(**config))
                elif strategy_type == 'feature':
                    self.strategies.append(FeatureEngineeringStrategy(**config))
                elif strategy_type == 'outlier':
                    self.strategies.append(OutlierRemovalStrategy(**config))
                elif strategy_type == 'anomaly':
                    self.strategies.append(AnomalyDetectionStrategy(**config))
                else:
                    logging.warning(f"Unknown strategy type: {strategy_type}")
            except Exception as e:
                logging.error(f"Failed to configure strategy {strategy_type}: {str(e)}")

    def add_strategy(self, strategy_type: str, **kwargs) -> None:
        """
        Add a transformation strategy to the pipeline.

        Args:
            strategy_type: Type of strategy to add
            **kwargs: Strategy configuration parameters
        """
        try:
            if strategy_type == 'text':
                self.strategies.append(TextTransformationStrategy(**kwargs))
            elif strategy_type == 'feature':
                self.strategies.append(FeatureEngineeringStrategy(**kwargs))
            elif strategy_type == 'outlier':
                self.strategies.append(OutlierRemovalStrategy(**kwargs))
            elif strategy_type == 'anomaly':
                self.strategies.append(AnomalyDetectionStrategy(**kwargs))
            else:
                logging.warning(f"Unknown strategy type: {strategy_type}")
        except Exception as e:
            logging.error(f"Failed to add strategy {strategy_type}: {str(e)}")

    def transform(self, data: Union[pd.DataFrame, np.ndarray, str, List[str]]) -> Any:
        """
        Apply all transformation strategies in sequence.

        Args:
            data: Input data to transform

        Returns:
            Transformed data
        """
        transformed_data = data
        for strategy in self.strategies:
            try:
                transformed_data = strategy.transform(transformed_data)
            except Exception as e:
                logging.error(f"Failed to apply strategy {type(strategy).__name__}: {str(e)}")
                continue
        return transformed_data

    def reset(self) -> None:
        """Reset the pipeline by clearing all strategies."""
        self.strategies = []
