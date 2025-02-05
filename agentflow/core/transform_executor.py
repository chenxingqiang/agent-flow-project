"""Transform executor module."""
from typing import Dict, Any, Optional, Union
import asyncio
from .base_executor import BaseExecutor
from .workflow_types import WorkflowStep
from ..transformations.advanced_strategies import FeatureEngineeringStrategy, OutlierRemovalStrategy
import numpy as np
import pandas as pd

class TransformExecutor(BaseExecutor):
    """Transform executor."""

    async def execute(self, context: Dict[str, Any]) -> Union[pd.DataFrame, np.ndarray]:
        """Execute transform step."""
        try:
            # Get input data
            data = context.get("data", {})

            # Get step config
            step = self.config.steps[0]
            step_config = step.config.model_dump() if hasattr(step.config, "model_dump") else step.config
            if not isinstance(step_config, dict):
                step_config = {}

            # Get execution function if available
            execute_fn = step_config.get("execute")
            if execute_fn and callable(execute_fn):
                return await execute_fn(data)

            # Get strategy and parameters from config
            strategy = step_config.get("strategy")
            if not strategy:
                # Default to feature engineering if no strategy specified
                strategy = "feature_engineering"

            params = step_config.get("params", {})

            # Execute transform based on strategy
            if strategy == "feature_engineering":
                return await self._execute_feature_engineering(data, params)
            elif strategy == "outlier_removal":
                return await self._execute_outlier_removal(data, params)
            elif strategy == "anomaly_detection":
                return await self._execute_anomaly_detection(data, params)
            else:
                raise ValueError(f"Unknown transformation strategy: {strategy}")

        except Exception as e:
            raise ValueError(f"Transform execution failed: {str(e)}")

    async def _execute_feature_engineering(self, data: Dict[str, Any], params: Dict[str, Any]) -> Union[pd.DataFrame, np.ndarray]:
        """Execute feature engineering transformation."""
        strategy = FeatureEngineeringStrategy(strategy=params.get('method', 'standard'), **params)
        input_data = data.get('data')
        if not isinstance(input_data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        return strategy.transform(input_data)

    async def _execute_outlier_removal(self, data: Dict[str, Any], params: Dict[str, Any]) -> Union[pd.DataFrame, np.ndarray]:
        """Execute outlier removal transformation."""
        strategy = OutlierRemovalStrategy(method=params.get('method', 'z_score'), threshold=params.get('threshold', 3.0))
        input_data = data.get('data')
        if not isinstance(input_data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        return strategy.transform(input_data)

    async def _execute_anomaly_detection(self, data: Dict[str, Any], params: Dict[str, Any]) -> Union[pd.DataFrame, np.ndarray]:
        """Execute anomaly detection transformation."""
        # TODO: Implement anomaly detection
        input_data = data.get('data')
        if not isinstance(input_data, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input data must be a numpy array or pandas DataFrame")
        return input_data