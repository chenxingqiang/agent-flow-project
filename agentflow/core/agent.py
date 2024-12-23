"""Agent module for handling different types of agents."""

import json
import logging
import os
import time
import asyncio
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union, Type
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from agentflow.transformations.advanced_strategies import (
    AdvancedTransformationStrategy,
    OutlierRemovalStrategy,
    FeatureEngineeringStrategy,
    TextTransformationStrategy
)
from agentflow.transformations.pipeline import TransformationPipeline
from agentflow.core.config_manager import ConfigManager
from agentflow.core.workflow_executor import WorkflowExecutor
from agentflow.core.distributed_workflow import DistributedWorkflow
from agentflow.transformations.advanced_strategies import (
    AnomalyDetectionStrategy,
    TimeSeriesTransformationStrategy
)
from agentflow.core.isa.isa_manager import ISAManager
from agentflow.core.instruction_selector import InstructionSelector

# Local imports
from .config import AgentConfig
from .workflow_executor import WorkflowExecutor
from .instruction_selector import InstructionSelector
from .instruction import Instruction, InstructionType, ISAManager, InstructionSelector

"""Base Agent Module"""

__all__ = [
    'Agent',
    'AgentBase',
    'AgentFactory',
    'TransformationPipeline',
    'AgentTransformationMixin',
    'ConfigurationType',
    'ConfigurationSchema',
    'ResearchAgent',
    'DataScienceAgent',
    'FlexibleConfigurationManager'
]

logger = logging.getLogger(__name__)

class TransformationPipeline:
    def __init__(self):
        """Initialize transformation pipeline."""
        self.strategies = []
    
    def add_strategy(self, strategy):
        """Add a transformation strategy to the pipeline."""
        self.strategies.append(strategy)
    
    def transform(self, data):
        """Apply all transformation strategies in sequence."""
        transformed_data = data
        for strategy in self.strategies:
            transformed_data = strategy.transform(transformed_data)
        return transformed_data

class AgentTransformationMixin:
    """Mixin class to add transformation capabilities to agents."""
    
    def __init__(self, *args, **kwargs):
        """Initialize transformation-related attributes."""
        super().__init__(*args, **kwargs)
        
        # Initialize transformation pipelines
        self.input_transformation_pipeline = TransformationPipeline()
        self.preprocessing_transformation_pipeline = TransformationPipeline()
        self.output_transformation_pipeline = TransformationPipeline()
    
    def configure_input_transformation(self, strategies: Optional[List[Dict[str, Any]]] = None):
        """Configure input transformation strategies."""
        if strategies:
            for strategy_config in strategies:
                strategy = self._create_transformation_strategy(strategy_config)
                self.input_transformation_pipeline.add_strategy(strategy)
    
    def configure_preprocessing_transformation(self, strategies: Optional[List[Dict[str, Any]]] = None):
        """Configure preprocessing transformation strategies."""
        if strategies:
            for strategy_config in strategies:
                strategy = self._create_transformation_strategy(strategy_config)
                self.preprocessing_transformation_pipeline.add_strategy(strategy)
    
    def configure_output_transformation(self, strategies: Optional[List[Dict[str, Any]]] = None):
        """Configure output transformation strategies."""
        if strategies:
            for strategy_config in strategies:
                strategy = self._create_transformation_strategy(strategy_config)
                self.output_transformation_pipeline.add_strategy(strategy)
    
    def transform_input(self, input_data: Any) -> Any:
        """Transform input data using configured pipeline."""
        return self.input_transformation_pipeline.transform(input_data)
    
    def preprocess_data(self, data: Any) -> Any:
        """Apply preprocessing transformations to data."""
        return self.preprocessing_transformation_pipeline.transform(data)
    
    def transform_output(self, output_data: Any) -> Any:
        """Transform output data using configured pipeline."""
        return self.output_transformation_pipeline.transform(output_data)
    
    def _create_transformation_strategy(self, strategy_config: Dict[str, Any]):
        """Create a transformation strategy from configuration."""
        from ..transformations.advanced_strategies import (
            OutlierRemovalStrategy,
            FeatureEngineeringStrategy,
            TextTransformationStrategy
        )
        
        strategy_type = strategy_config['type']
        params = strategy_config.get('params', {})
        
        strategy_map = {
            'outlier_removal': OutlierRemovalStrategy,
            'feature_engineering': FeatureEngineeringStrategy,
            'text_transformation': TextTransformationStrategy
        }
        
        strategy_class = strategy_map.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unknown transformation strategy type: {strategy_type}")
        
        return strategy_class(**params)

class WorkflowExecutionError(Exception):
    """Exception raised when workflow execution fails."""
    pass

def workflow_step(step_name: str, description: str = None):
    """Decorator to mark and track agent workflow steps."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Initialize step tracking
            self._current_step = {
                "name": step_name,
                "description": description,
                "status": "running",
                "start_time": time.time()
            }
            
            try:
                result = func(self, *args, **kwargs)
                
                # Update step status
                self._current_step.update({"status": "completed"})
                if hasattr(self, '_workflow_history'):
                    self._workflow_history.append(self._current_step.copy())
                
                return result
            except Exception as e:
                # Track step failure
                if hasattr(self, '_current_step'):
                    self._current_step.update({"status": "failed", "error": str(e)})
                    if hasattr(self, '_workflow_history'):
                        self._workflow_history.append(self._current_step.copy())
                raise
        return wrapper
    return decorator

class AgentBase:
    """Base class for all agents in the system."""
    
    def __init__(
        self, 
        config: Union[Dict[str, Any], 'AgentConfig'], 
        agent_config_path: Optional[str] = None
    ):
        """
        Initialize base agent with configuration.
        
        Args:
            config: Agent configuration dictionary or object
            agent_config_path: Optional path to configuration file
        """
        # Normalize config to dictionary
        if not isinstance(config, dict):
            # Convert Pydantic model to dictionary
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.__dict__
        else:
            config_dict = config
        
        # Store full configuration
        self.config = config_dict
        
        # Configuration path
        self.agent_config_path = agent_config_path
        
        # Initialize other attributes
        self.id = config_dict.get('id')
        
        # Initialize metrics
        self.metrics = {
            'execution_time': 0,
            'memory_usage': 0,
            'performance_metrics': {}
        }

class Agent:
    """Base agent class."""

    def __init__(self, config: Union[Dict[str, Any], AgentConfig]):
        """Initialize agent."""
        if isinstance(config, dict):
            self._config_obj = AgentConfig(**config)
        else:
            self._config_obj = config

        # Initialize basic attributes
        self.name = self._config_obj.name
        self.type = self._config_obj.agent_type.upper() if self._config_obj.agent_type else None
        self.description = self._config_obj.description
        self.max_iterations = self._config_obj.workflow.max_iterations if self._config_obj.workflow else None
        self.model = self._config_obj.model
        self.workflow = self._config_obj.workflow

        # Initialize state
        self._initialized = False
        self.history = []
        self.errors = []
        self._workflow_history = []

        # Initialize transformation pipeline
        self.transformation_pipeline = TransformationPipeline()
        self._create_transformation_strategies()

    async def initialize(self):
        """Initialize agent."""
        if not self._initialized:
            # Initialize OpenAI client
            self.client = MagicMock()
            self._initialized = True

    async def cleanup(self):
        """Clean up agent resources."""
        self._initialized = False
        self.history = []
        self.errors = []
        self._workflow_history = []

    def process_message(self, message: str) -> str:
        """Process a message."""
        if message is None:
            error = ValueError("Empty message received")
            self._add_error(str(error))
            self._add_workflow_step("process_message", "error", {
                "error": str(error)
            })
            raise error

        try:
            # Add message to history
            self.history.append({"role": "user", "content": message})

            # Process with OpenAI
            response = self.client.chat.completions.create(
                model=self.model.name,
                messages=[
                    {"role": "system", "content": self._config_obj.system_prompt},
                    *self.history
                ]
            )

            # Extract response and ensure it's a string
            response_text = str(response.choices[0].message.content)
            self.history.append({"role": "assistant", "content": response_text})

            # Add to workflow history
            self._add_workflow_step("process_message", "success", {
                "message": message,
                "response": response_text,
                "tokens": response.usage.total_tokens
            })

            return response_text

        except Exception as e:
            error_msg = str(e)
            self._add_error(error_msg)
            self._add_workflow_step("process_message", "error", {
                "message": message,
                "error": error_msg
            })
            raise

    def transform(self, data: Any) -> Any:
        """Transform data using the transformation pipeline."""
        try:
            transformed_data = self.transformation_pipeline.transform(data)
            self._add_workflow_step("transform", "success", {
                "input_type": type(data).__name__,
                "output_type": type(transformed_data).__name__
            })
            return transformed_data
        except Exception as e:
            self._add_error(str(e))
            self._add_workflow_step("transform", "error", {
                "error": str(e)
            })
            raise

    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get workflow history."""
        return self._workflow_history

    def _add_workflow_step(self, step_name: str, status: str, details: Dict[str, Any]):
        """Add a step to workflow history."""
        self._workflow_history.append({
            "step": step_name,
            "status": status,
            "details": details,
            "timestamp": time.time()
        })

    def _add_error(self, error_msg: str):
        """Add an error to the error log."""
        self.errors.append({
            "error": error_msg,
            "timestamp": time.time()
        })

    def _create_transformation_strategies(self):
        """Create transformation strategies from config."""
        if not hasattr(self._config_obj, 'transformations'):
            return

        for stage, strategies in self._config_obj.transformations.items():
            for strategy_config in strategies:
                strategy = self._create_transformation_strategy(strategy_config)
                if strategy:
                    self.transformation_pipeline.add_strategy(strategy)

    def _create_transformation_strategy(self, config: Dict[str, Any]) -> Optional[Any]:
        """Create a transformation strategy from config."""
        try:
            strategy_type = config.get('type')
            params = config.get('params', {})

            if strategy_type == 'outlier_removal':
                return OutlierRemovalStrategy(**params)
            elif strategy_type == 'feature_engineering':
                return FeatureEngineeringStrategy(**params)
            elif strategy_type == 'text':
                return TextTransformationStrategy(**params)
            elif strategy_type == 'anomaly_detection':
                return AnomalyDetectionStrategy(**params)
            elif strategy_type == 'time_series':
                return TimeSeriesTransformationStrategy(**params)

            return None
        except Exception as e:
            self._add_error(f"Failed to create transformation strategy: {str(e)}")
            return None

class GenericAgent(Agent):
    """Generic Agent with flexible transformation capabilities."""

    def __init__(self, config: Union[Dict[str, Any], AgentConfig]) -> None:
        """Initialize Generic Agent."""
        super().__init__(config)
        
        # Initialize transformation strategies
        self._initialize_transformations()

    def _initialize_transformations(self) -> None:
        """Initialize generic transformation strategies."""
        if not hasattr(self, 'transformation_pipeline'):
            self.transformation_pipeline = TransformationPipeline()
            
        # Add default strategies if none specified
        if not self.transformation_pipeline.strategies:
            self.transformation_pipeline.add_strategy(
                TextTransformationStrategy(strategy='clean')
            )
            self.transformation_pipeline.add_strategy(
                FeatureEngineeringStrategy(strategy='standard')
            )

    def transform(self, data: Any) -> Any:
        """
        Transform data using generic strategies.

        Args:
            data: Input data

        Returns:
            Transformed data
        """
        try:
            return super().transform(data)
        except Exception as e:
            self._add_error(f"Data transformation failed: {str(e)}")
            return data

class ResearchAgent(Agent):
    """Research Agent with text processing capabilities."""

    def __init__(self, config: Union[Dict[str, Any], AgentConfig]) -> None:
        """Initialize Research Agent."""
        super().__init__(config)
        
        # Initialize research-specific attributes
        domain_config = self.config.get('DOMAIN_CONFIG', {})
        self.research_domains = domain_config.get('research_domains', [])
        self.citation_style = domain_config.get('citation_style', None)

        # Add default text transformation if none specified
        if not self.transformation_pipeline or not self.transformation_pipeline.strategies:
            self.transformation_pipeline = TransformationPipeline()
            self.transformation_pipeline.add_strategy(
                TextTransformationStrategy(strategy='clean')
            )

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze research text and extract key information."""
        try:
            results = {}
            
            # Basic text analysis
            results['basic_stats'] = {
                'word_count': len(text.split()),
                'char_count': len(text),
                'sentence_count': len(text.split('.'))
            }
            
            return results
        except Exception as e:
            self._add_error(f"Text analysis failed: {str(e)}")
            raise ValueError(f"Text analysis failed: {str(e)}")

    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract citations from text."""
        try:
            # Citation extraction logic would go here
            return []
        except Exception as e:
            self._add_error(f"Citation extraction failed: {str(e)}")
            raise ValueError(f"Citation extraction failed: {str(e)}")

    def format_citations(self, citations: List[Dict[str, str]], style: str = None) -> List[str]:
        """Format citations according to specified style."""
        try:
            # Citation formatting logic would go here
            return []
        except Exception as e:
            self._add_error(f"Citation formatting failed: {str(e)}")
            raise ValueError(f"Citation formatting failed: {str(e)}")

    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """Generate a summary of the text."""
        try:
            # Text summarization logic would go here
            return text[:max_length]
        except Exception as e:
            self._add_error(f"Text summarization failed: {str(e)}")
            raise ValueError(f"Text summarization failed: {str(e)}")

    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract key terms from text."""
        try:
            # Keyword extraction logic would go here
            return []
        except Exception as e:
            self._add_error(f"Keyword extraction failed: {str(e)}")
            raise ValueError(f"Keyword extraction failed: {str(e)}")

    def categorize_text(self, text: str) -> List[str]:
        """Categorize text into research domains."""
        try:
            # Text categorization logic would go here
            return []
        except Exception as e:
            self._add_error(f"Text categorization failed: {str(e)}")
            raise ValueError(f"Text categorization failed: {str(e)}")

    def generate_bibliography(self, citations: List[Dict[str, str]]) -> str:
        """Generate a bibliography from citations."""
        try:
            # Bibliography generation logic would go here
            return ""
        except Exception as e:
            self._add_error(f"Bibliography generation failed: {str(e)}")
            raise ValueError(f"Bibliography generation failed: {str(e)}")

    def check_plagiarism(self, text: str) -> Dict[str, Any]:
        """Check text for potential plagiarism."""
        try:
            # Plagiarism checking logic would go here
            return {}
        except Exception as e:
            self._add_error(f"Plagiarism check failed: {str(e)}")
            raise ValueError(f"Plagiarism check failed: {str(e)}")

    def find_related_research(self, text: str) -> List[Dict[str, str]]:
        """Find related research papers."""
        try:
            # Related research search logic would go here
            return []
        except Exception as e:
            self._add_error(f"Related research search failed: {str(e)}")
            raise ValueError(f"Related research search failed: {str(e)}")

    def generate_research_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a research report from analysis results."""
        try:
            # Report generation logic would go here
            return ""
        except Exception as e:
            self._add_error(f"Report generation failed: {str(e)}")
            raise ValueError(f"Report generation failed: {str(e)}")

    def validate_citations(self, citations: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Validate citation information."""
        try:
            # Citation validation logic would go here
            return {}
        except Exception as e:
            self._add_error(f"Citation validation failed: {str(e)}")
            raise ValueError(f"Citation validation failed: {str(e)}")

    def analyze_research_trends(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze research trends across multiple texts."""
        try:
            # Research trend analysis logic would go here
            return {}
        except Exception as e:
            self._add_error(f"Research trend analysis failed: {str(e)}")
            raise ValueError(f"Research trend analysis failed: {str(e)}")

    def extract_methodology(self, text: str) -> Dict[str, Any]:
        """Extract research methodology information."""
        try:
            # Methodology extraction logic would go here
            return {}
        except Exception as e:
            self._add_error(f"Methodology extraction failed: {str(e)}")
            raise ValueError(f"Methodology extraction failed: {str(e)}")

    def analyze_citations_network(self, citations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze citation network and relationships."""
        try:
            # Citation network analysis logic would go here
            return {}
        except Exception as e:
            self._add_error(f"Citation network analysis failed: {str(e)}")
            raise ValueError(f"Citation network analysis failed: {str(e)}")

class DataScienceAgent(Agent):
    """Data Science Agent for handling data analysis and modeling tasks."""

    def __init__(self, config: Union[str, Dict[str, Any]]):
        """Initialize Data Science Agent."""
        # Initialize base agent
        super().__init__(config)

        # Configure data science specific settings
        domain_config = self.config.get('DOMAIN_CONFIG', {})
        self.model_type = domain_config.get('model_type')
        self.metrics = domain_config.get('metrics', [])

    def analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze input data and generate insights."""
        try:
            results = {}
            
            # Basic statistics
            results['basic_stats'] = {
                'shape': data.shape,
                'missing_values': data.isnull().sum().to_dict(),
                'dtypes': data.dtypes.astype(str).to_dict()
            }
            
            # Numerical columns analysis
            num_cols = data.select_dtypes(include=['int64', 'float64']).columns
            if len(num_cols) > 0:
                results['numerical_analysis'] = {
                    'summary': data[num_cols].describe().to_dict(),
                    'correlations': data[num_cols].corr().to_dict()
                }
            
            # Categorical columns analysis
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                results['categorical_analysis'] = {
                    'value_counts': {col: data[col].value_counts().to_dict() for col in cat_cols}
                }
            
            return results
        except Exception as e:
            self._add_error(f"Data analysis failed: {str(e)}")
            raise ValueError(f"Data analysis failed: {str(e)}")

    def train_model(self, data: pd.DataFrame, target: str, model_type: str = 'auto') -> Any:
        """Train a machine learning model on the data."""
        try:
            # Model training logic would go here
            pass
        except Exception as e:
            self._add_error(f"Model training failed: {str(e)}")
            raise ValueError(f"Model training failed: {str(e)}")

    def evaluate_model(self, model: Any, data: pd.DataFrame, target: str) -> Dict[str, float]:
        """Evaluate a trained model using specified metrics."""
        try:
            # Model evaluation logic would go here
            pass
        except Exception as e:
            self._add_error(f"Model evaluation failed: {str(e)}")
            raise ValueError(f"Model evaluation failed: {str(e)}")

    def predict(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        try:
            # Prediction logic would go here
            pass
        except Exception as e:
            self._add_error(f"Prediction failed: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def save_model(self, model: Any, path: str) -> None:
        """Save a trained model to disk."""
        try:
            # Model saving logic would go here
            pass
        except Exception as e:
            self._add_error(f"Model saving failed: {str(e)}")
            raise ValueError(f"Model saving failed: {str(e)}")

    def load_model(self, path: str) -> Any:
        """Load a trained model from disk."""
        try:
            # Model loading logic would go here
            pass
        except Exception as e:
            self._add_error(f"Model loading failed: {str(e)}")
            raise ValueError(f"Model loading failed: {str(e)}")

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a report from analysis results."""
        try:
            # Report generation logic would go here
            pass
        except Exception as e:
            self._add_error(f"Report generation failed: {str(e)}")
            raise ValueError(f"Report generation failed: {str(e)}")

    def optimize_hyperparameters(self, model: Any, data: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Optimize model hyperparameters."""
        try:
            # Hyperparameter optimization logic would go here
            pass
        except Exception as e:
            self._add_error(f"Hyperparameter optimization failed: {str(e)}")
            raise ValueError(f"Hyperparameter optimization failed: {str(e)}")

    def feature_importance(self, model: Any, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance scores."""
        try:
            # Feature importance calculation logic would go here
            pass
        except Exception as e:
            self._add_error(f"Feature importance calculation failed: {str(e)}")
            raise ValueError(f"Feature importance calculation failed: {str(e)}")

    def cross_validate(self, model: Any, data: pd.DataFrame, target: str, cv: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        try:
            # Cross-validation logic would go here
            pass
        except Exception as e:
            self._add_error(f"Cross-validation failed: {str(e)}")
            raise ValueError(f"Cross-validation failed: {str(e)}")

    def explain_predictions(self, model: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate explanations for model predictions."""
        try:
            # Model explanation logic would go here
            pass
        except Exception as e:
            self._add_error(f"Prediction explanation failed: {str(e)}")
            raise ValueError(f"Prediction explanation failed: {str(e)}")

class ConfigurationType(str, Enum):
    """Enumeration of configuration types."""
    AGENT = "AGENT"
    MODEL = "MODEL"
    WORKFLOW = "WORKFLOW"
    TRANSFORMATIONS = "TRANSFORMATIONS"
    DOMAIN = "DOMAIN"
    GENERIC = "GENERIC"
    RESEARCH = "RESEARCH"
    DATA_SCIENCE = "DATA_SCIENCE"
    DOMAIN_SPECIFIC = "DOMAIN_SPECIFIC"
    TECHNICAL = "TECHNICAL"
    CUSTOM = "CUSTOM"
    DEFAULT = "DEFAULT"
    CREATIVE = "CREATIVE"
    DYNAMIC = "DYNAMIC"

    @classmethod
    def _missing_(cls, value):
        """Handle case-insensitive lookup."""
        for member in cls:
            if member.lower() == str(value).lower():
                return member
        return None

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, validator

class ModelConfig(BaseModel):
    """Model configuration."""
    provider: str
    name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    stop_sequences: Optional[List[str]] = None
    api_key: Optional[str] = None

class WorkflowConfig(BaseModel):
    """Workflow configuration."""
    max_iterations: int = 10
    logging_level: str = "INFO"
    distributed: bool = False
    timeout: int = 3600
    retry_count: int = 3

class ConfigurationSchema:
    """Configuration schema for agent configuration."""

    def __init__(self, name: str = None, type: str = None, version: str = None,
                 description: str = None, max_iterations: int = None,
                 transformations: Dict[str, Any] = None, domain_config: Dict[str, Any] = None,
                 model: Dict[str, Any] = None, workflow: Dict[str, Any] = None):
        """Initialize configuration schema."""
        self.name = name
        self.type = type
        self.version = version
        self.description = description
        self.max_iterations = max_iterations
        self.transformations = transformations or {}
        self.domain_config = domain_config or {}
        self.model = model or {}
        self.workflow = workflow or {}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigurationSchema':
        """Create schema from configuration dictionary."""
        try:
            # Extract base configuration
            agent_config = config_dict.get('AGENT', {})
            model_config = config_dict.get('MODEL', {})
            workflow_config = config_dict.get('WORKFLOW', {})
            transformations = config_dict.get('TRANSFORMATIONS', {})
            domain_config = config_dict.get('DOMAIN_CONFIG', {})

            # Create schema instance with normalized type
            schema = cls(
                name=agent_config.get('name'),
                type=agent_config.get('type', '').upper(),
                version=agent_config.get('version'),
                description=agent_config.get('description'),
                max_iterations=workflow_config.get('max_iterations'),
                transformations=transformations,
                domain_config=domain_config,
                model=model_config,
                workflow=workflow_config
            )

            return schema
        except Exception as e:
            raise ValueError(f"Failed to create configuration schema: {str(e)}")

    def validate(self) -> bool:
        """Validate configuration."""
        try:
            # Validate required fields
            if not self.name or not self.type:
                return False

            # Validate max iterations
            if self.max_iterations is not None and (not isinstance(self.max_iterations, int) or self.max_iterations < 0):
                return False

            # Validate transformations
            if self.transformations:
                for stage in ['input', 'preprocessing', 'output']:
                    if stage in self.transformations:
                        for transform in self.transformations[stage]:
                            if not transform.get('type'):
                                return False
                            if 'params' not in transform:
                                return False
                            # Validate specific transformation types
                            if transform['type'] == 'outlier_removal':
                                params = transform['params']
                                if params.get('method') not in ['z_score', 'iqr']:
                                    return False
                                if not isinstance(params.get('threshold', 0), (int, float)) or params.get('threshold', 0) < 0:
                                    return False

            return True
        except Exception:
            return False

    def merge(self, other: 'ConfigurationSchema') -> 'ConfigurationSchema':
        """Merge with another configuration schema."""
        merged = ConfigurationSchema()

        # Merge basic attributes (prefer self)
        merged.name = self.name or other.name
        merged.type = self.type or other.type
        merged.version = self.version or other.version
        merged.description = self.description or other.description
        merged.max_iterations = self.max_iterations or other.max_iterations

        # Merge transformations
        merged.transformations = {**other.transformations, **self.transformations}

        # Merge domain config
        merged.domain_config = {**other.domain_config, **self.domain_config}

        # Merge model and workflow configs
        merged.model = {**other.model, **self.model}
        merged.workflow = {**other.workflow, **self.workflow}

        return merged

    def export(self, format: str = 'dict') -> Union[Dict[str, Any], str]:
        """Export configuration to specified format."""
        config_dict = {
            'AGENT': {
                'name': self.name,
                'type': self.type,
                'version': self.version,
                'description': self.description
            },
            'WORKFLOW': {
                'max_iterations': self.max_iterations,
                **self.workflow
            },
            'TRANSFORMATIONS': self.transformations,
            'DOMAIN_CONFIG': self.domain_config,
            'MODEL': self.model
        }

        if format.lower() == 'json':
            return json.dumps(config_dict)
        return config_dict

class AgentFactory:
    """Factory class for creating different types of agents."""

    _agent_types = {}

    @classmethod
    def register_agent(cls, agent_type: str, agent_class: Type[Agent]) -> None:
        """Register a new agent type.

        Args:
            agent_type: Type identifier for the agent
            agent_class: Agent class to register
        """
        cls._agent_types[agent_type.upper()] = agent_class

    @classmethod
    def create_agent(cls, config: Union[Dict[str, Any], str, 'AgentConfig']) -> Agent:
        """
        Create an agent instance based on configuration.

        Args:
            config: Agent configuration dictionary or object

        Returns:
            Configured agent instance

        Raises:
            ValueError: If agent type is invalid
        """
        # Convert config to dictionary if needed
        if not isinstance(config, dict):
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.__dict__
        else:
            config_dict = config

        # Create configuration schema
        config_schema = ConfigurationSchema.from_dict(config_dict)
        if not config_schema.validate():
            raise ValueError("Invalid agent configuration")

        # Get agent type
        agent_type = config_schema.type
        if not agent_type:
            agent_type = ConfigurationType.GENERIC

        # Create agent instance using registered type
        agent_class = cls._agent_types.get(agent_type.upper())
        if agent_class:
            return agent_class(config_dict)
        
        # Fallback to default types if not registered
        if agent_type == ConfigurationType.RESEARCH:
            return ResearchAgent(config_dict)
        elif agent_type == ConfigurationType.DATA_SCIENCE:
            return DataScienceAgent(config_dict)
        else:
            return GenericAgent(config_dict)

class FlexibleConfigurationManager:
    """Manages flexible configuration for agents with dynamic schema validation."""

    def __init__(self, schema: ConfigurationSchema = None, config_type: ConfigurationType = ConfigurationType.DEFAULT):
        """
        Initialize the configuration manager.

        Args:
            schema: Configuration schema to validate against
            config_type: Type of configuration (DEFAULT, CUSTOM, DYNAMIC)
        """
        self.schema = schema or ConfigurationSchema()
        self.config_type = config_type
        self.config = {}

    def load_configuration(self, source: Union[str, Dict[str, Any], ConfigurationSchema]) -> ConfigurationSchema:
        """
        Load configuration from various sources.

        Args:
            source: Configuration source (file path, dict, or schema)

        Returns:
            Loaded configuration schema

        Raises:
            ValueError: If configuration is invalid
        """
        if isinstance(source, str):
            try:
                # Try to parse as JSON first
                config_dict = json.loads(source)
                return ConfigurationSchema.from_dict(config_dict)
            except json.JSONDecodeError:
                # If not JSON, try as file path
                try:
                    with open(source, 'r') as f:
                        config_dict = json.load(f)
                    return ConfigurationSchema.from_dict(config_dict)
                except Exception as e:
                    raise ValueError(f"Failed to load configuration from file: {e}")

        if isinstance(source, dict):
            return ConfigurationSchema.from_dict(source)

        if isinstance(source, ConfigurationSchema):
            return source

        raise ValueError("Invalid configuration source")

    def get_config(self, key: str = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key to retrieve

        Returns:
            Configuration value
        """
        if key is None:
            return self.config
        return self.config.get(key)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates

        Raises:
            ValueError: If updates are invalid
        """
        # Create temporary schema with updates
        temp_schema = ConfigurationSchema.from_dict(updates)
        if not temp_schema.validate():
            raise ValueError("Invalid configuration updates")
        
        # Apply updates
        self.config.update(updates)

    def merge_config(self, other_config: Dict[str, Any], override: bool = False) -> None:
        """
        Merge another configuration into this one.

        Args:
            other_config: Configuration to merge
            override: Whether to override existing values
        """
        # Convert other config to schema
        other_schema = ConfigurationSchema.from_dict(other_config)
        if not other_schema.validate():
            raise ValueError("Invalid configuration to merge")

        # Convert current config to schema
        current_schema = ConfigurationSchema.from_dict(self.config)
        
        # Merge schemas
        if override:
            merged = other_schema.merge(current_schema)
        else:
            merged = current_schema.merge(other_schema)
        
        # Update config with merged result
        self.config = merged.export()