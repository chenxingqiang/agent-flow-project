"""Base classes for the agent framework."""

from typing import Dict, Any, Optional, Union, List, Callable, Type
from enum import Enum
import logging
import asyncio

from .config import AgentConfig

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
        self.output_transformation_pipeline = TransformationPipeline()
    
    def configure_input_transformation(self, strategies: Optional[List[Dict[str, Any]]] = None):
        """Configure input transformation strategies."""
        if strategies:
            for strategy_config in strategies:
                strategy = self._create_transformation_strategy(strategy_config)
                self.input_transformation_pipeline.add_strategy(strategy)
    
    def configure_output_transformation(self, strategies: Optional[List[Dict[str, Any]]] = None):
        """Configure output transformation strategies."""
        if strategies:
            for strategy_config in strategies:
                strategy = self._create_transformation_strategy(strategy_config)
                self.output_transformation_pipeline.add_strategy(strategy)
    
    def transform_input(self, input_data: Any) -> Any:
        """Transform input data using configured pipeline."""
        return self.input_transformation_pipeline.transform(input_data)
    
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
        
        # List to store sub-agents
        self.agents = []
    
    def add_agent(self, agent: 'AgentBase'):
        """Add a sub-agent to this agent."""
        self.agents.append(agent)
    
    def remove_agent(self, agent: 'AgentBase'):
        """Remove a sub-agent from this agent."""
        self.agents.remove(agent)
    
    @classmethod
    def from_config(cls, config_path: str) -> 'AgentBase':
        """Create an agent from a configuration file."""
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config)

class ConfigurationType(str, Enum):
    """Agent configuration types."""
    RESEARCH = "RESEARCH"
    DATA_SCIENCE = "DATA_SCIENCE"
    DOMAIN_SPECIFIC = "DOMAIN_SPECIFIC"
    TECHNICAL = "TECHNICAL"
    GENERIC = "GENERIC"
    CREATIVE = "CREATIVE"
