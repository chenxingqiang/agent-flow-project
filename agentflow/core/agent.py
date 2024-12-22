"""Base Agent Module"""

from typing import Dict, Any, Optional, Union, List, Callable
import logging
import asyncio
from functools import partial, wraps
import importlib
import json
from pathlib import Path
from .config import AgentConfig
from ..transformations.advanced_strategies import AdvancedTransformationStrategy
from ..core.transformation import TransformationPipeline
from .isa_manager import ISAManager, Instruction, InstructionType
from .instruction_selector import InstructionSelector
from .rl_optimizer import create_optimizer, RLOptimizer
import time

logger = logging.getLogger(__name__)

def workflow_step(step_name: str, description: str = None):
    """Decorator to mark and track agent workflow steps, inspired by ell's versioning"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            logger.info(f"Executing workflow step: {step_name}")
            try:
                # Track step execution for monitoring
                if not hasattr(self, '_current_step'):
                    self._current_step = {}
                
                self._current_step.update({
                    "name": step_name,
                    "description": description or func.__doc__,
                    "status": "running"
                })
                
                result = await func(self, *args, **kwargs)
                
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
    
    def __init__(self, config: Optional[Union[Dict[str, Any], 'AgentConfig']] = None):
        """Initialize base agent.
        
        Args:
            config (Optional[Union[Dict[str, Any], 'AgentConfig']]): Agent configuration
        """
        self.config = config or {}
        if isinstance(self.config, dict):
            self.name = self.config.get('name', 'base_agent')
            self.type = self.config.get('type', 'base')
        else:
            # Handle Pydantic model
            self.name = getattr(self.config, 'name', 'base_agent')
            self.type = getattr(self.config, 'type', 'base')
        
        self._workflow_history = []
        self._current_step = {}

class Agent(AgentBase):
    """Advanced agent with workflow tracking and optimization capabilities."""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], 'AgentConfig']] = None):
        """Initialize agent with configuration.
        
        Args:
            config (Optional[Union[Dict[str, Any], 'AgentConfig']]): Agent configuration
        """
        super().__init__(config)
        
        # Initialize components
        self.isa_manager = ISAManager()
        self.instruction_selector = InstructionSelector()
        
        # Initialize optimizer only if algorithm is specified
        if isinstance(config, dict):
            algorithm = config.get('algorithm')
        else:
            algorithm = getattr(config, 'algorithm', None)
            
        self.optimizer = create_optimizer(config) if algorithm else None
        
        # Initialize pipelines
        self.input_pipeline = TransformationPipeline()
        self.output_pipeline = TransformationPipeline()
        
        # Initialize metrics
        self.token_count = 0
        self.last_latency = 0
        self.memory_usage = 0
        self.history = []
        self.errors = []
        
        # Flag to track initialization
        self._initialized = False
    
    @classmethod
    async def create(cls, config: Optional[Union[Dict[str, Any], 'AgentConfig']] = None):
        """Create and initialize an agent.
        
        Args:
            config (Optional[Union[Dict[str, Any], 'AgentConfig']]): Agent configuration
            
        Returns:
            Agent: Initialized agent instance
        """
        agent = cls(config)
        await agent.initialize()
        return agent
    
    async def initialize(self):
        """Initialize agent asynchronously"""
        if self._initialized:
            return self
        
        # Initialize base state
        self.id = self.config.id or 'default_agent'
        self.type = self.config.type or 'base'
        self.capabilities = []  # Initialize empty capabilities list
        
        # Initialize metrics
        self.token_count = 0
        self.last_latency = 0
        self.memory_usage = 0
        
        # Initialize transformation functions
        self._input_transform = None
        self._output_transform = None
        self._preprocess_transform = None
        
        # Initialize components
        await self.isa_manager.initialize()
        await self.instruction_selector.initialize()
        
        # Load agent instructions
        await self._load_agent_instructions()
        
        # Load strategies
        await self._load_strategies()
        
        # Initialize optimizer if needed
        if self.optimizer:
            await self.optimizer.initialize()
        
        self._initialized = True
        return self
    
    @workflow_step("load_strategies", "Load transformation strategies")
    async def _load_strategies(self):
        """Load transformation strategies from configuration."""
        try:
            strategies = []
            if isinstance(self.config, dict):
                strategies = self.config.get('strategies', [])
            else:
                strategies = getattr(self.config, 'strategies', [])
                
            for strategy in strategies:
                if isinstance(strategy, dict):
                    strategy_type = strategy.get('type')
                    params = strategy.get('params', {})
                else:
                    strategy_type = getattr(strategy, 'type', None)
                    params = getattr(strategy, 'params', {})
                
                if strategy_type:
                    # Create and add strategy
                    strategy_instance = AdvancedTransformationStrategy(
                        strategy_type=strategy_type,
                        **params
                    )
                    self.input_pipeline.add_strategy(strategy_instance)
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
            raise
    
    @workflow_step("load_instructions", "Load and initialize agent instructions")
    async def _load_agent_instructions(self):
        """Load and initialize instructions"""
        try:
            # Load default instructions
            instructions = []
            if isinstance(self.config, dict):
                instructions = self.config.get('instructions', [])
            else:
                instructions = getattr(self.config, 'instructions', [])
            
            # Initialize ISA with instructions
            for instruction in instructions:
                if isinstance(instruction, dict):
                    instruction_type = instruction.get('type', InstructionType.BASIC)
                    name = instruction.get('name')
                    description = instruction.get('description')
                    dependencies = instruction.get('dependencies', [])
                    cost = instruction.get('cost', 1.0)
                    parallelizable = instruction.get('parallelizable', False)
                    agent_requirements = instruction.get('agent_requirements', [])
                else:
                    instruction_type = getattr(instruction, 'type', InstructionType.BASIC)
                    name = getattr(instruction, 'name')
                    description = getattr(instruction, 'description')
                    dependencies = getattr(instruction, 'dependencies', [])
                    cost = getattr(instruction, 'cost', 1.0)
                    parallelizable = getattr(instruction, 'parallelizable', False)
                    agent_requirements = getattr(instruction, 'agent_requirements', [])
                
                if name:
                    instruction_obj = Instruction(
                        name=name,
                        type=instruction_type,
                        description=description,
                        dependencies=dependencies,
                        cost=cost,
                        parallelizable=parallelizable,
                        agent_requirements=agent_requirements
                    )
                    self.isa_manager.register_instruction(instruction_obj)
            
            # Load from config path if specified
            if hasattr(self.config, "isa_config_path"):
                self.isa_manager.load_instructions(self.config.isa_config_path)
            
            # Train instruction selector
            self.instruction_selector.train(self.isa_manager.instructions)
            return self
            
        except Exception as e:
            logger.error(f"Error loading instructions: {e}")
            raise

    def transform_input(self, input_data: Dict[str, Any]) -> List[str]:
        """Transform input data into optimal instruction sequence"""
        # Select best instructions based on input
        selected = self.instruction_selector.select_instructions(
            input_data,
            self.isa_manager.instructions
        )
        
        # Optimize instruction sequence
        optimized = self.instruction_selector.optimize_sequence(
            selected,
            self.isa_manager.instructions
        )
        
        return optimized

    def _get_environment_state(self) -> Dict[str, Any]:
        """Get current environment state for RL"""
        stats = self.get_instruction_stats()
        
        return {
            "load": stats.get("current_load", 0.5),
            "latency": stats.get("avg_latency", 50.0),
            "success_rate": stats.get("success_rate", 0.9),
            "cache_hit_rate": stats.get("cache_hit_rate", 0.5),
            "resource_usage": {
                "cpu_usage": stats.get("cpu_usage", 0.5),
                "memory_usage": stats.get("memory_usage", 0.5),
                "network_usage": stats.get("network_usage", 0.5),
                "disk_usage": stats.get("disk_usage", 0.5)
            }
        }
    
    def _apply_rl_optimization(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply RL-based optimization to input processing"""
        # Get current state
        state = self._get_environment_state()
        
        # Get optimized parameters from RL
        params = self.optimizer.get_action(state)
        
        # Apply optimization parameters
        optimized_input = dict(input_data)
        optimized_input.update({
            "batch_size": int(params["batch_size"][0]),
            "num_parallel": int(params["num_parallel"][0]),
            "cache_size": int(params["cache_size"][0]),
            "timeout": float(params["timeout"][0])
        })
        
        return optimized_input

    @workflow_step("process_input", "Process user input according to ISA instructions")
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input using RL-optimized instruction sequence"""
        try:
            # Apply RL optimization
            optimized_input = self._apply_rl_optimization(input_data)
            
            # Transform input into instruction sequence
            instructions = self.transform_input(optimized_input)
            
            # Execute optimized sequence
            results = await self.isa_manager.execute_instruction_sequence(
                instructions,
                context={"input": optimized_input, "agent": self}
            )
            
            # Update instruction history with metrics
            for instr_name in instructions:
                metrics = self.isa_manager.get_instruction_metrics(instr_name)
                self.instruction_selector.update_history(instr_name, metrics)
            
            return self.combine_results(results)
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            raise

    async def process_message(self, message: str) -> str:
        """Process a message and return a response.
        
        Args:
            message: Input message to process
            
        Returns:
            Response string
            
        Raises:
            Exception: If message is None or processing fails
        """
        try:
            if message is None:
                raise ValueError("Message cannot be None")
                
            # Track message in history
            self.history.append({
                "message": message,
                "timestamp": time.time()
            })
            
            # Call LLM and get response
            response = await self._call_llm({"message": message})
            
            # Update metrics
            self.token_count += response.get("tokens", 0)
            self.last_latency = response.get("latency", 0)
            
            return response.get("response", "")
            
        except Exception as e:
            # Track error with full context
            error = {
                "error": str(e),
                "message": message,
                "timestamp": time.time(),
                "type": type(e).__name__
            }
            self.errors.append(error)
            raise
    
    def train_rl_optimizer(self, num_iterations: int = 100):
        """Train the RL optimizer"""
        self.optimizer.train(num_iterations)
    
    def save_rl_model(self, path: str):
        """Save trained RL model"""
        self.optimizer.save_model(path)
    
    def load_rl_model(self, path: str):
        """Load trained RL model"""
        self.optimizer.load_model(path)

    def combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple instructions into final output"""
        # This is a placeholder implementation
        # Actual implementation should properly combine results based on instruction types
        if not results:
            return {}
        
        combined = {}
        for result in results:
            combined.update(result)
        return combined

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about instruction execution"""
        return self.isa_manager.get_instruction_stats()

    async def _call_llm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call language model and get response"""
        # Mock implementation for testing
        if input_data is None:
            raise ValueError("Input data cannot be None")
            
        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary")
            
        message = input_data.get("message")
        if message is None:
            raise ValueError("Message cannot be None")
            
        # Mock response for testing
        return {
            "response": f"Mock response to: {message}",
            "tokens": 10,
            "latency": 0.1
        }

    @workflow_step("cleanup", "Reset agent state and metrics")
    async def cleanup(self):
        """Clean up agent state"""
        # Clear history and errors
        self.history = []
        self.errors = []
        
        # Reset metrics
        self.token_count = 0
        self.last_latency = 0
        self.memory_usage = 0
        
        # Reset workflow tracking
        self._workflow_history = []
        self._current_step = {}
        
        # Reset initialization flag
        self._initialized = False
        
        # Clean up components
        await self.isa_manager.cleanup()
        await self.instruction_selector.cleanup()
        
        # Clean up optimizer if present
        if self.optimizer:
            await self.optimizer.cleanup()
            
        # Reset transformation functions
        self._input_transform = None
        self._output_transform = None 
        self._preprocess_transform = None
        
        # Reset pipelines
        await self.input_pipeline.cleanup()
        await self.output_pipeline.cleanup()
        
        return self

    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get the history of workflow steps executed, inspired by ell's monitoring"""
        return self._workflow_history

    async def transform_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform input data using configured transform function"""
        if self._input_transform:
            return await self._input_transform(input_data)
        return input_data
    
    async def transform_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform output data using configured transform function"""
        if self._output_transform:
            return await self._output_transform(output_data)
        return output_data
    
    async def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data using configured transform function"""
        if self._preprocess_transform:
            return await self._preprocess_transform(data)
        return data
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data through the agent's pipeline
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        try:
            # Transform input
            transformed_input = await self.transform_input(input_data)
            
            # Preprocess
            preprocessed = await self.preprocess(transformed_input)
            
            # Core processing logic - override in subclasses
            result = await self._process_core(preprocessed)
            
            # Transform output
            final_output = await self.transform_output(result)
            
            return final_output
            
        except Exception as e:
            logger.error(f"Error in agent processing: {str(e)}")
            raise
    
    async def _process_core(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core processing logic to be implemented by subclasses
        
        Args:
            data: Preprocessed input data
            
        Returns:
            Processed data before output transformation
        """
        # Base implementation just returns the input
        return data

class AgentTransformationMixin:
    """Mixin class to add advanced transformation capabilities to agents."""
    def __init__(self, *args, **kwargs):
        """Initialize transformation-related attributes."""
        super().__init__(*args, **kwargs)
        
        # Transformation pipelines for different workflow stages
        self.input_transformation_pipeline = TransformationPipeline()
        self.preprocessing_transformation_pipeline = TransformationPipeline()
        self.output_transformation_pipeline = TransformationPipeline()
    
    def configure_input_transformation(
        self, 
        strategies: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Configure input transformation strategies.
        
        Args:
            strategies: List of strategy configurations
        """
        if strategies is None:
            return
        
        try:
            for strategy_config in strategies:
                strategy = self._create_transformation_strategy(strategy_config)
                self.input_transformation_pipeline.add_strategy(strategy)
        except Exception as e:
            logger.error(f"Input transformation configuration failed: {e}")
            raise ValueError(f"Input transformation configuration failed: {e}")

    def transform_input(self, input_data: Any):
        """
        Apply input transformation pipeline.
        
        Args:
            input_data: Raw input data
        
        Returns:
            Transformed input data
        """
        try:
            return self.input_transformation_pipeline.transform(input_data)
        except Exception as e:
            logger.error(f"Input transformation pipeline failed: {e}")
            raise ValueError(f"Input transformation pipeline failed: {e}")

    def configure_preprocessing_transformation(
        self, 
        strategies: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Configure preprocessing transformation strategies.
        
        Args:
            strategies: List of strategy configurations
        """
        if strategies is None:
            return
        
        try:
            for strategy_config in strategies:
                strategy = self._create_transformation_strategy(strategy_config)
                self.preprocessing_transformation_pipeline.add_strategy(strategy)
        except Exception as e:
            logger.error(f"Preprocessing transformation configuration failed: {e}")
            raise ValueError(f"Preprocessing transformation configuration failed: {e}")

    def preprocess_data(self, data: Any):
        """
        Apply preprocessing transformation pipeline.
        
        Args:
            data: Data to preprocess
        
        Returns:
            Preprocessed data
        """
        try:
            return self.preprocessing_transformation_pipeline.transform(data)
        except Exception as e:
            logger.error(f"Preprocessing transformation pipeline failed: {e}")
            raise ValueError(f"Preprocessing transformation pipeline failed: {e}")

    def configure_output_transformation(
        self, 
        strategies: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Configure output transformation strategies.
        
        Args:
            strategies: List of strategy configurations
        """
        if strategies is None:
            return
        
        try:
            for strategy_config in strategies:
                strategy = self._create_transformation_strategy(strategy_config)
                self.output_transformation_pipeline.add_strategy(strategy)
        except Exception as e:
            logger.error(f"Output transformation configuration failed: {e}")
            raise ValueError(f"Output transformation configuration failed: {e}")

    def transform_output(self, output_data: Any):
        """
        Apply output transformation pipeline.
        
        Args:
            output_data: Raw output data
        
        Returns:
            Transformed output data
        """
        try:
            return self.output_transformation_pipeline.transform(output_data)
        except Exception as e:
            logger.error(f"Output transformation pipeline failed: {e}")
            raise ValueError(f"Output transformation pipeline failed: {e}")

    def _create_transformation_strategy(self, strategy_config: Dict[str, Any]) -> AdvancedTransformationStrategy:
        """
        Create a transformation strategy based on configuration.
        
        Args:
            strategy_config: Configuration dictionary for the strategy
        
        Returns:
            Instantiated transformation strategy
        """
        strategy_type = strategy_config.get('type')
        strategy_params = strategy_config.get('params', {})
        
        # Dynamically import and instantiate strategy based on type
        try:
            strategy_module = importlib.import_module('agentflow.transformations')
            strategy_class = getattr(strategy_module, f"{strategy_type.capitalize()}TransformationStrategy")
            return strategy_class(**strategy_params)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to create transformation strategy: {e}")
            raise ValueError(f"Invalid transformation strategy: {strategy_type}")

class DataScienceAgent(AgentBase):
    """Enhanced Data Science Agent with transformation capabilities."""
    
    def __init__(self, config: Union[Dict[str, Any], 'AgentConfig']):
        """Initialize DataScienceAgent with configuration."""
        super().__init__(config)
        
        # Initialize transformation pipelines
        self.input_transformation_pipeline = TransformationPipeline()
        self.preprocessing_transformation_pipeline = TransformationPipeline()
        self.output_transformation_pipeline = TransformationPipeline()
        
        # Default transformation configurations
        default_input_transformations = [
            {
                'type': 'outlier_removal',
                'params': {
                    'method': 'iqr',
                    'threshold': 1.5
                }
            }
        ]
        
        default_preprocessing_transformations = [
            {
                'type': 'feature_engineering',
                'params': {
                    'strategy': 'log'
                }
            }
        ]
        
        default_output_transformations = [
            {
                'type': 'feature_engineering',
                'params': {
                    'strategy': 'binning',
                    'degree': 5
                }
            }
        ]
        
        # Configure default transformations
        self.configure_input_transformation(default_input_transformations)
        self.configure_preprocessing_transformation(default_preprocessing_transformations)
        self.configure_output_transformation(default_output_transformations)
    
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
    
    def _create_transformation_strategy(self, strategy_config: Dict[str, Any]) -> AdvancedTransformationStrategy:
        """Create a transformation strategy from configuration."""
        try:
            strategy_type = strategy_config['type']
            strategy_params = strategy_config.get('params', {})
            
            strategy_module = importlib.import_module('agentflow.transformations')
            strategy_class = getattr(strategy_module, f"{strategy_type.capitalize()}TransformationStrategy")
            return strategy_class(**strategy_params)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to create transformation strategy: {e}")
            raise ValueError(f"Invalid transformation strategy: {strategy_type}")

class AgentFactory:
    """
    Factory class for creating and managing different types of agents.
    Supports lazy loading of agent implementations to avoid circular imports.
    """
    _agent_registry = {}
    _agent_configs = {}
    
    @classmethod
    def register_agent(cls, agent_type: str, agent_class: type):
        """
        Register an agent class for a given type
        
        Args:
            agent_type: Type identifier for the agent
            agent_class: Agent class to register
        """
        cls._agent_registry[agent_type.lower()] = agent_class
    
    @classmethod
    def register_config(cls, agent_type: str, config: Union[Dict[str, Any], 'AgentConfig']):
        """
        Register configuration for an agent type
        
        Args:
            agent_type: Type identifier for the agent
            config: Configuration for the agent
        """
        cls._agent_configs[agent_type.lower()] = config
    
    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        config: Optional[Union[Dict[str, Any], 'AgentConfig']] = None,
        agent_config_path: Optional[str] = None
    ) -> AgentBase:
        """
        Create an agent instance of the specified type
        
        Args:
            agent_type: Type of agent to create
            config: Optional configuration for the agent
            agent_config_path: Optional path to configuration file
            
        Returns:
            AgentBase: Agent instance
            
        Raises:
            ValueError: If agent type is not registered
        """
        agent_type = agent_type.lower()
        
        # Get agent class
        if agent_type not in cls._agent_registry:
            # Try lazy loading the agent class
            try:
                if agent_type == "research":
                    from .research_agent import ResearchAgent
                    cls.register_agent("research", ResearchAgent)
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")
            except ImportError as e:
                raise ValueError(f"Failed to load agent type {agent_type}: {e}")
        
        agent_class = cls._agent_registry[agent_type]
        
        # Get configuration
        if config is None:
            config = cls._agent_configs.get(agent_type, {})
        
        # Create agent instance
        return agent_class(config)
    
    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get list of registered agent types"""
        return list(cls._agent_registry.keys())

# Now register the default agent types after all classes are defined
AgentFactory.register_agent('base', AgentBase)
AgentFactory.register_agent('advanced', Agent)
AgentFactory.register_agent('data_science', DataScienceAgent)
# Other default registrations can be added here

class ConfigurationType:
    """Enum-like class for configuration types"""
    RESEARCH = "RESEARCH"
    DATA_SCIENCE = "DATA_SCIENCE"
    GENERIC = "GENERIC"
    CREATIVE = "CREATIVE"
    TECHNICAL = "TECHNICAL"

class ConfigurationSchema:
    """
    Flexible configuration schema for agents.
    Provides a standardized way to define and validate agent configurations.
    """
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration schema.
        
        :param config_dict: Dictionary containing configuration details
        """
        self.config = config_dict
        self._validate_config()
    
    def _validate_config(self):
        """
        Validate the configuration against basic requirements.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ['type', 'name']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        valid_types = [
            ConfigurationType.RESEARCH, 
            ConfigurationType.DATA_SCIENCE, 
            ConfigurationType.GENERIC,
            ConfigurationType.CREATIVE,
            ConfigurationType.TECHNICAL
        ]
        
        if self.config.get('type') not in valid_types:
            raise ValueError(f"Invalid agent type: {self.config.get('type')}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        :return: Configuration dictionary
        """
        return self.config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigurationSchema':
        """
        Create a ConfigurationSchema instance from a dictionary.
        
        :param config_dict: Dictionary containing configuration details
        :return: ConfigurationSchema instance
        """
        return cls(config_dict)

class FlexibleConfigurationManager:
    """
    Advanced configuration management for agents.
    Supports dynamic configuration loading, validation, and transformation.
    """
    def __init__(self, base_config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.
        
        :param base_config: Optional base configuration dictionary
        """
        self.base_config = base_config or {}
        self.configurations: Dict[str, ConfigurationSchema] = {}
    
    def add_configuration(
        self, 
        name: str, 
        config: Union[Dict[str, Any], ConfigurationSchema]
    ):
        """
        Add a configuration to the manager.
        
        :param name: Unique name for the configuration
        :param config: Configuration dictionary or schema
        """
        if isinstance(config, dict):
            config = ConfigurationSchema(config)
        
        self.configurations[name] = config
    
    def get_configuration(self, name: str) -> ConfigurationSchema:
        """
        Retrieve a configuration by name.
        
        :param name: Name of the configuration
        :return: Configuration schema
        :raises KeyError: If configuration not found
        """
        return self.configurations[name]
    
    def merge_configurations(
        self, 
        base_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Merge multiple configurations with a base configuration.
        
        :param base_config: Optional base configuration to merge with
        :return: Merged configuration dictionary
        """
        merged_config = base_config or self.base_config.copy()
        
        for config in self.configurations.values():
            merged_config.update(config.to_dict())
        
        return merged_config

    def load_configuration(self, config: Union[Dict[str, Any], str]) -> ConfigurationSchema:
        """
        Load a configuration from various sources.
        
        :param config: Configuration dictionary, JSON string, or partial configuration
        :return: ConfigurationSchema instance
        """
        if isinstance(config, str):
            try:
                # Try parsing as JSON string
                config = json.loads(config)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid configuration format: {config}")
        
        # Merge with base configuration if needed
        if self.base_config:
            # Deep merge of configurations
            merged_config = {**self.base_config, **config}
        else:
            merged_config = config
        
        # Create and validate configuration schema
        config_schema = ConfigurationSchema.from_dict(merged_config)
        
        return config_schema

# Additional Agent Types
class GenericAgent(AgentBase):
    """
    Flexible generic agent with configurable behavior.
    Supports dynamic strategy and transformation configuration.
    """
    def __init__(
        self, 
        config: Union[Dict[str, Any], 'AgentConfig'], 
        agent_config_path: Optional[str] = None
    ):
        """
        Initialize generic agent with flexible configuration.
        
        :param config: Agent configuration
        :param agent_config_path: Optional path to configuration file
        """
        super().__init__(config)
        
        # Additional generic agent initialization logic
        self.dynamic_strategies = {}
    
    def add_dynamic_strategy(
        self, 
        strategy_name: str, 
        strategy_func: Callable
    ):
        """
        Add a dynamic strategy to the agent.
        
        :param strategy_name: Name of the strategy
        :param strategy_func: Function implementing the strategy
        """
        self.dynamic_strategies[strategy_name] = strategy_func
    
    def execute_dynamic_strategy(
        self, 
        strategy_name: str, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a dynamically added strategy.
        
        :param strategy_name: Name of the strategy to execute
        :param input_data: Input data for the strategy
        :return: Strategy execution result
        :raises KeyError: If strategy not found
        """
        if strategy_name not in self.dynamic_strategies:
            raise KeyError(f"Dynamic strategy not found: {strategy_name}")
        
        return self.dynamic_strategies[strategy_name](input_data)

# Lazy import of ResearchAgent to avoid circular import
from .research_agent import ResearchAgent
AgentFactory.register_agent('research', ResearchAgent)