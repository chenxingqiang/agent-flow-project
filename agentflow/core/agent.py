"""Agent module for handling different types of agents."""

import json
import logging
import os
import time
import asyncio
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

from agentflow.transformations.advanced_strategies import (
    AdvancedTransformationStrategy,
    FeatureEngineeringStrategy,
    OutlierRemovalStrategy,
    AnomalyDetectionStrategy,
    TextTransformationStrategy,
    TimeSeriesTransformationStrategy
)
from agentflow.transformations.pipeline import TransformationPipeline
from agentflow.core.config_manager import ConfigManager
from agentflow.core.workflow_executor import WorkflowExecutor
from agentflow.core.distributed_workflow import DistributedWorkflow

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
    """Base agent class with transformation capabilities."""

    def __init__(self, config: Union[Dict[str, Any], str, 'AgentConfig'], workflow_config: Optional[str] = None) -> None:
        """
        Initialize agent with configuration.

        Args:
            config: Agent configuration as dict, path to config file, or AgentConfig object
            workflow_config: Optional path to workflow configuration file
        """
        # Initialize logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Load and validate configuration
        if isinstance(config, str):
            self.config = self._load_config_from_file(config)
            if workflow_config:
                workflow_data = self._load_config_from_file(workflow_config)
                self.config.update(workflow_data)
        elif isinstance(config, dict):
            self.config = config
        else:
            # Save original config for accessing fields
            self._config_obj = config
            self.config = config.model_dump() if hasattr(config, 'model_dump') else config.dict()

        # Create and validate configuration schema
        self.config_schema = ConfigurationSchema.from_dict(self.config)
        if not self.config_schema.validate():
            raise ValueError("Invalid agent configuration")

        # Set basic attributes from schema
        self.name = self.config_schema.name
        self.type = self.config_schema.type
        self.max_iterations = self.config_schema.max_iterations
        
        # Set model configuration
        model_config = self.config.get('MODEL', {})
        self.model = SimpleNamespace(**model_config) if model_config else None

        # Set workflow configuration
        workflow_config = self.config.get('WORKFLOW', {})
        self.workflow = SimpleNamespace(**workflow_config) if workflow_config else None

        # Initialize state and history
        self.state = {}
        self.history = []
        self.errors = []
        self._workflow_history = []
        self._current_step = {}

        # Initialize instruction components
        self.isa_manager = ISAManager()
        self.instruction_selector = InstructionSelector()

        # Initialize transformation pipeline
        self.transformation_pipeline = None
        self._configure_transformations()

        # Initialize input pipeline
        self.input_pipeline = None

        # Initialize distributed attributes
        self.is_distributed = workflow_config.get('distributed', False)
        self.ray_actor = None
        if self.is_distributed:
            self._initialize_distributed()
            
        # Initialize metrics
        self.token_count = 0
        self.last_latency = 0
        self.memory_usage = 0
            
        # Initialize transformation functions
        self._input_transform = None
        self._output_transform = None
        
        # Initialize optimizer
        self.optimizer = None
        
        # Set initialization flag
        self._initialized = False

    def _load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {str(e)}")
            raise

    def _initialize_distributed(self) -> None:
        """Initialize distributed processing capabilities."""
        try:
            import ray
            if not ray.is_initialized():
                ray.init()
            self.ray_actor = ray.remote(self.__class__).remote(self.config)
        except ImportError:
            self.logger.warning("Ray not installed. Distributed processing disabled.")
            self.is_distributed = False

    def _validate_config(self) -> None:
        """Validate agent configuration."""
        # Basic validation is handled by ConfigurationSchema
        if not hasattr(self, 'config_schema') or not self.config_schema.validate():
            raise ValueError("Invalid agent configuration")

        # Additional validation for workflow settings
        workflow_config = self.config.get('WORKFLOW', {})
        max_iterations = workflow_config.get('max_iterations')
        if max_iterations is not None and max_iterations < 1:
            raise ValueError("Workflow max_iterations must be greater than or equal to 1")

    def _configure_transformations(self) -> None:
        """Configure transformation pipeline from config."""
        # Get transformations from schema
        transformations = self.config_schema.transformations

        # Configure input transformations
        for transform in transformations.get('input', []):
            strategy = self._create_transformation_strategy(
                transform['type'],
                **transform.get('params', {})
            )
            self.input_transformation_pipeline.add_strategy(strategy)

        # Configure preprocessing transformations
        for transform in transformations.get('preprocessing', []):
            strategy = self._create_transformation_strategy(
                transform['type'],
                **transform.get('params', {})
            )
            self.preprocessing_transformation_pipeline.add_strategy(strategy)

        # Configure output transformations
        for transform in transformations.get('output', []):
            strategy = self._create_transformation_strategy(
                transform['type'],
                **transform.get('params', {})
            )
            self.output_transformation_pipeline.add_strategy(strategy)

    def _create_transformation_strategy(self, strategy_type: str, **kwargs) -> AdvancedTransformationStrategy:
        """Create a transformation strategy based on type and parameters."""
        if strategy_type == 'feature_engineering':
            # Handle both 'strategy' and 'method' parameters for backward compatibility
            strategy = kwargs.pop('strategy', kwargs.pop('method', 'standard'))
            return FeatureEngineeringStrategy(strategy=strategy, **kwargs)
        
        elif strategy_type == 'outlier_removal':
            # Handle both 'method' and 'strategy' parameters
            method = kwargs.pop('method', kwargs.pop('strategy', 'z_score'))
            return OutlierRemovalStrategy(method=method, **kwargs)
        
        elif strategy_type == 'anomaly_detection':
            # Handle strategy parameter and detection_methods
            strategy = kwargs.pop('strategy', 'isolation_forest')
            detection_methods = kwargs.pop('detection_methods', None)
            return AnomalyDetectionStrategy(strategy=strategy, detection_methods=detection_methods, **kwargs)
        
        elif strategy_type == 'time_series':
            strategy = kwargs.pop('strategy', 'rolling_features')
            return TimeSeriesTransformationStrategy(strategy=strategy, **kwargs)
        
        elif strategy_type == 'text':
            method = kwargs.pop('method', kwargs.pop('strategy', 'basic'))
            return TextTransformationStrategy(method=method, **kwargs)
        
        else:
            raise ValueError(f"Unknown transformation strategy type: {strategy_type}")

    def transform(self, data: Any) -> Any:
        """Transform data using configured pipeline."""
        if not self.transformation_pipeline:
            return data

        try:
            return self.transformation_pipeline.transform(data)
        except Exception as e:
            self.logger.error(f"Failed to transform data: {str(e)}")
            return data

    def validate_config(self) -> None:
        """Validate agent configuration."""
        if not self.config:
            raise ValueError("Agent configuration is required")
        
        if self.workflow and self.workflow.max_iterations < 1:
            raise ValueError("Workflow max_iterations must be greater than or equal to 1")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, type={self.type})"

    async def initialize(self):
        """Initialize agent asynchronously"""
        if self._initialized:
            return self
        
        # Initialize base state
        self.id = self.config.get('AGENT', {}).get('id') or 'default_agent'
        self.type = self.config.get('AGENT', {}).get('type') or 'base'
        self.capabilities = []  # Initialize empty capabilities list
        
        # Initialize metrics
        self.token_count = 0
        self.last_latency = 0
        self.memory_usage = 0
        
        # Initialize transformation functions
        self._input_transform = None
        self._output_transform = None
        
        # Load strategies and instructions
        await self._load_strategies()
        await self._load_agent_instructions()
        
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
            self.logger.error(f"Error loading strategies: {e}")
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
            self.logger.error(f"Error loading instructions: {e}")
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
            self.logger.error(f"Error processing input: {e}")
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
            response = self._call_llm({"message": message})
            
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
    
    @workflow_step("process_message", "Process input message")
    def process_message(self, message: str) -> str:
        """Process input message using the agent's language model.

        Args:
            message: Input message to process

        Returns:
            str: Response from the language model

        Raises:
            RuntimeError: If message processing fails
            ValueError: If message is None
        """
        if message is None:
            error = ValueError("Message cannot be None")
            self._current_step["status"] = "error"
            self._current_step["error"] = str(error)
            self._workflow_history.append(self._current_step.copy())
            raise error

        try:
            # Call language model
            response = self._call_llm({"message": message})
            return response
        except Exception as e:
            # Record error in workflow history
            self._current_step["status"] = "error"
            self._current_step["error"] = str(e)
            self._workflow_history.append(self._current_step.copy())
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

    def _call_llm(self, input_data: Dict[str, Any]) -> str:
        """Call language model with input data.
        
        Args:
            input_data: Input data for the language model
            
        Returns:
            str: Language model response
            
        Raises:
            RuntimeError: If language model call fails
        """
        try:
            import openai
            import time
            
            # Format input for the model
            messages = [
                {"role": "system", "content": "You are a research assistant helping with academic research."},
                {"role": "user", "content": json.dumps(input_data)}
            ]
            
            # Record start time
            start_time = time.time()
            
            # Call OpenAI API
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model.name,
                messages=messages,
                temperature=getattr(self.model, 'temperature', 0.7)
            )
            
            # Update metrics
            self.last_latency = time.time() - start_time
            self.token_count += response.usage.total_tokens
            
            # Extract and return response text
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Language model call failed: {str(e)}")
            raise RuntimeError(f"Language model call failed: {str(e)}")

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
        
        # Reset pipelines
        if self.input_pipeline:
            await self.input_pipeline.cleanup()
        self.input_pipeline = None
        
        return self

    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get the history of workflow steps executed"""
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
            self.logger.error(f"Error in agent processing: {str(e)}")
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

    async def execute_workflow_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously.
        
        Args:
            input_data: Input data for the workflow
            
        Returns:
            Dict[str, Any]: Results from workflow execution
            
        Raises:
            ValueError: If input data is invalid
            WorkflowExecutionError: If workflow execution fails
        """
        # Initialize agent if not already initialized
        if not self._initialized:
            await self.initialize()
        
        # Validate input data
        if not input_data or not isinstance(input_data, dict):
            raise ValueError("Input data must be a non-empty dictionary")
        
        required_fields = {"research_topic", "deadline", "academic_level"}
        missing_fields = required_fields - set(input_data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        try:
            # Transform input data
            transformed_input = await self.transform_input(input_data)
            
            # Process input through workflow steps
            step_results = {}
            
            # Execute research step
            research_result = await self._execute_research_step(transformed_input)
            step_results['step_1'] = research_result
            
            # Execute document generation if enabled
            if hasattr(self.workflow, 'document_generation') and self.workflow.document_generation:
                doc_result = await self._execute_document_step(research_result)
                step_results['step_2'] = doc_result
            
            # Combine results
            final_results = {
                'research_output': research_result,
                **step_results
            }
            
            # Transform output
            return await self.transform_output(final_results)
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise WorkflowExecutionError(f"Workflow execution failed: {str(e)}")
    
    def execute_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow synchronously.
        
        Args:
            input_data: Input data for the workflow
            
        Returns:
            Dict[str, Any]: Results from workflow execution
            
        Raises:
            ValueError: If input data is invalid
            WorkflowExecutionError: If workflow execution fails
        """
        return asyncio.run(self.execute_workflow_async(input_data))
    
    async def _execute_research_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research step of the workflow.
        
        Args:
            input_data: Preprocessed input data
            
        Returns:
            Dict[str, Any]: Research step results
        """
        try:
            # Apply RL optimization if enabled
            if self.optimizer:
                optimized_input = self._apply_rl_optimization(input_data)
            else:
                optimized_input = input_data
            
            # Call language model
            llm_response = await self._call_llm(optimized_input)
            
            # Process response
            processed_result = {
                'result': llm_response,
                'input': input_data,
                'metadata': {
                    'tokens': self.token_count,
                    'latency': self.last_latency
                }
            }
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Research step failed: {str(e)}")
            raise WorkflowExecutionError(f"Research step failed: {str(e)}")
    
    async def _execute_document_step(self, research_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document generation step of the workflow.
        
        Args:
            research_result: Results from research step
            
        Returns:
            Dict[str, Any]: Document generation results
        """
        try:
            # Prepare input for document generation
            doc_input = {
                'research_result': research_result['result'],
                'format': getattr(self.workflow, 'document_format', 'academic'),
                'style': getattr(self.workflow, 'document_style', 'standard')
            }
            
            # Call language model for document generation
            llm_response = await self._call_llm(doc_input)
            
            # Process response
            processed_result = {
                'result': llm_response,
                'input': doc_input,
                'metadata': {
                    'tokens': self.token_count,
                    'latency': self.last_latency
                }
            }
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Document generation failed: {str(e)}")
            raise WorkflowExecutionError(f"Document generation failed: {str(e)}")

    def _execute_step(
        self, 
        step_config: Dict[str, Any], 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a generic workflow step.
        
        Args:
            step_config: Configuration for the current step
            input_data: Input data for the step
            
        Returns:
            Step execution result
        """
        # Placeholder implementation
        step_type = step_config.get('type')
        
        # Log the step execution
        self.logger.info(f"Executing step: {step_type}")
        
        # Return a mock result
        return {
            'status': 'success',
            'step_type': step_type,
            'details': f"Executed {step_type} step"
        }

class GenericAgent(Agent):
    """Generic Agent with flexible transformation capabilities."""

    def __init__(self, config: Union[Dict[str, Any], str, 'AgentConfig']) -> None:
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
            self.logger.error(f"Data transformation failed: {str(e)}")
            return data

class ResearchAgent(Agent):
    """Research Agent with text processing capabilities."""

    def __init__(self, config: Union[Dict[str, Any], str, 'AgentConfig']) -> None:
        """Initialize Research Agent."""
        super().__init__(config)
        
        # Initialize research-specific attributes
        research_config = self.config.get('RESEARCH', {})
        self.domains = research_config.get('domains', [])
        self.citation_style = research_config.get('citation_style', None)

        # Add default text transformation if none specified
        if not self.transformation_pipeline or not self.transformation_pipeline.strategies:
            self.transformation_pipeline = TransformationPipeline()
            self.transformation_pipeline.add_strategy(
                TextTransformationStrategy(strategy='clean')
            )

    def _create_transformation_strategy(self, strategy_type: str, **kwargs) -> AdvancedTransformationStrategy:
        """Create a transformation strategy based on type and parameters."""
        if strategy_type == 'feature_engineering':
            # Handle both 'strategy' and 'method' parameters for backward compatibility
            strategy = kwargs.pop('strategy', kwargs.pop('method', 'standard'))
            return FeatureEngineeringStrategy(strategy=strategy, **kwargs)
        
        elif strategy_type == 'outlier_removal':
            # Handle both 'method' and 'strategy' parameters
            method = kwargs.pop('method', kwargs.pop('strategy', 'z_score'))
            return OutlierRemovalStrategy(method=method, **kwargs)
        
        elif strategy_type == 'anomaly_detection':
            # Handle strategy parameter and detection_methods
            strategy = kwargs.pop('strategy', 'isolation_forest')
            detection_methods = kwargs.pop('detection_methods', None)
            return AnomalyDetectionStrategy(strategy=strategy, detection_methods=detection_methods, **kwargs)
        
        elif strategy_type == 'time_series':
            strategy = kwargs.pop('strategy', 'rolling_features')
            return TimeSeriesTransformationStrategy(strategy=strategy, **kwargs)
        
        elif strategy_type == 'text':
            method = kwargs.pop('method', kwargs.pop('strategy', 'basic'))
            return TextTransformationStrategy(method=method, **kwargs)
        
        else:
            raise ValueError(f"Unknown transformation strategy type: {strategy_type}")

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
            self.logger.error(f"Text analysis failed: {str(e)}")
            raise ValueError(f"Text analysis failed: {str(e)}")

    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract citations from text."""
        try:
            # Citation extraction logic would go here
            return []
        except Exception as e:
            self.logger.error(f"Citation extraction failed: {str(e)}")
            raise ValueError(f"Citation extraction failed: {str(e)}")

    def format_citations(self, citations: List[Dict[str, str]], style: str = None) -> List[str]:
        """Format citations according to specified style."""
        try:
            # Citation formatting logic would go here
            return []
        except Exception as e:
            self.logger.error(f"Citation formatting failed: {str(e)}")
            raise ValueError(f"Citation formatting failed: {str(e)}")

    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """Generate a summary of the text."""
        try:
            # Text summarization logic would go here
            return text[:max_length]
        except Exception as e:
            self.logger.error(f"Text summarization failed: {str(e)}")
            raise ValueError(f"Text summarization failed: {str(e)}")

    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract key terms from text."""
        try:
            # Keyword extraction logic would go here
            return []
        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {str(e)}")
            raise ValueError(f"Keyword extraction failed: {str(e)}")

    def categorize_text(self, text: str) -> List[str]:
        """Categorize text into research domains."""
        try:
            # Text categorization logic would go here
            return []
        except Exception as e:
            self.logger.error(f"Text categorization failed: {str(e)}")
            raise ValueError(f"Text categorization failed: {str(e)}")

    def generate_bibliography(self, citations: List[Dict[str, str]]) -> str:
        """Generate a bibliography from citations."""
        try:
            # Bibliography generation logic would go here
            return ""
        except Exception as e:
            self.logger.error(f"Bibliography generation failed: {str(e)}")
            raise ValueError(f"Bibliography generation failed: {str(e)}")

    def check_plagiarism(self, text: str) -> Dict[str, Any]:
        """Check text for potential plagiarism."""
        try:
            # Plagiarism checking logic would go here
            return {}
        except Exception as e:
            self.logger.error(f"Plagiarism check failed: {str(e)}")
            raise ValueError(f"Plagiarism check failed: {str(e)}")

    def find_related_research(self, text: str) -> List[Dict[str, str]]:
        """Find related research papers."""
        try:
            # Related research search logic would go here
            return []
        except Exception as e:
            self.logger.error(f"Related research search failed: {str(e)}")
            raise ValueError(f"Related research search failed: {str(e)}")

    def generate_research_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a research report from analysis results."""
        try:
            # Report generation logic would go here
            return ""
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise ValueError(f"Report generation failed: {str(e)}")

    def validate_citations(self, citations: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Validate citation information."""
        try:
            # Citation validation logic would go here
            return {}
        except Exception as e:
            self.logger.error(f"Citation validation failed: {str(e)}")
            raise ValueError(f"Citation validation failed: {str(e)}")

    def analyze_research_trends(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze research trends across multiple texts."""
        try:
            # Research trend analysis logic would go here
            return {}
        except Exception as e:
            self.logger.error(f"Research trend analysis failed: {str(e)}")
            raise ValueError(f"Research trend analysis failed: {str(e)}")

    def extract_methodology(self, text: str) -> Dict[str, Any]:
        """Extract research methodology information."""
        try:
            # Methodology extraction logic would go here
            return {}
        except Exception as e:
            self.logger.error(f"Methodology extraction failed: {str(e)}")
            raise ValueError(f"Methodology extraction failed: {str(e)}")

    def analyze_citations_network(self, citations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze citation network and relationships."""
        try:
            # Citation network analysis logic would go here
            return {}
        except Exception as e:
            self.logger.error(f"Citation network analysis failed: {str(e)}")
            raise ValueError(f"Citation network analysis failed: {str(e)}")

class ConfigurationType(str, Enum):
    """Enumeration of configuration types."""
    RESEARCH = "RESEARCH"
    DATA_SCIENCE = "DATA_SCIENCE"
    DOMAIN_SPECIFIC = "DOMAIN_SPECIFIC"
    TECHNICAL = "TECHNICAL"
    GENERIC = "GENERIC"
    CREATIVE = "CREATIVE"
    DEFAULT = "default"
    CUSTOM = "custom"
    DYNAMIC = "dynamic"

class ConfigurationSchema:
    """Schema for validating agent configurations."""

    def __init__(self, name: str = None, type: str = None, max_iterations: int = None, transformations: Dict[str, Any] = None):
        """
        Initialize configuration schema.

        Args:
            name: Agent name
            type: Agent type
            max_iterations: Maximum workflow iterations
            transformations: Transformation configurations
        """
        self.name = name
        self.type = type
        self.max_iterations = max_iterations
        self.transformations = transformations or {}
        self.domain_config = {}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ConfigurationSchema':
        """Create configuration schema from dictionary."""
        agent_config = config.get('AGENT', {})
        workflow_config = config.get('WORKFLOW', {})
        transformations = config.get('TRANSFORMATIONS', {})
        domain_config = config.get('DOMAIN_CONFIG', {})

        schema = cls(
            name=agent_config.get('name'),
            type=ConfigurationType(agent_config.get('type', 'GENERIC')),
            max_iterations=workflow_config.get('max_iterations'),
            transformations=transformations
        )
        schema.domain_config = domain_config
        return schema

    def validate(self) -> bool:
        """Validate configuration schema."""
        try:
            # Validate required fields
            if not self.name:
                return False
            
            # Validate type
            if self.type:
                try:
                    ConfigurationType(self.type)
                except ValueError:
                    return False

            # Validate max_iterations
            if self.max_iterations is not None and self.max_iterations < 0:
                return False

            # Validate transformations
            if not isinstance(self.transformations, dict):
                return False

            for pipeline in self.transformations.values():
                if not isinstance(pipeline, list):
                    return False
                
                for transform in pipeline:
                    if not isinstance(transform, dict):
                        return False
                    if 'type' not in transform or 'params' not in transform:
                        return False
                    if not isinstance(transform['params'], dict):
                        return False

            return True
        except Exception:
            return False

    def merge(self, other: 'ConfigurationSchema') -> 'ConfigurationSchema':
        """
        Merge another configuration schema into this one.
        
        Args:
            other: Configuration schema to merge

        Returns:
            New merged configuration schema
        """
        # Create new schema with base values
        merged = ConfigurationSchema(
            name=self.name,  # Keep original name
            type=self.type,  # Keep original type
            max_iterations=self.max_iterations or other.max_iterations,
            transformations={}
        )

        # Merge transformations
        for key in set(self.transformations.keys()) | set(other.transformations.keys()):
            merged.transformations[key] = (
                self.transformations.get(key, []) +
                other.transformations.get(key, [])
            )

        # Merge domain configs
        merged.domain_config = {
            **other.domain_config,
            **self.domain_config  # Original domain config takes precedence
        }

        return merged

    def export(self, format: str = 'dict') -> Union[Dict[str, Any], str]:
        """
        Export configuration schema.
        
        Args:
            format: Export format ('dict' or 'json')

        Returns:
            Configuration in requested format
        """
        config = {
            'AGENT': {
                'name': self.name,
                'type': self.type.value if isinstance(self.type, ConfigurationType) else self.type
            },
            'WORKFLOW': {
                'max_iterations': self.max_iterations
            },
            'TRANSFORMATIONS': self.transformations,
            'DOMAIN_CONFIG': self.domain_config
        }

        if format == 'json':
            return json.dumps(config)
        return config

class AgentFactory:
    """Factory class for creating different types of agents."""

    @classmethod
    def create_agent(cls, config: Union[Dict[str, Any], 'AgentConfig']) -> Agent:
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

        # Create appropriate agent type
        if agent_type == ConfigurationType.RESEARCH:
            return ResearchAgent(config_dict)
        elif agent_type == ConfigurationType.DATA_SCIENCE:
            return DataScienceAgent(config_dict)
        else:
            return GenericAgent(config_dict)

class DataScienceAgent(Agent):
    """Data Science Agent for handling data analysis and modeling tasks."""

    def __init__(self, config: Union[str, Dict[str, Any]]):
        """Initialize Data Science Agent."""
        # Set default attributes
        self.model_types = None
        self.evaluation_metrics = None
        
        # Initialize base agent
        super().__init__(config)

        # Configure data science specific settings
        if 'DATA_SCIENCE' in self.config:
            ds_config = self.config['DATA_SCIENCE']
            self.model_types = ds_config.get('model_types', [])
            self.evaluation_metrics = ds_config.get('evaluation_metrics', [])

    def _create_transformation_strategy(self, strategy_type: str, **kwargs) -> AdvancedTransformationStrategy:
        """Create a transformation strategy based on type and parameters."""
        if strategy_type == 'feature_engineering':
            return FeatureEngineeringStrategy(**kwargs)
        
        elif strategy_type == 'outlier_removal':
            return OutlierRemovalStrategy(**kwargs)
        
        elif strategy_type == 'anomaly_detection':
            return AnomalyDetectionStrategy(**kwargs)
        
        elif strategy_type == 'time_series':
            return TimeSeriesTransformationStrategy(**kwargs)
        
        elif strategy_type == 'text':
            return TextTransformationStrategy(**kwargs)
        
        else:
            raise ValueError(f"Unknown transformation strategy type: {strategy_type}")

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
            self.logger.error(f"Data analysis failed: {str(e)}")
            raise ValueError(f"Data analysis failed: {str(e)}")

    def train_model(self, data: pd.DataFrame, target: str, model_type: str = 'auto') -> Any:
        """Train a machine learning model on the data."""
        try:
            # Model training logic would go here
            pass
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise ValueError(f"Model training failed: {str(e)}")

    def evaluate_model(self, model: Any, data: pd.DataFrame, target: str) -> Dict[str, float]:
        """Evaluate a trained model using specified metrics."""
        try:
            # Model evaluation logic would go here
            pass
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise ValueError(f"Model evaluation failed: {str(e)}")

    def predict(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        try:
            # Prediction logic would go here
            pass
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def save_model(self, model: Any, path: str) -> None:
        """Save a trained model to disk."""
        try:
            # Model saving logic would go here
            pass
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise ValueError(f"Model saving failed: {str(e)}")

    def load_model(self, path: str) -> Any:
        """Load a trained model from disk."""
        try:
            # Model loading logic would go here
            pass
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise ValueError(f"Model loading failed: {str(e)}")

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a report from analysis results."""
        try:
            # Report generation logic would go here
            pass
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise ValueError(f"Report generation failed: {str(e)}")

    def optimize_hyperparameters(self, model: Any, data: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Optimize model hyperparameters."""
        try:
            # Hyperparameter optimization logic would go here
            pass
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {str(e)}")
            raise ValueError(f"Hyperparameter optimization failed: {str(e)}")

    def feature_importance(self, model: Any, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance scores."""
        try:
            # Feature importance calculation logic would go here
            pass
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {str(e)}")
            raise ValueError(f"Feature importance calculation failed: {str(e)}")

    def cross_validate(self, model: Any, data: pd.DataFrame, target: str, cv: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        try:
            # Cross-validation logic would go here
            pass
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {str(e)}")
            raise ValueError(f"Cross-validation failed: {str(e)}")

    def explain_predictions(self, model: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate explanations for model predictions."""
        try:
            # Model explanation logic would go here
            pass
        except Exception as e:
            self.logger.error(f"Prediction explanation failed: {str(e)}")
            raise ValueError(f"Prediction explanation failed: {str(e)}")

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
            # Load from file
            with open(source, 'r') as f:
                config_data = json.load(f)
        elif isinstance(source, dict):
            config_data = source
        elif isinstance(source, ConfigurationSchema):
            return source
        else:
            raise ValueError("Invalid configuration source")

        # Create and validate schema
        schema = ConfigurationSchema.from_dict(config_data)
        if not schema.validate():
            raise ValueError("Invalid configuration data")
        return schema

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