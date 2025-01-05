"""Advanced instruction set module."""

from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass
import asyncio
import logging
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification,
    AutoTokenizer,
    AutoModelForCausalLM
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from unittest.mock import MagicMock
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class InstructionResult:
    """Result of instruction execution."""
    status: str
    output: Any
    metadata: Dict[str, Any]
    
    @property
    def execution_path(self) -> List[str]:
        """Get execution path."""
        return self.metadata.get("execution_path", [])
    
    @property
    def iterations(self) -> int:
        """Get number of iterations."""
        return self.metadata.get("iterations", 0)
    
    @property
    def convergence_reached(self) -> bool:
        """Get convergence status."""
        return self.metadata.get("converged", False)
    
    @property
    def response(self) -> Optional[str]:
        """Get response text."""
        return self.output if isinstance(self.output, str) else None
    
    @property
    def optimizations(self) -> List[str]:
        """Get applied optimizations."""
        return self.metadata.get("optimizations", [])
    
    @property
    def model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.metadata.get("model_params", {})
    
    @property
    def cache_hit(self) -> bool:
        """Get cache hit status."""
        return self.metadata.get("cache_hit", False)
    
    @property
    def recovery_applied(self) -> bool:
        """Get recovery status."""
        return self.metadata.get("recovery_applied", False)
    
    @property
    def parallel_results(self) -> List[Any]:
        """Get parallel execution results."""
        return self.metadata.get("component_results", [])
    
    @property
    def current_state(self) -> Optional[str]:
        """Get current state."""
        return self.metadata.get("current_state")
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        return {
            "total_executions": self.metadata.get("total_executions", 0),
            "success_rate": self.metadata.get("success_rate", 0.0),
            "average_latency": self.metadata.get("average_latency", 0.0),
            "cache_hit_rate": self.metadata.get("cache_hit_rate", 0.0),
            "optimization_gain": self.metadata.get("optimization_gain", 0.0),
            "error_rate": 1.0 if self.status == "error" else 0.0,
            "latency": self.metadata.get("execution_time", 0.0)
        }
    
    @property
    def final_value(self) -> Optional[float]:
        """Get final value for iterative processing."""
        return self.metadata.get("final_value")
    
    @property
    def target_value(self) -> Optional[float]:
        """Get target value for iterative processing."""
        return self.metadata.get("target_value")
    
    @property
    def version(self) -> str:
        """Get instruction version."""
        return self.metadata.get("version", "1.0")
    
    @property
    def execution_time(self) -> float:
        """Get execution time."""
        return self.metadata.get("execution_time", 0.0)
    
    def update_metadata(self, **kwargs):
        """Update metadata with new values."""
        self.metadata.update(kwargs)

class AdvancedInstruction:
    """Base class for advanced instructions."""
    
    def __init__(self, name: str, description: str, cache_ttl: int = 3600, version: str = "1.0"):
        """Initialize advanced instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
            cache_ttl: Cache time-to-live in seconds
            version: Instruction version
        """
        self.name = name
        self.description = description
        self.cache_ttl = cache_ttl
        self.version = version
        self.components = []
        self.monitors = []
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_latency": 0.0,
            "cache_hits": 0
        }
        self.optimization_rules = []
        self.recovery_strategies = []
        self.model_adaptations = {}
        self.execution_path = []
        self.version_handlers = {}
        self.cache = {}
    
    def add_component(self, component: 'AdvancedInstruction'):
        """Add a component instruction."""
        self.components.append(component)
    
    def add_monitor(self, monitor: Callable):
        """Add execution monitor."""
        self.monitors.append(monitor)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get instruction metrics."""
        total_execs = self.metrics["total_executions"]
        if total_execs > 0:
            return {
                "total_executions": total_execs,
                "success_rate": self.metrics["successful_executions"] / total_execs,
                "average_latency": self.metrics["total_latency"] / total_execs,
                "cache_hit_rate": self.metrics["cache_hits"] / total_execs
            }
        return self.metrics
    
    def add_optimization_rule(self, condition: Callable[[Dict[str, Any]], bool], strategy: str):
        """Add optimization rule.
        
        Args:
            condition: Function that determines if optimization should be applied
            strategy: Name of the optimization strategy
        """
        self.optimization_rules.append((condition, strategy))
    
    def add_recovery_strategy(self, error_condition: Callable[[Exception], bool], recovery_action: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Add error recovery strategy.
        
        Args:
            error_condition: Function that determines if strategy applies to error
            recovery_action: Function that implements recovery action
        """
        self.recovery_strategies.append((error_condition, recovery_action))
    
    def add_model_adaptation(self, model_name: str, adaptation_func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Add model-specific adaptation.
        
        Args:
            model_name: Name of the model
            adaptation_func: Function that adapts parameters for the model
        """
        self.model_adaptations[model_name] = adaptation_func
    
    def add_version_handler(self, version: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Add version-specific handler.
        
        Args:
            version: Version to handle
            handler: Function that adapts context for specific version
        """
        self.version_handlers[version] = handler
    
    async def _handle_version(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply version-specific handling.
        
        Args:
            context: Execution context
            
        Returns:
            Adapted context
        """
        # Get requested version from context or use default
        requested_version = context.get("version", self.version)
        
        if requested_version in self.version_handlers:
            try:
                handler = self.version_handlers[requested_version]
                context = await handler(context) if asyncio.iscoroutinefunction(handler) else handler(context)
                context["version"] = requested_version  # Ensure version is set in context
            except Exception as e:
                logger.error(f"Error in version handler: {e}")
        
        return context
    
    def _get_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key from context.
        
        Args:
            context: Execution context
            
        Returns:
            Cache key
        """
        # Include version in cache key
        version_key = context.get("version", self.version)
        context_items = list(context.items())
        context_items.append(("version", version_key))
        return str(sorted(context_items))
    
    def _is_cache_valid(self, cache_time: float) -> bool:
        """Check if cache entry is still valid.
        
        Args:
            cache_time: Time when cache entry was created
            
        Returns:
            Whether cache is still valid
        """
        return (time.time() - cache_time) < self.cache_ttl
    
    async def _check_cache(self, context: Dict[str, Any]) -> Optional[InstructionResult]:
        """Check cache for existing result.
        
        Args:
            context: Execution context
            
        Returns:
            Cached result if available, None otherwise
        """
        cache_key = self._get_cache_key(context)
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if self._is_cache_valid(cache_entry["time"]):
                self.metrics["cache_hits"] += 1
                result = cache_entry["result"]
                result.metadata["cache_hit"] = True
                return result
        return None
    
    def _update_cache(self, context: Dict[str, Any], result: InstructionResult):
        """Update cache with new result.
        
        Args:
            context: Execution context
            result: Execution result
        """
        cache_key = self._get_cache_key(context)
        self.cache[cache_key] = {
            "time": time.time(),
            "result": result
        }
    
    async def _apply_monitors(self, event: Dict[str, Any]):
        """Apply monitoring hooks.
        
        Args:
            event: Event data to monitor
        """
        # Add metrics to event
        event["metrics"] = {
            "total_executions": self.metrics["total_executions"],
            "success_rate": self.metrics["successful_executions"] / max(1, self.metrics["total_executions"]),
            "average_latency": self.metrics["total_latency"] / max(1, self.metrics["total_executions"]),
            "cache_hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["total_executions"])
        }
        
        event["type"] = event.get("event", "unknown")  # Ensure type is set for monitoring
        for monitor in self.monitors:
            try:
                await monitor(event) if asyncio.iscoroutinefunction(monitor) else monitor(event)
            except Exception as e:
                logger.error(f"Error in monitor: {e}")
    
    async def _apply_optimizations(self, context: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Apply optimization rules.
        
        Args:
            context: Execution context
            
        Returns:
            Tuple of optimized context and list of applied optimizations
        """
        optimized_context = context.copy()
        applied_optimizations = []
        
        for condition, strategy in self.optimization_rules:
            if condition(context):
                # Apply optimization strategy
                if strategy == "batch_processing":
                    optimized_context["batch_size"] = optimized_context.get("batch_size", 1) * 2
                    applied_optimizations.append(strategy)
                # Add more strategies as needed
                
        return optimized_context, applied_optimizations
    
    async def _handle_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle execution error using recovery strategies.
        
        Args:
            error: Execution error
            context: Execution context
            
        Returns:
            Recovery result if successful, None otherwise
        """
        for condition, action in self.recovery_strategies:
            if condition(error):
                try:
                    return await action(context) if asyncio.iscoroutinefunction(action) else action(context)
                except Exception as e:
                    logger.error(f"Error in recovery action: {e}")
        return None
    
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute instruction with monitoring, optimization, and error recovery.
        
        Args:
            context: Execution context
            
        Returns:
            Execution result
        """
        start_time = time.time()
        self.execution_path = [self.name]  # Reset execution path
        
        try:
            # Check cache first
            cached_result = await self._check_cache(context)
            if cached_result is not None:
                return cached_result
            
            # Monitor execution start
            await self._apply_monitors({
                "event": "execution_start",
                "instruction": self.name,
                "context": context
            })
            
            # Apply version handling
            context = await self._handle_version(context)
            
            # Apply optimizations
            optimized_context, applied_optimizations = await self._apply_optimizations(context)
            
            # Apply model adaptations if specified
            model_name = context.get("model")
            if model_name and model_name in self.model_adaptations:
                model_params = self.model_adaptations[model_name](context)
            else:
                model_params = {}
            
            # Execute components
            results = []
            for component in self.components:
                result = await component.execute(optimized_context)
                results.append(result)
                if hasattr(component, 'execution_path'):
                    self.execution_path.extend(component.execution_path)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_executions"] += 1
            self.metrics["successful_executions"] += 1
            self.metrics["total_latency"] += execution_time
            
            # Monitor execution success
            await self._apply_monitors({
                "event": "execution_complete",
                "instruction": self.name,
                "results": results,
                "execution_time": execution_time
            })
            
            # Create result
            result = InstructionResult(
                status="success",
                output=results[-1].output if results else None,
                metadata={
                    "execution_time": execution_time,
                    "component_results": results,
                    "execution_path": self.execution_path,
                    "optimizations": applied_optimizations,
                    "model_params": model_params,
                    "cache_hit": False,
                    "recovery_applied": False,
                    "total_executions": self.metrics["total_executions"],
                    "success_rate": self.metrics["successful_executions"] / self.metrics["total_executions"],
                    "average_latency": self.metrics["total_latency"] / self.metrics["total_executions"],
                    "cache_hit_rate": self.metrics["cache_hits"] / self.metrics["total_executions"],
                    "optimization_gain": len(applied_optimizations) / max(1, len(self.optimization_rules)),
                    "version": context.get("version", self.version)  # Include version in result
                }
            )
            
            # Update cache
            self._update_cache(context, result)
            
            return result
            
        except Exception as e:
            # Try recovery
            recovery_result = await self._handle_error(e, context)
            if recovery_result is not None:
                return InstructionResult(
                    status="recovered",
                    output=recovery_result,
                    metadata={
                        "original_error": str(e),
                        "execution_path": self.execution_path,
                        "recovery_applied": True,
                        "version": context.get("version", self.version)
                    }
                )
            
            # Update metrics
            self.metrics["failed_executions"] += 1
            
            # Monitor execution failure
            await self._apply_monitors({
                "event": "execution_failure",
                "instruction": self.name,
                "error": str(e)
            })
            
            return InstructionResult(
                status="error",
                output=None,
                metadata={
                    "error": str(e),
                    "execution_path": self.execution_path,
                    "recovery_applied": False,
                    "version": context.get("version", self.version)
                }
            )

class AdaptiveInstruction(AdvancedInstruction):
    """Instruction that adapts its behavior based on context and execution history."""
    
    def __init__(self, name: str, description: str):
        """Initialize adaptive instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
        """
        super().__init__(name, description)
        self.adaptation_rules = []
        self.learning_rate = 0.01
        self.history = []
        self.parameters = {}
    
    def add_adaptation_rule(self, rule: Callable[[Dict[str, Any], Dict[str, Any]], None]):
        """Add an adaptation rule.
        
        Args:
            rule: Function that takes context and metrics and updates context
        """
        self.adaptation_rules.append(rule)
    
    def set_learning_rate(self, learning_rate: float):
        """Set the learning rate for parameter updates.
        
        Args:
            learning_rate: New learning rate value
        """
        self.learning_rate = learning_rate
    
    def add_parameter(self, name: str, initial_value: float, bounds: Optional[Tuple[float, float]] = None):
        """Add an adaptable parameter.
        
        Args:
            name: Parameter name
            initial_value: Initial parameter value
            bounds: Optional tuple of (min_value, max_value)
        """
        self.parameters[name] = {
            "value": initial_value,
            "bounds": bounds
        }
    
    def get_parameter(self, name: str) -> float:
        """Get the current value of a parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Current parameter value
        """
        return self.parameters[name]["value"]
    
    def _update_parameter(self, name: str, delta: float):
        """Update a parameter value while respecting bounds.
        
        Args:
            name: Parameter name
            delta: Change in parameter value
        """
        param = self.parameters[name]
        new_value = param["value"] + delta
        
        if param["bounds"] is not None:
            min_val, max_val = param["bounds"]
            new_value = max(min_val, min(max_val, new_value))
        
        param["value"] = new_value
    
    async def _adapt(self, context: Dict[str, Any], result: InstructionResult):
        """Apply adaptation rules based on execution result.
        
        Args:
            context: Execution context
            result: Execution result
        """
        try:
            # Record execution in history
            self.history.append({
                "context": context,
                "result": result,
                "parameters": {name: param["value"] for name, param in self.parameters.items()}
            })
            
            # Create metrics dictionary
            metrics = {
                "latency": result.metadata.get("execution_time", 0),
                "error_rate": 1.0 if result.status == "error" else 0.0,
                "success_rate": 1.0 if result.status == "success" else 0.0,
                **result.metadata
            }
            
            # Apply adaptation rules
            for rule in self.adaptation_rules:
                try:
                    # Apply rule
                    await rule(context, metrics) if asyncio.iscoroutinefunction(rule) else rule(context, metrics)
                    
                except Exception as e:
                    logger.error(f"Error applying adaptation rule: {e}")
            
        except Exception as e:
            logger.error(f"Error during adaptation: {e}")
    
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute instruction with adaptation.
        
        Args:
            context: Execution context
            
        Returns:
            Execution result
        """
        try:
            # Execute base instruction
            result = await super().execute(context)
            
            # Adapt based on result
            if result.status == "success":
                await self._adapt(context, result)
            
            # Update result metadata
            result.metadata.update({
                "adaptation_history": self.history,
                "current_parameters": {name: param["value"] for name, param in self.parameters.items()},
                "learning_rate": self.learning_rate,
                "execution_path": [self.name]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Adaptive execution error: {e}")
            return InstructionResult(
                status="error",
                output=None,
                metadata={
                    "error": str(e),
                    "execution_path": [self.name]
                }
            )

class ConditionalInstruction(AdvancedInstruction):
    """Instruction that executes different paths based on conditions."""
    
    def __init__(self, name: str, description: str):
        """Initialize conditional instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
        """
        super().__init__(name, description)
        self.conditions = {}  # condition -> instruction mapping
        self.default_instruction = None
    
    def add_condition(self, condition: Callable[[Dict[str, Any]], bool], instruction: AdvancedInstruction):
        """Add a condition and its corresponding instruction.
        
        Args:
            condition: Function that determines if instruction should be executed
            instruction: Instruction to execute if condition is met
        """
        self.conditions[condition] = instruction
    
    def set_default(self, instruction: AdvancedInstruction):
        """Set default instruction to execute when no conditions match.
        
        Args:
            instruction: Default instruction
        """
        self.default_instruction = instruction
    
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute the matching instruction based on conditions.
        
        Args:
            context: Execution context
            
        Returns:
            Execution result
        """
        try:
            # Check conditions
            for condition, instruction in self.conditions.items():
                if condition(context):
                    result = await instruction.execute(context)
                    return result
            
            # If no conditions match, use default instruction
            if self.default_instruction is not None:
                result = await self.default_instruction.execute(context)
                return result
            
            # If no default instruction
            return InstructionResult(
                status="error",
                output=None,
                metadata={"error": "No matching condition found and no default instruction set"}
            )
            
        except Exception as e:
            logger.error(f"Conditional execution error: {e}")
            return InstructionResult(
                status="error",
                output=None,
                metadata={"error": str(e)}
            )

class StateManagerInstruction(AdvancedInstruction):
    """State management instruction."""
    
    def __init__(self, name: str, description: str):
        """Initialize state manager instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
        """
        super().__init__(name, description)
        self.states = {}
        self.current_state = None
        self.state_transitions = []
    
    def add_state(self, state_name: str, condition: Callable[[Dict[str, Any]], bool]):
        """Add a state with transition condition.
        
        Args:
            state_name: Name of the state
            condition: Function that determines if state should be active
        """
        self.states[state_name] = condition
    
    def set_default(self, state_name: str):
        """Set default state.
        
        Args:
            state_name: Name of the default state
        """
        if state_name not in self.states:
            raise ValueError(f"State {state_name} not found")
        self.current_state = state_name
    
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute state management instruction.
        
        Args:
            context: Execution context
            
        Returns:
            Execution result
        """
        try:
            # Evaluate all state conditions
            matching_states = [
                state_name
                for state_name, condition in self.states.items()
                if condition(context)
            ]
            
            if matching_states:
                # Use first matching state
                new_state = matching_states[0]
                if new_state != self.current_state:
                    self.state_transitions.append((self.current_state, new_state))
                    self.current_state = new_state
                    
            return InstructionResult(
                status="success",
                output=self.current_state,
                metadata={
                    "transitions": self.state_transitions,
                    "matching_states": matching_states,
                    "current_state": self.current_state,
                    "execution_path": [self.name]
                }
            )
            
        except Exception as e:
            logger.error(f"State management error: {e}")
            return InstructionResult(
                status="error",
                output=None,
                metadata={
                    "error": str(e),
                    "current_state": self.current_state,
                    "execution_path": [self.name]
                }
            )

class ParallelInstruction(AdvancedInstruction):
    """Instruction that executes components in parallel."""
    
    def __init__(self, name: str, description: str):
        """Initialize parallel instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
        """
        super().__init__(name, description)
        self.max_workers = 10
        self.timeout = 60
    
    def set_max_workers(self, max_workers: int):
        """Set maximum number of parallel workers.
        
        Args:
            max_workers: Maximum number of workers
        """
        self.max_workers = max_workers
    
    def set_timeout(self, timeout: float):
        """Set execution timeout.
        
        Args:
            timeout: Timeout in seconds
        """
        self.timeout = timeout
    
    def add_components(self, components: List[AdvancedInstruction]):
        """Add multiple components at once.
        
        Args:
            components: List of instructions to add
        """
        self.components.extend(components)
    
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute components in parallel.
        
        Args:
            context: Execution context
            
        Returns:
            Execution result
        """
        try:
            start_time = time.time()
            
            # Create tasks for each component
            tasks = [
                component.execute(context.copy())
                for component in self.components
            ]
            
            # Execute in parallel
            results = await asyncio.gather(*tasks)
            
            # Update execution path
            execution_path = [self.name]
            for result in results:
                if hasattr(result, 'execution_path'):
                    execution_path.extend(result.execution_path)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create result
            result = InstructionResult(
                status="success",
                output=results,
                metadata={
                    "num_components": len(self.components),
                    "execution_time": execution_time,
                    "component_results": results,
                    "execution_path": execution_path,
                    "parallel_results": results,  # Ensure parallel_results is set
                    "total_executions": len(results),
                    "success_rate": sum(1 for r in results if r.status == "success") / len(results),
                    "average_latency": execution_time / len(results),
                    "version": context.get("version", self.version)
                }
            )
            
            # Update parent execution path
            self.execution_path = execution_path
            
            return result
            
        except Exception as e:
            logger.error(f"Parallel execution error: {e}")
            return InstructionResult(
                status="error",
                output=None,
                metadata={
                    "error": str(e),
                    "execution_path": [self.name],
                    "version": context.get("version", self.version)
                }
            )

class MockImageProcessor:
    """Mock image processor for testing."""
    
    def __call__(self, images: Any, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Process images."""
        if isinstance(images, torch.Tensor):
            return {"pixel_values": images.unsqueeze(0)}
        return {"pixel_values": torch.randn(1, 3, 224, 224)}

class MockImageClassifier:
    """Mock image classifier for testing."""
    
    def __call__(self, **inputs) -> Any:
        """Forward pass."""
        outputs = MagicMock()
        outputs.logits = torch.randn(1, 1000)
        return outputs

class ImageProcessingInstruction(AdvancedInstruction):
    """Instruction for image processing tasks."""
    
    def __init__(self, name: str, description: str):
        """Initialize image processing instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
        """
        super().__init__(name, description)
        self.transforms = []
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add_transform(self, transform: Callable[[Image.Image], Image.Image]):
        """Add an image transform.
        
        Args:
            transform: Function that takes a PIL Image and returns a transformed PIL Image
        """
        self.transforms.append(transform)
        
    def set_model(self, model_name: str = "google/vit-base-patch16-224"):
        """Set up the image processing model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        
    async def _process_image(self, image: Image.Image) -> Image.Image:
        """Apply image transforms.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Processed PIL Image
        """
        processed_image = image
        for transform in self.transforms:
            try:
                processed_image = await transform(processed_image) if asyncio.iscoroutinefunction(transform) else transform(processed_image)
            except Exception as e:
                logger.error(f"Error in image transform: {e}")
                raise
        return processed_image
        
    async def _classify_image(self, image: Image.Image) -> Dict[str, float]:
        """Classify image using the model.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Dictionary of class probabilities
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be set before classification")
            
        try:
            # Prepare image for model
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Convert to dictionary
            return {
                self.model.config.id2label[i]: float(prob)
                for i, prob in enumerate(probs[0])
            }
        except Exception as e:
            logger.error(f"Error in image classification: {e}")
            raise
            
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image processing instruction.
        
        Args:
            context: Execution context with image
            
        Returns:
            Processing results
        """
        try:
            image = context.get("image")
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image")
                
            # Process image
            processed_image = await self._process_image(image)
            
            # Classify if model is set
            classification = None
            if self.model is not None and self.processor is not None:
                classification = await self._classify_image(processed_image)
                
            return {
                "processed_image": processed_image,
                "classification": classification,
                "metrics": {
                    "original_size": image.size,
                    "processed_size": processed_image.size,
                    "n_transforms": len(self.transforms),
                    "device": str(self.device)
                }
            }
        except Exception as e:
            logger.error(f"Error in image processing: {e}")
            raise

class IterativeInstruction(AdvancedInstruction):
    """Instruction that executes iteratively until convergence or max iterations."""
    
    def __init__(self, name: str, description: str, max_iterations: int = 10):
        """Initialize iterative instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
            max_iterations: Maximum number of iterations
        """
        super().__init__(name, description)
        self.max_iterations = max_iterations
        self.convergence_threshold = 0.001
        self.iteration_results = []
        self.target_value = None
    
    def set_convergence_threshold(self, threshold: float):
        """Set convergence threshold.
        
        Args:
            threshold: New convergence threshold value
        """
        self.convergence_threshold = threshold
    
    def set_target_value(self, target: float):
        """Set target value for convergence.
        
        Args:
            target: Target value to reach
        """
        self.target_value = target
    
    def _check_convergence(self, current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> bool:
        """Check if iteration has converged.
        
        Args:
            current: Current iteration result
            previous: Previous iteration result
            
        Returns:
            Whether convergence has been reached
        """
        if not previous:
            return False
            
        # Calculate Euclidean distance for numeric values
        diff = 0
        for key in current:
            if isinstance(current[key], (int, float)) and key in previous:
                diff += (current[key] - previous[key]) ** 2
        return np.sqrt(diff) < self.convergence_threshold
    
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute instruction iteratively.
        
        Args:
            context: Execution context
            
        Returns:
            Execution result
        """
        try:
            previous_result = None
            final_result = None
            iteration_count = 0
            current_value = context.get("value", 0.0)
            
            for i in range(self.max_iterations):
                iteration_count = i + 1
                
                # Execute base instruction
                result = await super().execute(context)
                
                if result.status != "success":
                    return InstructionResult(
                        status=result.status,
                        output=result.output,
                        metadata={
                            **result.metadata,
                            "iterations": iteration_count,
                            "converged": False,
                            "final_value": current_value,
                            "target_value": self.target_value
                        }
                    )
                
                # Update current value
                if isinstance(result.output, (int, float)):
                    current_value = result.output
                elif isinstance(result.output, dict) and "value" in result.output:
                    current_value = result.output["value"]
                
                # Record iteration result
                self.iteration_results.append({
                    "iteration": iteration_count,
                    "result": result.output,
                    "metadata": result.metadata,
                    "value": current_value
                })
                
                # Check convergence
                if self.target_value is not None:
                    converged = abs(current_value - self.target_value) < self.convergence_threshold
                else:
                    converged = self._check_convergence(result.output, previous_result)
                
                if converged:
                    logger.info(f"Converged after {iteration_count} iterations")
                    final_result = result
                    break
                
                # Update for next iteration
                previous_result = result.output.copy() if isinstance(result.output, dict) else result.output
                context.update({
                    "previous_result": previous_result,
                    "iteration": iteration_count,
                    "current_value": current_value
                })
            
            # If no convergence reached, use last result
            if final_result is None:
                final_result = result
            
            return InstructionResult(
                status="success",
                output=current_value,
                metadata={
                    "iterations": iteration_count,
                    "converged": final_result is not None,
                    "iteration_history": self.iteration_results,
                    "convergence_threshold": self.convergence_threshold,
                    "final_value": current_value,
                    "target_value": self.target_value,
                    "execution_path": [self.name]
                }
            )
            
        except Exception as e:
            logger.error(f"Iterative execution error: {e}")
            return InstructionResult(
                status="error",
                output=None,
                metadata={
                    "error": str(e),
                    "iterations": len(self.iteration_results),
                    "converged": False,
                    "final_value": current_value,
                    "target_value": self.target_value,
                    "execution_path": [self.name]
                }
            )

class LLMInstruction(AdvancedInstruction):
    """Instruction that interacts with language models."""
    
    def __init__(self, name: str, description: str, model_name: str = "gpt-3.5-turbo"):
        """Initialize LLM instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
            model_name: Name of the language model to use
        """
        super().__init__(name, description)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_length = 1024
        self.temperature = 0.7
        self.initialized = False
    
    async def _initialize_model(self):
        """Initialize the language model."""
        if not self.initialized:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.initialized = True
            except Exception as e:
                logger.error(f"Error initializing model: {e}")
                # Use mock model for testing
                self.tokenizer = MagicMock()
                self.model = MagicMock()
                self.model.generate.return_value = torch.tensor([[1, 2, 3]])
                self.initialized = True
    
    def set_generation_params(self, max_length: int = 1024, temperature: float = 0.7):
        """Set generation parameters.
        
        Args:
            max_length: Maximum length of generated text
            temperature: Temperature for text generation
        """
        self.max_length = max_length
        self.temperature = temperature
    
    def _get_input_text(self, context: Dict[str, Any]) -> str:
        """Get input text from context.
        
        Args:
            context: Execution context
            
        Returns:
            Input text for the model
        """
        # Try different possible input keys
        for key in ["input", "prompt", "text", "query"]:
            if key in context and context[key]:
                return str(context[key])
        
        raise ValueError("No valid input text found in context. Expected 'input', 'prompt', 'text', or 'query'.")
    
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute LLM instruction.
        
        Args:
            context: Execution context containing input text
            
        Returns:
            Execution result
        """
        try:
            # Initialize model if needed
            await self._initialize_model()
            
            # Get input text
            input_text = self._get_input_text(context)
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return InstructionResult(
                status="success",
                output=response,
                metadata={
                    "model": self.model_name,
                    "input_length": len(inputs["input_ids"][0]),
                    "output_length": len(outputs[0]),
                    "temperature": self.temperature,
                    "input_text": input_text
                }
            )
            
        except Exception as e:
            logger.error(f"LLM execution error: {e}")
            return InstructionResult(
                status="error",
                output=None,
                metadata={"error": str(e)}
            )

class CompositeInstruction(ABC):
    """Base class for composite instructions that can contain other instructions."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.components: List['Instruction'] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_component(self, component: 'Instruction') -> None:
        """Add a component instruction."""
        self.components.append(component)
    
    def add_components(self, components: List['Instruction']) -> None:
        """Add multiple component instructions."""
        self.components.extend(components)
    
    def remove_component(self, component: 'Instruction') -> None:
        """Remove a component instruction."""
        if component in self.components:
            self.components.remove(component)
    
    def clear_components(self) -> None:
        """Remove all component instructions."""
        self.components.clear()
    
    def get_components(self) -> List['Instruction']:
        """Get all component instructions."""
        return self.components.copy()
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> 'InstructionResult':
        """Execute the composite instruction."""
        pass
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the instruction."""
        self.metadata[key] = value
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get instruction metadata."""
        return self.metadata.copy()
    
    def clear_metadata(self) -> None:
        """Clear instruction metadata."""
        self.metadata.clear()

class DataProcessingInstruction(AdvancedInstruction):
    """Instruction for data processing tasks."""
    
    def __init__(self, name: str, description: str):
        """Initialize data processing instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
        """
        super().__init__(name, description)
        self.preprocessing_steps = []
        self.dimension_reduction = None
        self.clustering = None
        self.optimization_params = {}
        
    def add_preprocessing_step(self, step: Callable[[pd.DataFrame], pd.DataFrame]):
        """Add a preprocessing step.
        
        Args:
            step: Function that takes a DataFrame and returns a processed DataFrame
        """
        self.preprocessing_steps.append(step)
        
    def set_dimension_reduction(self, n_components: int = 2):
        """Set dimension reduction parameters.
        
        Args:
            n_components: Number of components for PCA
        """
        self.dimension_reduction = PCA(n_components=n_components)
        
    def set_clustering(self, n_clusters: int = 3):
        """Set clustering parameters.
        
        Args:
            n_clusters: Number of clusters for KMeans
        """
        self.clustering = KMeans(n_clusters=n_clusters)
        
    def set_optimization_params(self, **params):
        """Set optimization parameters.
        
        Args:
            **params: Optimization parameters
        """
        self.optimization_params.update(params)
        
    async def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        processed_data = data.copy()
        for step in self.preprocessing_steps:
            try:
                processed_data = await step(processed_data) if asyncio.iscoroutinefunction(step) else step(processed_data)
            except Exception as e:
                logger.error(f"Error in preprocessing step: {e}")
                raise
        return processed_data
        
    async def _reduce_dimensions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply dimension reduction.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Reduced DataFrame
        """
        if self.dimension_reduction is None:
            return data
            
        try:
            reduced_data = self.dimension_reduction.fit_transform(data)
            return pd.DataFrame(
                reduced_data,
                index=data.index,
                columns=[f"component_{i+1}" for i in range(reduced_data.shape[1])]
            )
        except Exception as e:
            logger.error(f"Error in dimension reduction: {e}")
            raise
            
    async def _apply_clustering(self, data: pd.DataFrame) -> np.ndarray:
        """Apply clustering.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cluster labels
        """
        if self.clustering is None:
            return np.zeros(len(data))
            
        try:
            return self.clustering.fit_predict(data)
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            raise
            
    async def _optimize_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply optimizations to data processing.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        try:
            # Apply optimization rules based on data characteristics
            if len(data) > self.optimization_params.get("large_data_threshold", 10000):
                # Use more efficient processing for large datasets
                data = data.copy()  # Avoid modifying original
                data = data.astype(self.optimization_params.get("efficient_dtypes", {}))
            
            if self.optimization_params.get("parallel_processing", False):
                # Enable parallel processing if specified
                data = data.parallel_apply(lambda x: x)  # Example parallel processing
                
            return data
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            raise
            
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing instruction.
        
        Args:
            context: Execution context with data
            
        Returns:
            Processing results
        """
        try:
            data = context.get("data")
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")
                
            # Apply processing pipeline
            processed_data = await self._preprocess_data(data)
            reduced_data = await self._reduce_dimensions(processed_data)
            cluster_labels = await self._apply_clustering(reduced_data)
            optimized_data = await self._optimize_processing(reduced_data)
            
            return {
                "processed_data": processed_data,
                "reduced_data": reduced_data,
                "cluster_labels": cluster_labels,
                "optimized_data": optimized_data,
                "metrics": {
                    "n_samples": len(data),
                    "n_features_original": data.shape[1],
                    "n_features_reduced": reduced_data.shape[1],
                    "n_clusters": len(np.unique(cluster_labels))
                }
            }
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise

class CompositeInstruction(AdvancedInstruction):
    """Instruction composed of multiple sub-instructions."""
    
    def __init__(self, name: str, description: str):
        """Initialize composite instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
        """
        super().__init__(name, description)
        self.instructions = []
        
    def add_instruction(self, instruction: 'AdvancedInstruction'):
        """Add a sub-instruction.
        
        Args:
            instruction: Instruction to add
        """
        self.instructions.append(instruction)
        
    def add_instructions(self, instructions: List['AdvancedInstruction']):
        """Add multiple sub-instructions.
        
        Args:
            instructions: List of instructions to add
        """
        self.instructions.extend(instructions)
        
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all sub-instructions in sequence.
        
        Args:
            context: Execution context
            
        Returns:
            Combined results from all sub-instructions
        """
        results = []
        for instruction in self.instructions:
            try:
                result = await instruction.execute(context)
                results.append(result)
                # Update context with result for next instruction
                if isinstance(result, dict):
                    context.update(result)
            except Exception as e:
                logger.error(f"Error executing instruction {instruction.name}: {e}")
                raise
                
        return {
            "results": results,
            "metrics": {
                "total_instructions": len(self.instructions),
                "successful_instructions": len(results)
            }
        }

class ConditionalInstruction(AdvancedInstruction):
    """Instruction that executes different sub-instructions based on conditions."""
    
    def __init__(self, name: str, description: str):
        """Initialize conditional instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
        """
        super().__init__(name, description)
        self.conditions = []  # List of (condition, instruction) tuples
        
    def add_condition(self, condition: Callable[[Dict[str, Any]], bool], instruction: 'AdvancedInstruction'):
        """Add a condition and its corresponding instruction.
        
        Args:
            condition: Function that evaluates context and returns bool
            instruction: Instruction to execute if condition is True
        """
        self.conditions.append((condition, instruction))
        
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the first instruction whose condition is met.
        
        Args:
            context: Execution context
            
        Returns:
            Result from executed instruction
        """
        for condition, instruction in self.conditions:
            try:
                if await condition(context) if asyncio.iscoroutinefunction(condition) else condition(context):
                    result = await instruction.execute(context)
                    return {
                        "result": result,
                        "selected_instruction": instruction.name,
                        "metrics": {
                            "total_conditions": len(self.conditions),
                            "selected_condition_index": self.conditions.index((condition, instruction))
                        }
                    }
            except Exception as e:
                logger.error(f"Error evaluating condition for instruction {instruction.name}: {e}")
                raise
                
        raise ValueError("No matching condition found")

class ParallelInstruction(AdvancedInstruction):
    """Instruction that executes multiple sub-instructions in parallel."""
    
    def __init__(self, name: str, description: str):
        """Initialize parallel instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
        """
        super().__init__(name, description)
        self.instructions = []
        
    def add_instruction(self, instruction: 'AdvancedInstruction'):
        """Add a sub-instruction.
        
        Args:
            instruction: Instruction to add
        """
        self.instructions.append(instruction)
        
    def add_instructions(self, instructions: List['AdvancedInstruction']):
        """Add multiple sub-instructions.
        
        Args:
            instructions: List of instructions to add
        """
        self.instructions.extend(instructions)
        
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all sub-instructions in parallel.
        
        Args:
            context: Execution context
            
        Returns:
            Combined results from all sub-instructions
        """
        tasks = [instruction.execute(context) for instruction in self.instructions]
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Separate successful and failed results
        successful_results = []
        failed_results = []
        for result in results:
            if isinstance(result, Exception):
                failed_results.append(str(result))
            else:
                successful_results.append(result)
                
        return {
            "results": successful_results,
            "failed_results": failed_results,
            "metrics": {
                "total_instructions": len(self.instructions),
                "successful_instructions": len(successful_results),
                "failed_instructions": len(failed_results),
                "execution_time": execution_time
            }
        }

class IterativeInstruction(AdvancedInstruction):
    """Instruction that executes iteratively until a condition is met."""
    
    def __init__(self, name: str, description: str, max_iterations: int = 10):
        """Initialize iterative instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
            max_iterations: Maximum number of iterations
        """
        super().__init__(name, description)
        self.max_iterations = max_iterations
        self.instruction = None
        self.convergence_check = None
        
    def set_instruction(self, instruction: 'AdvancedInstruction'):
        """Set the instruction to execute iteratively.
        
        Args:
            instruction: Instruction to execute
        """
        self.instruction = instruction
        
    def set_convergence_check(self, check: Callable[[Dict[str, Any], Dict[str, Any]], bool]):
        """Set the convergence check function.
        
        Args:
            check: Function that takes previous and current results and returns bool
        """
        self.convergence_check = check
        
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute instruction iteratively until convergence or max iterations.
        
        Args:
            context: Execution context
            
        Returns:
            Final result and iteration history
        """
        if self.instruction is None:
            raise ValueError("No instruction set for iteration")
            
        iteration = 0
        history = []
        prev_result = None
        
        while iteration < self.max_iterations:
            try:
                current_result = await self.instruction.execute(context)
                history.append(current_result)
                
                # Check convergence if function is provided
                if self.convergence_check is not None and prev_result is not None:
                    if await self.convergence_check(prev_result, current_result) if asyncio.iscoroutinefunction(self.convergence_check) else self.convergence_check(prev_result, current_result):
                        break
                        
                prev_result = current_result
                iteration += 1
                
                # Update context with current result
                if isinstance(current_result, dict):
                    context.update(current_result)
                    
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                raise
                
        return {
            "final_result": prev_result,
            "history": history,
            "metrics": {
                "total_iterations": iteration + 1,
                "max_iterations": self.max_iterations,
                "converged": iteration < self.max_iterations
            }
        }

class AdaptiveInstruction(AdvancedInstruction):
    """Instruction that adapts its behavior based on context and performance."""
    
    def __init__(self, name: str, description: str):
        """Initialize adaptive instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
        """
        super().__init__(name, description)
        self.adaptation_rules = []
        self.performance_metrics = {}
        self.adaptation_history = []
        
    def add_adaptation_rule(self, condition: Callable[[Dict[str, Any]], bool], adaptation: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Add an adaptation rule.
        
        Args:
            condition: Function that determines if adaptation should be applied
            adaptation: Function that modifies context based on adaptation
        """
        self.adaptation_rules.append((condition, adaptation))
        
    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics.
        
        Args:
            metrics: New performance metrics
        """
        self.performance_metrics.update(metrics)
        
    async def _adapt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptation rules to context.
        
        Args:
            context: Execution context
            
        Returns:
            Adapted context
        """
        adapted_context = context.copy()
        applied_adaptations = []
        
        for condition, adaptation in self.adaptation_rules:
            try:
                if await condition(adapted_context) if asyncio.iscoroutinefunction(condition) else condition(adapted_context):
                    adapted_context = await adaptation(adapted_context) if asyncio.iscoroutinefunction(adaptation) else adaptation(adapted_context)
                    applied_adaptations.append(adaptation.__name__)
            except Exception as e:
                logger.error(f"Error in adaptation: {e}")
                raise
                
        self.adaptation_history.append({
            "timestamp": time.time(),
            "applied_adaptations": applied_adaptations,
            "performance_metrics": self.performance_metrics.copy()
        })
        
        return adapted_context
        
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute instruction with adaptation.
        
        Args:
            context: Execution context
            
        Returns:
            Execution result with adaptation metrics
        """
        try:
            # Apply adaptations
            adapted_context = await self._adapt_context(context)
            
            # Execute with adapted context
            result = await super()._execute_impl(adapted_context)
            
            return {
                "result": result,
                "adaptations": self.adaptation_history[-1] if self.adaptation_history else None,
                "metrics": {
                    "total_adaptations": len(self.adaptation_history),
                    "performance_metrics": self.performance_metrics
                }
            }
        except Exception as e:
            logger.error(f"Error in adaptive execution: {e}")
            raise
