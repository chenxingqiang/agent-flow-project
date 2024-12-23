"""Advanced instruction implementations with integrated validation."""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import logging
import ray
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ViTImageProcessor,
    ViTForImageClassification
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torchvision import transforms
import time
from .base import (
    OptimizableInstruction,
    CacheableInstruction,
    CompositeInstruction,
    ParallelInstruction,
    InstructionResult,
    InstructionStatus,
    InstructionMetrics
)
from ..isa.validation import ValidationRule, ValidationResult, ValidationType

logger = logging.getLogger(__name__)

class AdvancedInstruction(OptimizableInstruction, CacheableInstruction):
    """Base class for advanced instructions with optimization, caching, and validation."""
    
    def __init__(self, name: str, description: str, cache_ttl: int = 3600):
        """Initialize advanced instruction.
        
        Args:
            name (str): Instruction name
            description (str): Instruction description
            cache_ttl (int, optional): Cache time-to-live in seconds. Defaults to 3600.
        """
        OptimizableInstruction.__init__(self, name, description)
        CacheableInstruction.__init__(self, name, description, cache_ttl)
        self.validation_rules: List[ValidationRule] = []
    
    def add_validation_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule to the instruction.
        
        Args:
            rule (ValidationRule): Validation rule to add
        """
        self.validation_rules.append(rule)
    
    async def _validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate context against rules.
        
        Args:
            context (Dict[str, Any]): Execution context
            
        Returns:
            ValidationResult: Validation result with metrics and violations
        """
        violations = []
        metrics = {}
        
        for rule in self.validation_rules:
            try:
                if not await self._evaluate_rule(rule, context):
                    violations.append(f"Failed validation rule: {rule.condition}")
                metrics[f"rule_{rule.type.value}"] = rule.threshold
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.condition}: {e}")
                violations.append(f"Error in rule {rule.condition}: {str(e)}")
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            score=1.0 if len(violations) == 0 else 0.0,
            metrics=metrics,
            violations=violations
        )
    
    async def _evaluate_rule(self, rule: ValidationRule, context: Dict[str, Any]) -> bool:
        """Evaluate a single validation rule.
        
        Args:
            rule (ValidationRule): Rule to evaluate
            context (Dict[str, Any]): Execution context
            
        Returns:
            bool: True if rule passes, False otherwise
        """
        try:
            # Convert rule condition to callable if string
            if isinstance(rule.condition, str):
                condition = eval(f"lambda context: {rule.condition}")
            else:
                condition = rule.condition
                
            return await condition(context)
        except Exception as e:
            logger.error(f"Error evaluating rule condition: {e}")
            return False
    
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute with validation, optimization and caching"""
        try:
            # Validate context
            validation_result = await self._validate(context)
            if not validation_result.is_valid:
                self.status = InstructionStatus.FAILED
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Context validation failed: {', '.join(validation_result.violations)}",
                    metrics=self.metrics
                )
            
            # Execute with optimization and caching
            return await super().execute(context)
            
        except Exception as e:
            logger.error(f"Advanced instruction {self.name} failed: {str(e)}")
            self.status = InstructionStatus.FAILED
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e),
                metrics=self.metrics
            )

class ConditionalInstruction(AdvancedInstruction):
    """Instruction that executes different paths based on conditions"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.conditions = {}  # condition -> instruction mapping
        
    def add_condition(self, condition: callable, instruction: AdvancedInstruction):
        """Add a condition and its corresponding instruction"""
        self.conditions[condition] = instruction
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the matching instruction based on conditions"""
        for condition, instruction in self.conditions.items():
            if await condition(context):
                try:
                    result = await instruction.execute(context)
                    if result.status != InstructionStatus.COMPLETED:
                        raise Exception(f"Conditional instruction failed: {result.error}")
                    return result.data
                except Exception as e:
                    logger.error(f"Conditional instruction failed: {str(e)}")
                    raise
        # If no conditions match, raise a ValueError with specific message
        raise ValueError("No matching condition found")

class AdaptiveInstruction(AdvancedInstruction):
    """Instruction that adapts its behavior based on context"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.adaptation_rules = []
        self.learning_rate = 0.01
        
    def add_adaptation_rule(self, rule: callable):
        """Add adaptation rule"""
        self.adaptation_rules.append(rule)
    
    async def _adapt(self, context: Dict[str, Any], result: Dict[str, Any]):
        """Adapt instruction based on execution results"""
        for rule in self.adaptation_rules:
            await rule(self, context, result, self.learning_rate)
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with adaptation"""
        result = await super()._execute_impl(context)
        await self._adapt(context, result)
        return result

class IterativeInstruction(AdvancedInstruction):
    """Instruction that executes iteratively until a condition is met"""
    
    def __init__(self, name: str, description: str, max_iterations: int = 10):
        super().__init__(name, description)
        self.max_iterations = max_iterations
        self.convergence_threshold = 0.001
        
    def set_convergence_threshold(self, threshold: float):
        """Set convergence threshold"""
        self.convergence_threshold = threshold
    
    async def _check_convergence(self, current: Dict[str, Any], previous: Dict[str, Any]) -> bool:
        """Check if iteration has converged"""
        if not previous:
            return False
        
        # Simple Euclidean distance for numeric values
        diff = 0
        for key in current:
            if isinstance(current[key], (int, float)) and key in previous:
                diff += (current[key] - previous[key]) ** 2
        return np.sqrt(diff) < self.convergence_threshold
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute iteratively until convergence or max iterations"""
        previous_result = None
        for i in range(self.max_iterations):
            result = await super()._execute_impl(context)
            if await self._check_convergence(result, previous_result):
                logger.info(f"Converged after {i+1} iterations")
                break
            previous_result = result.copy()
            context.update(result)
        return result

class LLMInstruction(AdvancedInstruction):
    """Instruction that interacts with language models"""
    
    def __init__(self, name: str, description: str, model_name: str = "gpt-3.5-turbo"):
        super().__init__(name, description)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_length = 1024
        self.temperature = 0.7
        
    async def _initialize_model(self):
        """Initialize the language model"""
        if not self.model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM instruction"""
        await self._initialize_model()
        
        input_text = context.get("input", "")
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_length, truncation=True)
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=self.max_length,
            temperature=self.temperature,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}

class DataProcessingInstruction(AdvancedInstruction):
    """Instruction for data processing and analysis"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.clusterer = KMeans(n_clusters=3)
    
    async def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data"""
        return pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns)
    
    async def _reduce_dimensions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reduce dimensionality (alias for _dimension_reduction)"""
        return await self._dimension_reduction(data)
    
    async def _dimension_reduction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reduce dimensionality"""
        reduced_data = self.pca.fit_transform(data)
        return pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(reduced_data.shape[1])])
    
    async def _cluster(self, data: pd.DataFrame) -> np.ndarray:
        """Perform clustering"""
        return self.clusterer.fit_predict(data)
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing pipeline"""
        try:
            # Validate and extract data
            data = context.get('data')
            if data is None:
                raise ValueError("No data provided")
            
            # Convert to DataFrame if not already
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Validate data
            if data.empty:
                raise ValueError("Empty DataFrame provided")
            
            # Execute processing pipeline
            preprocessed_data = await self._preprocess(data)
            reduced_data = await self._dimension_reduction(preprocessed_data)
            clusters = await self._cluster(reduced_data)
            
            # Update status and return results
            self.status = InstructionStatus.COMPLETED
            return {
                'processed_data': reduced_data.to_dict(orient='records'),
                'clusters': clusters.tolist(),
                'metrics': {
                    'n_samples': len(data),
                    'n_features': data.shape[1],
                    'n_components': reduced_data.shape[1],
                    'explained_variance': float(self.pca.explained_variance_ratio_.sum())
                }
            }
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            self.status = InstructionStatus.FAILED
            raise

@ray.remote
class ImageProcessingInstruction(AdvancedInstruction):
    """Instruction for image processing and analysis with integrated validation."""
    
    def __init__(self, name: str = "process_image", description: str = "Process and analyze images"):
        """Initialize image processing instruction with default validation rules.
        
        Args:
            name (str, optional): Instruction name. Defaults to "process_image".
            description (str, optional): Instruction description. 
                Defaults to "Process and analyze images".
        """
        super().__init__(name=name, description=description)
        
        # Initialize models and transforms
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Configuration
        self.max_image_size = 1000
        self.batch_size = 32
        
        # Add default validation rules
        self.add_validation_rule(ValidationRule(
            type=ValidationType.RESOURCE,
            condition="'image' in context and isinstance(context['image'], (Image.Image, str))",
            threshold=1.0,
            priority=1,
            metadata={"description": "Validate image input"}
        ))
        
        self.add_validation_rule(ValidationRule(
            type=ValidationType.RESOURCE,
            condition="context.get('max_size', float('inf')) <= self.max_image_size",
            threshold=1.0,
            priority=2,
            metadata={"description": "Check image size limits"}
        ))
    
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute image processing with validation.
        
        Args:
            context (Dict[str, Any]): Execution context containing image data
            
        Returns:
            InstructionResult: Processing result with metrics
        """
        # Validate input
        validation_result = await self._validate(context)
        if not validation_result.is_valid:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                data={},
                error=f"Validation failed: {', '.join(validation_result.violations)}",
                metrics=InstructionMetrics(
                    start_time=time.time(),
                    end_time=time.time(),
                    tokens_used=0,
                    memory_used=0,
                    cache_hit=False,
                    optimization_applied=False,
                    parallel_execution=False
                )
            )
        
        try:
            # Process image
            start_time = time.time()
            
            # Load and preprocess image
            image = context['image']
            if isinstance(image, str):
                image = Image.open(image)
            
            # Optimize processing
            should_resize = self._should_resize_image(context)
            should_batch = self._should_batch_process(context)
            
            if should_resize:
                image = self._resize_image(image, (224, 224))
            
            if should_batch:
                results = await self._batch_process_images([image], self.batch_size)
            else:
                processed_image = self._preprocess_image(image)
                results = await self._classify_image(processed_image)
            
            end_time = time.time()
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={"results": results},
                error=None,
                metrics=InstructionMetrics(
                    start_time=start_time,
                    end_time=end_time,
                    tokens_used=len(results),
                    memory_used=0,  # TODO: Implement memory tracking
                    cache_hit=False,
                    optimization_applied=should_resize or should_batch,
                    parallel_execution=should_batch
                )
            )
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return InstructionResult(
                status=InstructionStatus.FAILED,
                data={},
                error=str(e),
                metrics=InstructionMetrics(
                    start_time=time.time(),
                    end_time=time.time(),
                    tokens_used=0,
                    memory_used=0,
                    cache_hit=False,
                    optimization_applied=False,
                    parallel_execution=False
                )
            )
