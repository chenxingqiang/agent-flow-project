"""Advanced validation and verification system for instruction validation."""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import re
import logging

from .formal import FormalInstruction
from .metrics import ValidationResult, MetricType
from .analyzer import AnalysisResult

logger = logging.getLogger(__name__)

class ValidationType(Enum):
    """Types of validation checks."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    FORMAL = "formal"
    RUNTIME = "runtime"
    SECURITY = "security"
    RESOURCE = "resource"
    BEHAVIORAL = "behavioral"
    COMPLIANCE = "compliance"

@dataclass
class ValidationRule:
    """Represents a validation rule with configurable parameters."""
    type: ValidationType
    condition: str
    threshold: float
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationContext:
    """Context for validation execution containing rules, metrics, and history."""
    rules: List[ValidationRule]
    metrics: Dict[MetricType, float]
    analysis: AnalysisResult
    history: List[Dict[str, Any]]

class ValidationSeverity(Enum):
    """Severity levels for validation results."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()

@dataclass
class ValidationResult:
    """Comprehensive validation result with metrics and violations."""
    is_valid: bool
    score: float
    metrics: Dict[str, float]
    violations: List[str]
    recommendations: List[str]

class BaseValidator:
    """Base validator class providing common validation functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base validator.
        
        Args:
            config (Optional[Dict[str, Any]]): Validator configuration. Defaults to None.
        """
        self.config = config or {}
        self.rules: Dict[str, bool] = {}
    
    def validate(self, instructions: List[FormalInstruction], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate instructions using configured rules.
        
        Args:
            instructions (List[FormalInstruction]): Instructions to validate
            context (Optional[ValidationContext]): Validation context. Defaults to None.
            
        Returns:
            ValidationResult: Result of validation
        """
        raise NotImplementedError("Subclasses must implement validate method")
    
    def _aggregate_results(self, results: List[ValidationResult]) -> ValidationResult:
        """Aggregate multiple validation results into a single result.
        
        Args:
            results (List[ValidationResult]): List of validation results
            
        Returns:
            ValidationResult: Aggregated result
        """
        if not results:
            return ValidationResult(is_valid=True, score=1.0, metrics={}, violations=[], recommendations=[])
            
        is_valid = all(r.is_valid for r in results)
        avg_score = sum(r.score for r in results) / len(results)
        all_metrics = {}
        all_violations = []
        all_recommendations = []
        
        for r in results:
            all_metrics.update(r.metrics)
            all_violations.extend(r.violations)
            all_recommendations.extend(r.recommendations)
            
        return ValidationResult(
            is_valid=is_valid,
            score=avg_score,
            metrics=all_metrics,
            violations=all_violations,
            recommendations=all_recommendations
        )
    
    def _check_types(self, instructions: List[FormalInstruction]) -> ValidationResult:
        """Check instruction types for validity.
        
        Args:
            instructions (List[FormalInstruction]): Instructions to check
            
        Returns:
            ValidationResult: Type check result
        """
        violations = []
        for instruction in instructions:
            if not isinstance(instruction, FormalInstruction):
                violations.append(f"Invalid instruction type: {type(instruction)}")
                
        return ValidationResult(
            is_valid=len(violations) == 0,
            score=1.0 if len(violations) == 0 else 0.0,
            metrics={"type_check": 1.0 if len(violations) == 0 else 0.0},
            violations=violations,
            recommendations=["Review instruction types"] if violations else []
        )

class StaticValidator(BaseValidator):
    """Static validator for instruction validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize static validator."""
        super().__init__(config)
        self.rules = {
            'type_check': True,
            'syntax_check': True,
            'naming_convention': True
        }

    def validate(self, instructions: List[FormalInstruction], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate instructions."""
        # If no instructions, return default result
        if not instructions:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                metrics={},
                violations=[],
                recommendations=[]
            )

        results = []
        for instruction in instructions:
            # If no validate_syntax method, return a default result
            result = ValidationResult(
                is_valid=True,
                score=1.0,
                metrics={},
                violations=[],
                recommendations=[]
            )
            
            # If a validate_syntax method exists, use it
            if hasattr(instruction, 'validate_syntax'):
                try:
                    syntax_valid = instruction.validate_syntax()
                    result = ValidationResult(
                        is_valid=syntax_valid,
                        score=1.0 if syntax_valid else 0.0,
                        metrics={},
                        violations=[] if syntax_valid else ["Syntax validation failed"],
                        recommendations=[] if syntax_valid else ["Review instruction syntax"]
                    )
                except Exception as e:
                    result = ValidationResult(
                        is_valid=False,
                        score=0.0,
                        metrics={},
                        violations=[f"Syntax check error: {str(e)}"],
                        recommendations=["Debug syntax validation method"]
                    )
            
            results.append(result)
        
        return self._aggregate_results(results)

    def _check_syntax(self, instructions: List[FormalInstruction]) -> ValidationResult:
        """Check instruction syntax."""
        # Return all instruction syntax check results
        return [
            ValidationResult(
                is_valid=hasattr(instruction, 'validate_syntax') and instruction.validate_syntax(),
                score=1.0 if hasattr(instruction, 'validate_syntax') and instruction.validate_syntax() else 0.0,
                metrics={},
                violations=[] if hasattr(instruction, 'validate_syntax') and instruction.validate_syntax() else ["Syntax validation failed"],
                recommendations=[] if hasattr(instruction, 'validate_syntax') and instruction.validate_syntax() else ["Review instruction syntax"]
            )
            for instruction in instructions
        ]

class SecurityValidator(BaseValidator):
    """Security validator for instruction validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security validator."""
        super().__init__(config)

    def validate(self, instructions: List[FormalInstruction], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate instructions."""
        # If no instructions, return default result
        if not instructions:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                metrics={},
                violations=[],
                recommendations=[]
            )

        results = []
        for instruction in instructions:
            # If no validate_security method, return a default result
            result = ValidationResult(
                is_valid=True,
                score=1.0,
                metrics={},
                violations=[],
                recommendations=[]
            )
            
            # If a validate_security method exists, use it
            if hasattr(instruction, 'validate_security'):
                try:
                    security_valid = instruction.validate_security()
                    result = ValidationResult(
                        is_valid=security_valid,
                        score=1.0 if security_valid else 0.0,
                        metrics={},
                        violations=[] if security_valid else ["Security validation failed"],
                        recommendations=[] if security_valid else ["Review instruction security"]
                    )
                except Exception as e:
                    result = ValidationResult(
                        is_valid=False,
                        score=0.0,
                        metrics={},
                        violations=[f"Security check error: {str(e)}"],
                        recommendations=["Debug security validation method"]
                    )
            
            results.append(result)
        
        return self._aggregate_results(results)

class ResourceValidator(BaseValidator):
    """Resource validator for instruction validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize resource validator."""
        super().__init__(config)

    def validate(self, instructions: List[FormalInstruction], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate instructions."""
        # If no instructions, return default result
        if not instructions:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                metrics={},
                violations=[],
                recommendations=[]
            )

        results = []
        for instruction in instructions:
            # If no validate_resource method, return a default result
            result = ValidationResult(
                is_valid=True,
                score=1.0,
                metrics={},
                violations=[],
                recommendations=[]
            )
            
            # If a validate_resource method exists, use it
            if hasattr(instruction, 'validate_resource'):
                try:
                    resource_valid = instruction.validate_resource()
                    result = ValidationResult(
                        is_valid=resource_valid,
                        score=1.0 if resource_valid else 0.0,
                        metrics={},
                        violations=[] if resource_valid else ["Resource validation failed"],
                        recommendations=[] if resource_valid else ["Review instruction resource usage"]
                    )
                except Exception as e:
                    result = ValidationResult(
                        is_valid=False,
                        score=0.0,
                        metrics={},
                        violations=[f"Resource check error: {str(e)}"],
                        recommendations=["Debug resource validation method"]
                    )
            
            results.append(result)
        
        return self._aggregate_results(results)

class BehavioralValidator(BaseValidator):
    """Behavioral validator for instruction validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize behavioral validator."""
        super().__init__(config)

    def validate(self, instructions: List[FormalInstruction], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate instructions."""
        # If no instructions, return default result
        if not instructions:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                metrics={},
                violations=[],
                recommendations=[]
            )

        results = []
        for instruction in instructions:
            # If no validate_behavior method, return a default result
            result = ValidationResult(
                is_valid=True,
                score=1.0,
                metrics={},
                violations=[],
                recommendations=[]
            )
            
            # If a validate_behavior method exists, use it
            if hasattr(instruction, 'validate_behavior'):
                try:
                    behavior_valid = instruction.validate_behavior()
                    result = ValidationResult(
                        is_valid=behavior_valid,
                        score=1.0 if behavior_valid else 0.0,
                        metrics={},
                        violations=[] if behavior_valid else ["Behavioral validation failed"],
                        recommendations=[] if behavior_valid else ["Review instruction behavioral characteristics"]
                    )
                except Exception as e:
                    result = ValidationResult(
                        is_valid=False,
                        score=0.0,
                        metrics={},
                        violations=[f"Behavioral check error: {str(e)}"],
                        recommendations=["Debug behavioral validation method"]
                    )
            
            results.append(result)
        
        return self._aggregate_results(results)

class DynamicValidator(BaseValidator):
    """Dynamic validator for instruction validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dynamic validator."""
        super().__init__(config)
        self.rules = config or {}

    def validate(self, instructions: List[FormalInstruction], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate instructions dynamically."""
        # If no instructions, return default result
        if not instructions:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                metrics={},
                violations=[],
                recommendations=[]
            )

        results = []
        for instruction in instructions:
            # If no validate_dynamic method, return a default result
            result = ValidationResult(
                is_valid=True,
                score=1.0,
                metrics={},
                violations=[],
                recommendations=[]
            )
            
            # If a validate_dynamic method exists, use it
            if hasattr(instruction, 'validate_dynamic'):
                try:
                    dynamic_valid = instruction.validate_dynamic()
                    result = ValidationResult(
                        is_valid=dynamic_valid,
                        score=1.0 if dynamic_valid else 0.0,
                        metrics={},
                        violations=[] if dynamic_valid else ["Dynamic validation failed"],
                        recommendations=[] if dynamic_valid else ["Review instruction dynamic characteristics"]
                    )
                except Exception as e:
                    result = ValidationResult(
                        is_valid=False,
                        score=0.0,
                        metrics={},
                        violations=[f"Dynamic check error: {str(e)}"],
                        recommendations=["Debug dynamic validation method"]
                    )
            
            results.append(result)
        
        return self._aggregate_results(results)

class ValidationEngine:
    """Advanced validation engine coordinating multiple validators."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize validation engine.
        
        Args:
            config (Dict[str, Any]): Configuration for validation engine and validators.
        """
        self.config = config
        self._validators = {}  # Lazy-loaded validators
        self.rules = self._load_rules()
    
    def _get_validator(self, validator_type: ValidationType) -> BaseValidator:
        """Get or create a validator instance.
        
        Args:
            validator_type (ValidationType): Type of validator to get
            
        Returns:
            BaseValidator: Validator instance
        """
        if validator_type not in self._validators:
            validator_class = {
                ValidationType.STATIC: StaticValidator,
                ValidationType.DYNAMIC: DynamicValidator,
                ValidationType.SECURITY: SecurityValidator,
                ValidationType.RESOURCE: ResourceValidator,
                ValidationType.BEHAVIORAL: BehavioralValidator
            }.get(validator_type)
            
            if validator_class:
                self._validators[validator_type] = validator_class(self.config)
                
        return self._validators.get(validator_type)
    
    def validate(self, instructions: List[FormalInstruction], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate instructions using all applicable validators.
        
        Args:
            instructions (List[FormalInstruction]): Instructions to validate
            context (Optional[ValidationContext]): Validation context. Defaults to None.
            
        Returns:
            ValidationResult: Combined validation result
        """
        results = []
        for rule in self.rules:
            validator = self._get_validator(rule.type)
            if validator:
                result = validator.validate(instructions, context)
                results.append(result)
                
        return self._aggregate_results(results)
    
    def _load_rules(self) -> List[ValidationRule]:
        """Load validation rules from configuration.
        
        Returns:
            List[ValidationRule]: List of validation rules
        """
        rules = []
        for rule_config in self.config.get('rules', []):
            try:
                rule = ValidationRule(
                    type=ValidationType(rule_config['type']),
                    condition=rule_config['condition'],
                    threshold=float(rule_config['threshold']),
                    priority=int(rule_config['priority']),
                    metadata=rule_config.get('metadata', {})
                )
                rules.append(rule)
            except (KeyError, ValueError) as e:
                logger.error(f"Invalid rule configuration: {e}")
                
        return sorted(rules, key=lambda x: x.priority, reverse=True)
