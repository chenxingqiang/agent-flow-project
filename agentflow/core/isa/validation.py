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
        results = []
        
        # Run type checks
        if self.rules.get('type_check'):
            type_violations = self._check_types(instructions)
            results.append(ValidationResult(
                is_valid=len(type_violations) == 0,
                score=1.0 if len(type_violations) == 0 else 0.0,
                metrics={'type_check': 1.0 if len(type_violations) == 0 else 0.0},
                violations=type_violations,
                recommendations=["Fix instruction type errors"] if type_violations else []
            ))
        
        # Run syntax checks
        if self.rules.get('syntax_check'):
            syntax_violations = self._check_syntax(instructions)
            results.append(ValidationResult(
                is_valid=len(syntax_violations) == 0,
                score=1.0 if len(syntax_violations) == 0 else 0.0,
                metrics={'syntax_check': 1.0 if len(syntax_violations) == 0 else 0.0},
                violations=syntax_violations,
                recommendations=["Fix syntax errors"] if syntax_violations else []
            ))
        
        return self._aggregate_results(results)
        
    def _check_types(self, instructions: List[FormalInstruction]) -> List[str]:
        """Check instruction types for validity.
        
        Args:
            instructions (List[FormalInstruction]): Instructions to check
            
        Returns:
            List[str]: List of type check violations
        """
        violations = []
        for instruction in instructions:
            if not isinstance(instruction, FormalInstruction):
                violations.append(f"Invalid instruction type: {type(instruction)}")
            elif not isinstance(instruction.params, dict):
                violations.append(f"Invalid params type for instruction {instruction.id}")
        return violations
        
    def _check_syntax(self, instructions: List[FormalInstruction]) -> List[str]:
        """Check instruction syntax.
        
        Args:
            instructions (List[FormalInstruction]): Instructions to check
            
        Returns:
            List[str]: List of syntax violations
        """
        violations = []
        for instruction in instructions:
            if not instruction.id or not instruction.name:
                violations.append(f"Missing required fields in instruction {instruction.id}")
        return violations

class SecurityValidator(BaseValidator):
    """Security validator for instruction validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security validator."""
        super().__init__(config)

    def validate(self, instructions: List[FormalInstruction], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate instructions."""
        return ValidationResult(
            is_valid=True,
            score=1.0,
            metrics={},
            violations=[],
            recommendations=[]
        )

class ResourceValidator(BaseValidator):
    """Resource validator for instruction validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize resource validator."""
        super().__init__(config)

    def validate(self, instructions: List[FormalInstruction], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate instructions."""
        return ValidationResult(
            is_valid=True,
            score=1.0,
            metrics={},
            violations=[],
            recommendations=[]
        )

class BehavioralValidator(BaseValidator):
    """Behavioral validator for instruction validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize behavioral validator."""
        super().__init__(config)

    def validate(self, instructions: List[FormalInstruction], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate instructions."""
        return ValidationResult(
            is_valid=True,
            score=1.0,
            metrics={},
            violations=[],
            recommendations=[]
        )

class DynamicValidator(BaseValidator):
    """Dynamic validator for instruction validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dynamic validator."""
        super().__init__(config)
        self.rules = config or {}

    def validate(self, instructions: List[FormalInstruction], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate instructions dynamically."""
        return ValidationResult(
            is_valid=True,
            score=1.0,
            metrics={},
            violations=[],
            recommendations=[]
        )

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

    def _get_validator(self, validator_type: ValidationType) -> Optional[BaseValidator]:
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

    def _aggregate_results(self, results: List[ValidationResult]) -> ValidationResult:
        """Aggregate multiple validation results into a single result.
        
        Args:
            results (List[ValidationResult]): List of validation results
            
        Returns:
            ValidationResult: Aggregated result
        """
        if not results:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                metrics={},
                violations=[],
                recommendations=[]
            )
        
        # Combine metrics
        all_metrics = {}
        for result in results:
            all_metrics.update(result.metrics)
            
        # Calculate overall score as average of individual scores
        avg_score = sum(r.score for r in results) / len(results)
        
        # Combine violations and recommendations
        all_violations = []
        all_recommendations = []
        for result in results:
            all_violations.extend(result.violations)
            all_recommendations.extend(result.recommendations)
            
        # Overall validity is True only if all results are valid
        is_valid = all(r.is_valid for r in results)
        
        return ValidationResult(
            is_valid=is_valid,
            score=avg_score,
            metrics=all_metrics,
            violations=list(set(all_violations)),
            recommendations=list(set(all_recommendations))
        )

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
