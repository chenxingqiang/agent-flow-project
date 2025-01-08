"""Tests for validation functionality."""
import pytest
from agentflow.core.isa.validation import (
    ValidationEngine,
    ValidationRule,
    ValidationContext,
    ValidationType,
    StaticValidator,
    DynamicValidator,
    SecurityValidator,
    ResourceValidator,
    BehavioralValidator
)
from agentflow.core.isa.formal import FormalInstruction
from agentflow.core.isa.analyzer import AnalysisResult, AnalysisType
from agentflow.core.isa.metrics import MetricType, ValidationResult

@pytest.fixture
def sample_instructions():
    """Create sample instructions for testing."""
    return [
        FormalInstruction(id="1", name="init", params={"x": 1}),
        FormalInstruction(id="2", name="process", params={"data": "test"}),
        FormalInstruction(id="3", name="validate", params={"check": True}),
        FormalInstruction(id="4", name="store", params={"key": "result"})
    ]

@pytest.fixture
def sample_analysis():
    """Create sample analysis result."""
    return AnalysisResult(
        type=AnalysisType.BEHAVIOR,
        metrics={},
        insights=[],
        recommendations=[],
        confidence=0.8
    )

@pytest.fixture
def validation_rules():
    """Create sample validation rules."""
    return [
        ValidationRule(
            type=ValidationType.STATIC,
            condition="syntax_check",
            threshold=0.9,
            priority=1,
            metadata={}
        ),
        ValidationRule(
            type=ValidationType.SECURITY,
            condition="access_control",
            threshold=0.95,
            priority=2,
            metadata={}
        )
    ]

@pytest.fixture
def validation_context(validation_rules, sample_analysis):
    """Create validation context."""
    return ValidationContext(
        rules=validation_rules,
        metrics={MetricType.PERFORMANCE: 0.9},
        analysis=sample_analysis,
        history=[]
    )

class TestValidationEngine:
    """Test validation engine functionality."""
    
    def test_validate(self, sample_instructions, sample_analysis):
        """Test overall validation process."""
        config = {}
        engine = ValidationEngine(config)
        result = engine.validate(sample_instructions, sample_analysis)
        
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.score, float)
        assert isinstance(result.violations, list)
        assert isinstance(result.recommendations, list)
        
    def test_rule_loading(self):
        """Test validation rule loading."""
        config = {}
        engine = ValidationEngine(config)
        rules = engine.validators
        
        assert isinstance(rules, dict)
        assert 'static' in rules
        assert 'security' in rules
        assert 'dynamic' in rules
        assert 'behavioral' in rules
        
    def test_result_aggregation(self):
        """Test validation result aggregation."""
        config = {}
        engine = ValidationEngine(config)
    
        # Create sample results
        results = [
            ValidationResult(
                type='static',
                is_valid=True,
                score=0.9,
                metrics={},
                violations=[],
                recommendations=[]
            ),
            ValidationResult(
                type='static',
                is_valid=False,
                score=0.7,
                metrics={},
                violations=["Error 1"],
                recommendations=["Fix 1"]
            )
        ]
        
        aggregated_result = engine._combine_results(results)
        
        assert not aggregated_result.is_valid
        assert aggregated_result.score == pytest.approx(0.8)
        assert len(aggregated_result.violations) == 1
        assert len(aggregated_result.recommendations) == 1

class TestStaticValidator:
    """Test static validation functionality."""
    
    def test_validate(self, sample_instructions, validation_context):
        """Test static validation."""
        config = {}
        validator = StaticValidator(config)
        result = validator.validate(sample_instructions, validation_context)
        
        assert isinstance(result, ValidationResult)
        assert result.type == 'static'
        assert result.is_valid is not None
        assert result.score is not None
        
    def test_syntax_check(self, sample_instructions):
        """Test syntax checking."""
        config = {}
        validator = StaticValidator(config)
        result = validator._check_syntax(sample_instructions)
        
        assert isinstance(result, ValidationResult)
        assert result.type == 'static'
        assert result.is_valid is False
        assert len(result.violations) > 0
        assert result.recommendations == ["Review instruction syntax"]
        
    def test_type_check(self, sample_instructions):
        """Test type checking."""
        config = {}
        validator = StaticValidator(config)
        violations = validator._check_types(sample_instructions)
        
        assert isinstance(violations, list)

class TestSecurityValidator:
    """Test security validation functionality."""
    
    def test_validate(self, sample_instructions, validation_context):
        """Test security validation."""
        config = {}
        validator = SecurityValidator(config)
        result = validator.validate(sample_instructions, validation_context)
        
        assert isinstance(result, ValidationResult)
        assert result.type == 'security'
        assert result.is_valid is not None
        assert result.score is not None
        
class TestResourceValidator:
    """Test resource validation functionality."""
    
    def test_validate(self, sample_instructions, validation_context):
        """Test resource validation."""
        config = {}
        validator = ResourceValidator(config)
        
        with pytest.raises(NotImplementedError):
            result = validator.validate(sample_instructions, validation_context)
        
class TestBehavioralValidator:
    """Test behavioral validation functionality."""
    
    def test_validate(self, sample_instructions, validation_context):
        """Test behavioral validation."""
        config = {}
        validator = BehavioralValidator(config)
        result = validator.validate(sample_instructions, validation_context)
        
        assert isinstance(result, ValidationResult)

class TestDynamicValidator:
    """Test dynamic validation functionality."""
    
    def test_validate(self, sample_instructions, validation_context):
        """Test dynamic validation."""
        validator = DynamicValidator()
        result = validator.validate(sample_instructions, validation_context)
        
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.violations, list)
        
    def test_runtime_checks(self, sample_instructions):
        """Test runtime validation checks."""
        validator = DynamicValidator()
        context = ValidationContext(
            rules=[
                ValidationRule(
                    type=ValidationType.DYNAMIC,
                    condition="runtime_check",
                    threshold=0.9,
                    priority=1
                )
            ],
            metrics={},
            analysis=AnalysisResult(
                type=AnalysisType.BEHAVIOR,
                metrics={},
                insights=[],
                recommendations=[],
                confidence=0.8
            ),
            history=[]
        )
        
        result = validator.validate(sample_instructions, context)
        assert isinstance(result, ValidationResult)
        
    def test_state_tracking(self, sample_instructions):
        """Test instruction state tracking."""
        validator = DynamicValidator()
        context = ValidationContext(
            rules=[
                ValidationRule(
                    type=ValidationType.DYNAMIC,
                    condition="state_tracking",
                    threshold=0.95,
                    priority=1
                )
            ],
            metrics={},
            analysis=AnalysisResult(
                type=AnalysisType.BEHAVIOR,
                metrics={},
                insights=[],
                recommendations=[],
                confidence=0.8
            ),
            history=[]
        )
        
        result = validator.validate(sample_instructions, context)
        assert isinstance(result, ValidationResult)

class TestAdvancedValidation:
    """Test advanced validation features."""
    
    def test_cross_instruction_validation(self, sample_instructions, validation_context):
        """Test validation across multiple instructions."""
        engine = ValidationEngine({})
        
        # Add dependent instructions
        dependent_instructions = [
            FormalInstruction(id="5", name="dependent", params={"ref": "1"}),
            FormalInstruction(id="6", name="dependent", params={"ref": "2"})
        ]
        all_instructions = sample_instructions + dependent_instructions
        
        result = engine.validate(all_instructions, validation_context.analysis)
        assert isinstance(result, ValidationResult)
        
    def test_conditional_validation(self, sample_instructions):
        """Test conditional validation rules."""
        engine = ValidationEngine({})
        context = ValidationContext(
            rules=[
                ValidationRule(
                    type=ValidationType.STATIC,
                    condition="if_then",
                    threshold=1.0,
                    priority=1,
                    metadata={
                        "if_instruction": "init",
                        "then_required": "process"
                    }
                )
            ],
            metrics={},
            analysis=AnalysisResult(
                type=AnalysisType.BEHAVIOR,
                metrics={},
                insights=[],
                recommendations=[],
                confidence=0.8
            ),
            history=[]
        )
        
        result = engine.validate(sample_instructions, context.analysis)
        assert isinstance(result, ValidationResult)
        
    def test_performance_validation(self, sample_instructions):
        """Test performance-related validation."""
        engine = ValidationEngine({})
        context = ValidationContext(
            rules=[
                ValidationRule(
                    type=ValidationType.RESOURCE,
                    condition="performance",
                    threshold=0.8,
                    priority=1,
                    metadata={
                        "max_execution_time": 1000,
                        "max_memory_usage": 1024
                    }
                )
            ],
            metrics={
                MetricType.PERFORMANCE: 0.9,
                MetricType.RESOURCE_USAGE: 0.7
            },
            analysis=AnalysisResult(
                type=AnalysisType.BEHAVIOR,
                metrics={},
                insights=[],
                recommendations=[],
                confidence=0.8
            ),
            history=[]
        )
        
        result = engine.validate(sample_instructions, context.analysis)
        assert isinstance(result, ValidationResult)
        
    def test_security_validation(self, sample_instructions):
        """Test security validation rules."""
        engine = ValidationEngine({})
        context = ValidationContext(
            rules=[
                ValidationRule(
                    type=ValidationType.SECURITY,
                    condition="access_control",
                    threshold=1.0,
                    priority=1,
                    metadata={
                        "required_permissions": ["read", "write"],
                        "restricted_operations": ["delete"]
                    }
                )
            ],
            metrics={},
            analysis=AnalysisResult(
                type=AnalysisType.BEHAVIOR,
                metrics={},
                insights=[],
                recommendations=[],
                confidence=0.8
            ),
            history=[]
        )
        
        result = engine.validate(sample_instructions, context.analysis)
        assert isinstance(result, ValidationResult)
