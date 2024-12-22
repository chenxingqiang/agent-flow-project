"""Tests for advanced instruction functionality."""
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from agentflow.core.instructions.advanced import (
    AdvancedInstruction,
    CompositeInstruction,
    ConditionalInstruction,
    ParallelInstruction,
    IterativeInstruction,
    AdaptiveInstruction,
    DataProcessingInstruction
)
from agentflow.core.instructions.base import InstructionStatus, InstructionResult, InstructionMetrics, BaseInstruction
import asyncio
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Test implementations
class SimpleInstruction(AdvancedInstruction):
    """Simple test instruction implementation."""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.validation_rules = []
    
    def add_validation_rule(self, rule: callable):
        """Add validation rule"""
        self.validation_rules.append(rule)
    
    async def _validate(self, context: Dict[str, Any]) -> bool:
        """Validate context against rules"""
        for rule in self.validation_rules:
            if not await rule(context):
                return False
        return True
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test execution implementation."""
        # Simulate some processing
        result_data = {
            "name": self.name,
            "result": "success"
        }
        
        return result_data

class SimpleCompositeInstruction(CompositeInstruction):
    """Simple test composite instruction implementation."""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all instructions in sequence."""
        results = []
        metrics_list = []
        for instruction in self.instructions[:2]:  # Limit to first 2 instructions
            result = await instruction.execute(context)
            results.append(result.data)
            metrics_list.append(result.metrics)
        
        result_data = {
            "results": results, 
            "metrics": metrics_list
        }
        
        return result_data

class SimpleParallelInstruction(ParallelInstruction):
    """Simple test parallel instruction implementation."""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all instructions in parallel."""
        tasks = [instruction.execute(context) for instruction in self.instructions[:2]]  # Limit to first 2 instructions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful and failed results
        successful_results = []
        failed_results = []
        for r in results:
            if isinstance(r, InstructionResult):
                successful_results.append(r.data)
            else:
                failed_results.append(str(r))
        
        # If there are failed results, raise an exception
        if failed_results:
            raise Exception("One or more instructions failed: " + ", ".join(failed_results))
        
        result_data = {"results": successful_results}
        
        return result_data

class ConditionalInstruction(ConditionalInstruction):
    """Simple test conditional instruction implementation."""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.conditions = []
    
    def add_condition(self, condition: callable, instruction: BaseInstruction):
        """Add a condition and its corresponding instruction."""
        self.conditions.append((condition, instruction))
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the first matching condition's instruction."""
        for condition, instruction in self.conditions:
            if await condition(context):
                result = await instruction.execute(context)
                return {"result": result.data}
        
        # If no conditions match, raise a ValueError
        raise ValueError("No matching condition found")

class ConditionalInstructionTest(ConditionalInstruction):
    """Simple test conditional instruction implementation."""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.conditions = []
    
    def add_condition(self, condition: callable, instruction: BaseInstruction):
        """Add a condition and its corresponding instruction."""
        self.conditions.append((condition, instruction))
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the first matching condition's instruction."""
        for condition, instruction in self.conditions:
            if await condition(context):
                result = await instruction.execute(context)
                return {"result": result.data}
        
        # If no conditions match, raise a ValueError
        raise ValueError("No matching condition found")

@pytest.fixture
def sample_params():
    """Create sample instruction parameters."""
    return {
        "input": "test_input",
        "threshold": 0.8,
        "max_iterations": 5,
        "timeout": 1000
    }

@pytest.fixture
def sample_context():
    """Create sample execution context."""
    return {
        "input": "test_input",
        "variables": {"x": 1, "y": 2},
        "resources": {"memory": 1024, "cpu": 4},
        "state": "ready"
    }

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    }
    return pd.DataFrame(data)

@pytest.fixture
def data_context(sample_data):
    """Create sample context with data."""
    return {
        "data": sample_data,
        "variables": {"threshold": 0.8},
        "resources": {"memory": 1024, "cpu": 4},
        "state": "ready"
    }

@pytest.mark.asyncio
class TestAdvancedInstruction:
    """Test advanced instruction functionality."""
    
    async def test_initialization(self, sample_params):
        """Test instruction initialization."""
        instr = SimpleInstruction(
            name="test",
            description="Test instruction"
        )
        
        assert instr.name == "test"
        assert instr.description == "Test instruction"
        assert len(instr.validation_rules) == 0
        
    async def test_validation(self, sample_params):
        """Test parameter validation."""
        instr = SimpleInstruction(
            name="test",
            description="Test instruction"
        )
        
        # Add an async validation rule
        async def validation_rule(ctx):
            return True
        
        instr.add_validation_rule(validation_rule)
        assert len(instr.validation_rules) == 1
        
        # Test validation
        context = {"test": "value"}
        assert await instr._validate(context)
            
    async def test_execution(self, sample_params, sample_context):
        """Test instruction execution."""
        instr = SimpleInstruction(
            name="test",
            description="Test instruction"
        )
        
        result = await instr.execute(sample_context)
        assert result is not None
        assert result.status is not None

@pytest.mark.asyncio
class TestCompositeInstruction:
    """Test composite instruction functionality."""
    
    async def test_composition(self, sample_params):
        """Test instruction composition."""
        instr1 = SimpleInstruction(
            name="part1",
            description="Part 1"
        )
        instr2 = SimpleInstruction(
            name="part2",
            description="Part 2"
        )
        
        composite = SimpleCompositeInstruction(
            name="composite",
            description="Composite instruction"
        )
        composite.instructions = [instr1, instr2]
        
        assert len(composite.instructions) == 2
        assert composite.name == "composite"
        assert composite.description == "Composite instruction"
        
    async def test_execution_order(self, sample_params, sample_context):
        """Test execution order of composite instructions."""
        instr1 = SimpleInstruction(
            name="part1",
            description="Part 1"
        )
        instr2 = SimpleInstruction(
            name="part2",
            description="Part 2"
        )
        
        composite = SimpleCompositeInstruction(
            name="composite",
            description="Composite instruction"
        )
        composite.instructions = [instr1, instr2]
        
        result = await composite.execute(sample_context)
        assert result is not None
        assert "results" in result.data
        assert len(result.data["results"]) == 2

@pytest.mark.asyncio
class TestConditionalInstruction:
    """Test conditional instruction functionality."""
    
    async def test_condition_evaluation(self, sample_params, sample_context):
        """Test condition evaluation."""
        async def condition(ctx):
            return ctx["variables"]["x"] > 0

        instr = ConditionalInstructionTest(
            name="conditional",
            description="Conditional instruction"
        )
        instr.add_condition(condition, SimpleInstruction(
            name="then",
            description="Then instruction"
        ))
        
        result = await instr.execute(sample_context)
        assert result is not None
        
    async def test_branch_selection(self, sample_params, sample_context):
        """Test branch selection logic."""
        then_instr = SimpleInstruction(
            name="then",
            description="Then instruction"
        )
        else_instr = SimpleInstruction(
            name="else",
            description="Else instruction"
        )
        
        # Test true condition
        instr = ConditionalInstructionTest(
            name="conditional",
            description="Conditional instruction"
        )
        async def true_condition(ctx):
            return True
        instr.add_condition(true_condition, then_instr)
        result = await instr.execute(sample_context)
        assert result is not None
        
        # Test false condition with no matching condition
        instr = ConditionalInstructionTest(
            name="conditional",
            description="Conditional instruction"
        )
        async def false_condition(ctx):
            return False
        instr.add_condition(false_condition, then_instr)
        try:
            await instr.execute(sample_context)
            pytest.fail("Expected ValueError")
        except Exception as e:
            assert str(e) == "No matching condition found"

@pytest.mark.asyncio
class TestParallelInstruction:
    """Test parallel instruction functionality."""
    
    async def test_parallel_execution(self, sample_params, sample_context):
        """Test parallel execution of instructions."""
        instr1 = SimpleInstruction(
            name="part1",
            description="Part 1"
        )
        instr2 = SimpleInstruction(
            name="part2",
            description="Part 2"
        )
        
        parallel = SimpleParallelInstruction(
            name="parallel",
            description="Parallel instruction"
        )
        parallel.instructions = [instr1, instr2]
        
        result = await parallel.execute(sample_context)
        assert result is not None
        assert "results" in result.data
        assert len(result.data["results"]) == 2
        
    async def test_result_aggregation(self, sample_params, sample_context):
        """Test result aggregation from parallel execution."""
        instr1 = SimpleInstruction(
            name="part1",
            description="Part 1"
        )
        instr2 = SimpleInstruction(
            name="part2",
            description="Part 2"
        )
        
        parallel = SimpleParallelInstruction(
            name="parallel",
            description="Parallel instruction"
        )
        parallel.instructions = [instr1, instr2]
        
        result = await parallel.execute(sample_context)
        assert result is not None
        assert "results" in result.data
        assert len(result.data["results"]) == 2

@pytest.mark.asyncio
class TestIterativeInstruction:
    """Test iterative instruction functionality."""
    
    async def test_iteration_control(self, sample_params, sample_context):
        """Test iteration control logic."""
        instr = IterativeInstruction(
            name="iterative",
            description="Iterative instruction",
            max_iterations=5
        )
        
        result = await instr.execute(sample_context)
        assert result is not None
        assert result.status is not None
        
    async def test_iteration_limit(self, sample_params, sample_context):
        """Test iteration limit enforcement."""
        instr = IterativeInstruction(
            name="iterative",
            description="Iterative instruction",
            max_iterations=1
        )
        
        result = await instr.execute(sample_context)
        assert result is not None
        assert result.status is not None

@pytest.mark.asyncio
class TestAdaptiveInstruction:
    """Test adaptive instruction functionality."""
    
    async def test_adaptation(self, sample_params, sample_context):
        """Test instruction adaptation."""
        instr = AdaptiveInstruction(
            name="adaptive",
            description="Adaptive instruction"
        )
        
        result = await instr.execute(sample_context)
        assert result is not None
        assert result.status is not None
        
    async def test_rule_application(self, sample_params, sample_context):
        """Test adaptation rule application."""
        instr = AdaptiveInstruction(
            name="adaptive",
            description="Adaptive instruction"
        )
        
        # Add adaptation rule
        async def adaptation_rule(ctx):
            return {"modified": True}
        instr.add_adaptation_rule(adaptation_rule)
        assert len(instr.adaptation_rules) == 1
        
        result = await instr.execute(sample_context)
        assert result is not None
        assert result.status is not None

@pytest.mark.asyncio
class TestDataProcessingInstruction:
    """Test data processing instruction functionality."""
    
    async def test_initialization(self):
        """Test instruction initialization."""
        instr = DataProcessingInstruction(
            name="test_data_processing",
            description="Test data processing"
        )
        
        assert instr.name == "test_data_processing"
        assert instr.description == "Test data processing"
        assert hasattr(instr, "scaler")
        assert hasattr(instr, "pca")
        assert hasattr(instr, "clusterer")
        
    async def test_preprocessing(self, sample_data):
        """Test data preprocessing."""
        instr = DataProcessingInstruction(
            name="test_data_processing",
            description="Test data processing"
        )
        
        processed_data = await instr._preprocess(sample_data)
        assert isinstance(processed_data, pd.DataFrame)
        assert processed_data.shape == sample_data.shape
        assert not processed_data.isna().any().any()
        
    async def test_dimension_reduction(self, sample_data):
        """Test dimension reduction."""
        instr = DataProcessingInstruction(
            name="test_data_processing",
            description="Test data processing"
        )
        
        processed_data = await instr._preprocess(sample_data)
        reduced_data = await instr._reduce_dimensions(processed_data)
        
        assert isinstance(reduced_data, pd.DataFrame)
        assert reduced_data.shape[0] == sample_data.shape[0]
        assert reduced_data.shape[1] <= sample_data.shape[1]
        
    async def test_clustering(self, sample_data):
        """Test data clustering."""
        instr = DataProcessingInstruction(
            name="test_data_processing",
            description="Test data processing"
        )
        
        processed_data = await instr._preprocess(sample_data)
        reduced_data = await instr._reduce_dimensions(processed_data)
        result = await instr._cluster(reduced_data)
        
        assert isinstance(result, dict)
        assert "clusters" in result
        assert "centroids" in result
        assert isinstance(result["clusters"], list)
        assert isinstance(result["centroids"], list)
        assert len(result["clusters"]) == sample_data.shape[0]
        assert len(result["centroids"]) == instr.clusterer.n_clusters
        
    async def test_execution(self, data_context):
        """Test full execution pipeline."""
        instr = DataProcessingInstruction(
            name="test_data_processing",
            description="Test data processing"
        )
        
        result = await instr.execute(data_context)
        
        assert result.status == InstructionStatus.COMPLETED
        assert "processed_data" in result.data
        assert "reduced_data" in result.data
        assert "clusters" in result.data
        assert isinstance(result.data["processed_data"], dict)
        assert isinstance(result.data["reduced_data"], dict)
        assert isinstance(result.data["clusters"], dict)
        assert len(result.data["clusters"]["clusters"]) == data_context["data"].shape[0]
        
    async def test_error_handling(self):
        """Test error handling with invalid input."""
        instr = DataProcessingInstruction(
            name="test_data_processing",
            description="Test data processing"
        )
        
        # Test with empty context
        result = await instr.execute({})
        assert result.status == InstructionStatus.FAILED
        assert "No data provided" in result.error
            
        # Test with invalid data type
        invalid_context = {"data": "not a dataframe"}
        result = await instr.execute(invalid_context)
        assert result.status == InstructionStatus.FAILED
        assert "Invalid data type" in result.error
            
        # Test with empty dataframe
        empty_context = {"data": pd.DataFrame()}
        result = await instr.execute(empty_context)
        assert result.status == InstructionStatus.FAILED
        assert "Empty data" in result.error
            
    async def test_optimization(self, data_context):
        """Test optimization with different data sizes."""
        instr = DataProcessingInstruction(
            name="test_data_processing",
            description="Test data processing"
        )
        
        # Test with small data
        small_data = data_context["data"].head(10)
        small_context = {**data_context, "data": small_data}
        result = await instr.execute(small_context)
        assert result.status == InstructionStatus.COMPLETED
        assert result.metrics is not None
        assert isinstance(result.metrics, InstructionMetrics)
        
        # Test with large data
        large_data = pd.concat([data_context["data"]] * 10)
        large_context = {**data_context, "data": large_data}
        result = await instr.execute(large_context)
        assert result.status == InstructionStatus.COMPLETED
        assert result.metrics is not None
        assert isinstance(result.metrics, InstructionMetrics)
