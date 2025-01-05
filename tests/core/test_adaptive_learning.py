"""Tests for adaptive learning and optimization functionality."""

import pytest
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from agentflow.core.adaptive_learning import (
    PatternMatcher,
    AdaptiveLearner,
    AdaptiveOptimizer,
    InstructionPattern,
    AdaptiveMetrics
)
from agentflow.core.agentic_calling import (
    AgentInstruction,
    FunctionContext,
    ExecutionResult
)

@pytest.fixture
def pattern_matcher():
    """Create a pattern matcher instance."""
    return PatternMatcher(min_frequency=2, min_success_rate=0.7)

@pytest.fixture
def adaptive_learner():
    """Create an adaptive learner instance."""
    return AdaptiveLearner()

@pytest.fixture
def adaptive_optimizer():
    """Create an adaptive optimizer instance."""
    return AdaptiveOptimizer()

@pytest.fixture
def sample_instruction():
    """Create a sample instruction."""
    async def test_func(x: int, context: FunctionContext) -> int:
        return x + len(context.variables)
    
    return AgentInstruction(
        name="test_instruction",
        func=test_func,
        parameters={"x": 0}
    )

@pytest.mark.asyncio
async def test_pattern_matching(pattern_matcher):
    """Test pattern matching functionality."""
    # Add sequence observations
    pattern_matcher.add_sequence(
        ["instr1", "instr2"],
        True,
        0.5,
        {"var1", "var2"}
    )
    pattern_matcher.add_sequence(
        ["instr1", "instr2"],
        True,
        0.6,
        {"var1", "var3"}
    )
    
    # Find patterns
    patterns = pattern_matcher.find_patterns()
    
    assert len(patterns) == 1
    assert patterns[0].sequence == ["instr1", "instr2"]
    assert patterns[0].frequency == 2
    assert patterns[0].success_rate == 1.0
    assert patterns[0].avg_latency == 0.55
    assert patterns[0].context_dependencies == {"var1", "var2", "var3"}

@pytest.mark.asyncio
async def test_adaptive_learning(adaptive_learner, sample_instruction):
    """Test adaptive learning functionality."""
    # Create execution result
    result = ExecutionResult(
        value=42,
        status="success",
        preserved_context=FunctionContext(
            variables={"test": "value"},
            memory={},
            state={}
        ),
        metrics={"execution_time": 0.1}
    )
    
    # Learn from execution
    await adaptive_learner.learn_from_execution(
        sample_instruction,
        result,
        result.preserved_context
    )
    
    # Verify learning
    assert adaptive_learner.metrics.pattern_discoveries == 1
    assert len(adaptive_learner.learned_patterns) > 0

@pytest.mark.asyncio
async def test_instruction_optimization(adaptive_learner, sample_instruction):
    """Test instruction optimization."""
    # Add learning data
    for i in range(3):
        result = ExecutionResult(
            value=42 + i,
            status="success",
            preserved_context=FunctionContext(
                variables={"test": f"value{i}"},
                memory={},
                state={}
            ),
            metrics={"execution_time": 0.1}
        )
        
        await adaptive_learner.learn_from_execution(
            sample_instruction,
            result,
            result.preserved_context
        )
    
    # Optimize instruction
    optimized = await adaptive_learner.optimize_instruction(sample_instruction)
    
    assert optimized is not None
    assert optimized.name == f"{sample_instruction.name}_optimized"
    assert "pattern_optimized" in optimized.optimization_hints
    assert optimized.optimization_hints["original_name"] == sample_instruction.name

@pytest.mark.asyncio
async def test_sequence_optimization(adaptive_optimizer, sample_instruction):
    """Test sequence optimization."""
    instructions = [sample_instruction] * 3
    results = []
    contexts = []
    
    # Generate execution data
    for i in range(3):
        results.append(ExecutionResult(
            value=42 + i,
            status="success",
            preserved_context=FunctionContext(
                variables={"test": f"value{i}"},
                memory={},
                state={}
            ),
            metrics={"execution_time": 0.1}
        ))
        contexts.append(results[-1].preserved_context)
    
    # Learn from sequence
    await adaptive_optimizer.learn_from_sequence(
        instructions,
        results,
        contexts
    )
    
    # Optimize sequence
    optimized = await adaptive_optimizer.optimize_instruction_sequence(instructions)
    
    assert len(optimized) == 3
    assert any(instr.name.endswith("_optimized") for instr in optimized)

@pytest.mark.asyncio
async def test_optimization_stability(adaptive_optimizer, sample_instruction):
    """Test optimization stability."""
    # Perform multiple optimization cycles
    results = []
    for i in range(5):
        instructions = [sample_instruction]
        optimized = await adaptive_optimizer.optimize_instruction_sequence(instructions)
        results.append(optimized)
    
    # Verify consistency
    first_result = results[0]
    for result in results[1:]:
        assert len(result) == len(first_result)
        assert result[0].name == first_result[0].name

@pytest.mark.asyncio
async def test_pattern_discovery(pattern_matcher):
    """Test pattern discovery functionality."""
    # Add diverse patterns
    sequences = [
        (["A", "B", "C"], True, 0.1),
        (["A", "B", "C"], True, 0.2),
        (["X", "Y"], False, 0.3),
        (["X", "Y"], True, 0.1),
        (["A", "B", "C"], True, 0.15)
    ]
    
    for seq, success, latency in sequences:
        pattern_matcher.add_sequence(seq, success, latency, {"var1"})
    
    # Find patterns
    patterns = pattern_matcher.find_patterns()
    
    assert len(patterns) > 0
    # ABC pattern should be found with high success rate
    abc_pattern = next((p for p in patterns if p.sequence == ["A", "B", "C"]), None)
    assert abc_pattern is not None
    assert abc_pattern.success_rate == 1.0
    assert abc_pattern.frequency == 3

@pytest.mark.asyncio
async def test_adaptive_metrics(adaptive_optimizer, sample_instruction):
    """Test adaptive metrics collection."""
    # Generate execution data
    instructions = [sample_instruction] * 3
    results = [
        ExecutionResult(
            value=42,
            status="success",
            preserved_context=FunctionContext(
                variables={"test": "value"},
                memory={},
                state={}
            ),
            metrics={"execution_time": 0.1}
        )
    ] * 3
    contexts = [r.preserved_context for r in results]
    
    # Learn and optimize
    await adaptive_optimizer.learn_from_sequence(instructions, results, contexts)
    await adaptive_optimizer.optimize_instruction_sequence(instructions)
    
    # Get statistics
    stats = adaptive_optimizer.get_optimization_stats()
    
    assert "total_patterns" in stats
    assert "active_optimizations" in stats
    assert "learning_cycles" in stats
    assert len(stats["optimization_history"]) > 0

@pytest.mark.asyncio
async def test_context_dependency_handling(adaptive_learner, sample_instruction):
    """Test context dependency handling."""
    # Create execution result with context dependencies
    result = ExecutionResult(
        value=42,
        status="success",
        preserved_context=FunctionContext(
            variables={"required_var": "value"},
            memory={},
            state={}
        ),
        metrics={"execution_time": 0.1}
    )
    
    # Learn from execution
    await adaptive_learner.learn_from_execution(
        sample_instruction,
        result,
        result.preserved_context
    )
    
    # Optimize instruction
    optimized = await adaptive_learner.optimize_instruction(sample_instruction)
    
    assert optimized is not None
    assert "context_dependencies" in optimized.optimization_hints
    assert "required_var" in optimized.optimization_hints["context_dependencies"]

@pytest.mark.asyncio
async def test_optimization_reset(adaptive_optimizer):
    """Test optimization state reset."""
    # Record initial state
    initial_stats = adaptive_optimizer.get_optimization_stats()
    
    # Reset state
    adaptive_optimizer.reset_optimization_state()
    
    # Verify reset
    reset_stats = adaptive_optimizer.get_optimization_stats()
    assert reset_stats["total_patterns"] == 0
    assert reset_stats["active_optimizations"] == 0
    assert reset_stats["learning_cycles"] == 0
    assert len(reset_stats["optimization_history"]) == 0

@pytest.mark.asyncio
async def test_concurrent_optimization(adaptive_optimizer, sample_instruction):
    """Test concurrent optimization handling."""
    # Create multiple instruction sequences
    sequences = [[sample_instruction] * 2 for _ in range(3)]
    
    # Optimize concurrently
    optimization_tasks = [
        adaptive_optimizer.optimize_instruction_sequence(seq)
        for seq in sequences
    ]
    
    results = await asyncio.gather(*optimization_tasks)
    
    # Verify results
    assert len(results) == 3
    for result in results:
        assert len(result) == 2 