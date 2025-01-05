"""Tests for optimization and performance tracking functionality."""

import pytest
import asyncio
import time
from typing import Dict, Any
from agentflow.core.optimization import (
    PipelineOptimizer,
    StaticOptimizer,
    DynamicOptimizer,
    OptimizationMetrics
)
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep

@pytest.fixture
def sample_workflow():
    """Create a sample workflow for testing."""
    return WorkflowConfig(
        name="test_workflow",
        steps=[
            WorkflowStep(
                id="step1",
                name="process1",
                type="transform",
                config={"param": "value1"}
            ),
            WorkflowStep(
                id="step2",
                name="process2",
                type="transform",
                config={"param": "value2"}
            ),
            WorkflowStep(
                id="step3",
                name="process3",
                type="transform",
                config={"param": "value3"}
            )
        ]
    )

@pytest.fixture
def pipeline_optimizer():
    """Create a pipeline optimizer instance."""
    return PipelineOptimizer()

@pytest.fixture
def static_optimizer():
    """Create a static optimizer instance."""
    return StaticOptimizer()

@pytest.fixture
def dynamic_optimizer():
    """Create a dynamic optimizer instance."""
    return DynamicOptimizer()

@pytest.mark.asyncio
async def test_pipeline_optimization(pipeline_optimizer, sample_workflow):
    """Test pipeline optimization."""
    # Optimize workflow
    optimized = pipeline_optimizer.optimize_workflow(sample_workflow)
    
    # Verify optimization results
    assert len(optimized.steps) <= len(sample_workflow.steps)
    assert pipeline_optimizer.verify_optimization(optimized)
    
    # Check optimization metrics
    metrics = pipeline_optimizer.get_optimization_metrics()
    assert "pipeline_reduction" in metrics
    assert metrics["pipeline_reduction"] > 0

@pytest.mark.asyncio
async def test_static_optimization(static_optimizer, sample_workflow):
    """Test static optimization techniques."""
    # Test peephole optimization
    peephole_result = static_optimizer.apply_peephole_optimization(sample_workflow)
    assert static_optimizer.verify_peephole_optimization(peephole_result)
    
    # Test dead code elimination
    eliminated = static_optimizer.eliminate_dead_code(sample_workflow)
    assert len(eliminated.steps) <= len(sample_workflow.steps)
    
    # Test instruction combining
    combined = static_optimizer.combine_instructions(sample_workflow)
    assert len(combined.steps) <= len(sample_workflow.steps)

@pytest.mark.asyncio
async def test_dynamic_optimization(dynamic_optimizer):
    """Test dynamic optimization techniques."""
    # Test hot path detection
    async def test_function():
        await asyncio.sleep(0.1)
        return {"result": "success"}
    
    # Execute function multiple times to create hot path
    for _ in range(5):
        await test_function()
    
    hot_paths = dynamic_optimizer.detect_hot_paths()
    assert len(hot_paths) > 0
    
    # Test trace formation
    traces = dynamic_optimizer.form_traces(hot_paths)
    assert len(traces) > 0
    
    # Test adaptive recompilation
    recompiled = dynamic_optimizer.recompile_traces(traces)
    assert len(recompiled) > 0

@pytest.mark.asyncio
async def test_optimization_verification(pipeline_optimizer, sample_workflow):
    """Test optimization verification."""
    # Optimize workflow
    optimized = pipeline_optimizer.optimize_workflow(sample_workflow)
    
    # Verify semantic equivalence
    assert pipeline_optimizer.verify_semantic_equivalence(
        sample_workflow,
        optimized
    )
    
    # Verify performance improvement
    assert pipeline_optimizer.verify_performance_improvement(
        sample_workflow,
        optimized
    )

@pytest.mark.asyncio
async def test_performance_tracking(pipeline_optimizer, sample_workflow):
    """Test performance tracking."""
    # Track original performance
    original_metrics = await pipeline_optimizer.measure_performance(sample_workflow)
    
    # Optimize and track optimized performance
    optimized = pipeline_optimizer.optimize_workflow(sample_workflow)
    optimized_metrics = await pipeline_optimizer.measure_performance(optimized)
    
    # Compare metrics
    assert optimized_metrics.execution_time < original_metrics.execution_time
    assert optimized_metrics.resource_usage <= original_metrics.resource_usage

@pytest.mark.asyncio
async def test_resource_optimization(pipeline_optimizer, sample_workflow):
    """Test resource usage optimization."""
    # Measure initial resource usage
    initial_usage = pipeline_optimizer.measure_resource_usage(sample_workflow)
    
    # Optimize for resources
    optimized = pipeline_optimizer.optimize_resource_usage(sample_workflow)
    
    # Measure optimized resource usage
    optimized_usage = pipeline_optimizer.measure_resource_usage(optimized)
    
    assert optimized_usage.memory <= initial_usage.memory
    assert optimized_usage.cpu <= initial_usage.cpu

@pytest.mark.asyncio
async def test_cost_optimization(pipeline_optimizer, sample_workflow):
    """Test cost optimization."""
    # Calculate initial cost
    initial_cost = pipeline_optimizer.calculate_execution_cost(sample_workflow)
    
    # Optimize for cost
    optimized = pipeline_optimizer.optimize_cost(sample_workflow)
    
    # Calculate optimized cost
    optimized_cost = pipeline_optimizer.calculate_execution_cost(optimized)
    
    assert optimized_cost < initial_cost
    assert pipeline_optimizer.verify_cost_optimization(optimized)

@pytest.mark.asyncio
async def test_optimization_stability(pipeline_optimizer, sample_workflow):
    """Test optimization stability."""
    # Perform multiple optimization runs
    results = []
    for _ in range(5):
        optimized = pipeline_optimizer.optimize_workflow(sample_workflow)
        results.append(optimized)
    
    # Verify consistency across runs
    for i in range(1, len(results)):
        assert pipeline_optimizer.compare_optimization_results(
            results[0],
            results[i]
        )

@pytest.mark.asyncio
async def test_optimization_metrics_collection(pipeline_optimizer, sample_workflow):
    """Test optimization metrics collection."""
    # Enable detailed metrics collection
    pipeline_optimizer.enable_detailed_metrics()
    
    # Perform optimization
    optimized = pipeline_optimizer.optimize_workflow(sample_workflow)
    
    # Get collected metrics
    metrics = pipeline_optimizer.get_detailed_metrics()
    
    assert "optimization_time" in metrics
    assert "memory_reduction" in metrics
    assert "cost_reduction" in metrics
    assert metrics["optimization_success"] == True

@pytest.mark.asyncio
async def test_cross_optimization_compatibility(
    pipeline_optimizer,
    static_optimizer,
    dynamic_optimizer,
    sample_workflow
):
    """Test compatibility between different optimization types."""
    # Apply optimizations in sequence
    pipeline_result = pipeline_optimizer.optimize_workflow(sample_workflow)
    static_result = static_optimizer.optimize_workflow(pipeline_result)
    dynamic_result = await dynamic_optimizer.optimize_workflow(static_result)
    
    # Verify final result
    assert pipeline_optimizer.verify_optimization(dynamic_result)
    assert static_optimizer.verify_optimization(dynamic_result)
    assert dynamic_optimizer.verify_optimization(dynamic_result) 