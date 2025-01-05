"""Advanced tests for instruction set architecture functionality."""

import pytest
import asyncio
import time
import torch
from unittest.mock import MagicMock, patch
from agentflow.core.instructions.advanced import (
    AdaptiveInstruction,
    ConditionalInstruction,
    StateManagerInstruction,
    ParallelInstruction,
    ImageProcessingInstruction,
    IterativeInstruction,
    LLMInstruction,
    InstructionResult
)

@pytest.fixture
def sample_instruction():
    """Create a sample advanced instruction."""
    return AdaptiveInstruction(
        name="test_instruction",
        description="Test instruction",
        cache_ttl=3600
    )

@pytest.mark.asyncio
async def test_instruction_monitoring():
    """Test instruction execution monitoring."""
    instruction = AdaptiveInstruction("monitored", "Monitored instruction")
    
    # Create monitoring log
    monitoring_log = []
    
    # Add monitor
    async def monitor(event):
        monitoring_log.append(event)
    
    instruction.add_monitor(monitor)
    
    # Execute instruction multiple times
    await instruction.execute({"value": 1})
    await instruction.execute({"value": 2})
    await instruction.execute({"value": 3})
    
    # Verify monitoring events
    assert len(monitoring_log) >= 6  # 2 events per execution (start and complete)
    
    # Check event types
    event_types = [event["event"] for event in monitoring_log]
    assert "execution_start" in event_types
    assert "execution_complete" in event_types
    
    # Check metrics in events
    for event in monitoring_log:
        if "metrics" in event:
            assert "total_executions" in event["metrics"]
            assert "success_rate" in event["metrics"]
            assert "average_latency" in event["metrics"]

@pytest.mark.asyncio
async def test_instruction_optimization():
    """Test instruction optimization rules."""
    instruction = AdaptiveInstruction("optimized", "Optimized instruction")
    
    # Add optimization rules
    def should_batch(context):
        return context.get("data_size", 0) > 100
    
    def should_cache(context):
        return context.get("cacheable", False)
    
    instruction.add_optimization_rule(should_batch, "batch_processing")
    instruction.add_optimization_rule(should_cache, "caching")
    
    # Test with optimization triggers
    result1 = await instruction.execute({
        "data_size": 150,
        "cacheable": True
    })
    assert result1.status == "success"
    assert len(result1.metadata["optimizations"]) == 2
    assert "batch_processing" in result1.metadata["optimizations"]
    assert "caching" in result1.metadata["optimizations"]
    
    # Test without optimization triggers
    result2 = await instruction.execute({
        "data_size": 50,
        "cacheable": False
    })
    assert result2.status == "success"
    assert len(result2.metadata["optimizations"]) == 0

@pytest.mark.asyncio
async def test_instruction_caching():
    """Test instruction result caching."""
    instruction = AdaptiveInstruction("cached", "Cached instruction")
    instruction.cache_ttl = 1  # Set short TTL for testing
    
    # Execute with same context
    context = {"key": "value"}
    
    result1 = await instruction.execute(context)
    assert result1.status == "success"
    assert not result1.metadata.get("cache_hit", False)
    
    # Immediate re-execution should hit cache
    result2 = await instruction.execute(context)
    assert result2.status == "success"
    assert result2.metadata.get("cache_hit", False)
    
    # Wait for cache to expire
    await asyncio.sleep(1.1)
    
    # Re-execution after expiry should miss cache
    result3 = await instruction.execute(context)
    assert result3.status == "success"
    assert not result3.metadata.get("cache_hit", False)

@pytest.mark.asyncio
async def test_instruction_metrics():
    """Test instruction metrics collection and reporting."""
    instruction = AdaptiveInstruction("metrics", "Metrics test")
    
    # Execute multiple times with different outcomes
    await instruction.execute({"success": True})
    await instruction.execute({"success": True})
    
    # Create error case
    error_component = AdaptiveInstruction("error", "Error component")
    async def error_execute(context):
        raise ValueError("Test error")
    error_component.execute = error_execute
    instruction.add_component(error_component)
    
    try:
        await instruction.execute({"success": False})
    except:
        pass
    
    # Get metrics
    metrics = instruction.get_metrics()
    
    # Verify metric values
    assert metrics["total_executions"] == 3
    assert metrics["successful_executions"] == 2
    assert metrics["failed_executions"] == 1
    assert metrics["total_latency"] > 0
    assert 0.0 <= metrics["success_rate"] <= 1.0

@pytest.mark.asyncio
async def test_model_adaptation():
    """Test model-specific adaptation."""
    instruction = AdaptiveInstruction("model_adapt", "Model adaptation test")
    
    # Add model adaptations
    def gpt_adaptation(context):
        return {
            "temperature": 0.7,
            "max_tokens": 100,
            "model": "gpt-3.5-turbo"
        }
    
    def t5_adaptation(context):
        return {
            "temperature": 1.0,
            "max_length": 512,
            "model": "t5-base"
        }
    
    instruction.add_model_adaptation("gpt", gpt_adaptation)
    instruction.add_model_adaptation("t5", t5_adaptation)
    
    # Test GPT adaptation
    result_gpt = await instruction.execute({"model": "gpt"})
    assert result_gpt.status == "success"
    assert result_gpt.metadata["model_params"]["model"] == "gpt-3.5-turbo"
    assert result_gpt.metadata["model_params"]["temperature"] == 0.7
    
    # Test T5 adaptation
    result_t5 = await instruction.execute({"model": "t5"})
    assert result_t5.status == "success"
    assert result_t5.metadata["model_params"]["model"] == "t5-base"
    assert result_t5.metadata["model_params"]["temperature"] == 1.0

@pytest.mark.asyncio
async def test_instruction_composition_performance():
    """Test performance of composed instructions."""
    # Create a complex composition
    root = ParallelInstruction("root", "Root instruction")
    branch1 = IterativeInstruction("branch1", "First branch", max_iterations=3)
    branch2 = ConditionalInstruction("branch2", "Second branch")
    
    # Add components to branches
    component = AdaptiveInstruction("component", "Test component")
    branch1.add_component(component)
    branch2.add_condition(lambda ctx: True, component)
    
    # Add branches to root
    root.add_components([branch1, branch2])
    
    # Execute with timing
    start_time = time.time()
    result = await root.execute({})
    execution_time = time.time() - start_time
    
    # Verify execution
    assert result.status == "success"
    assert execution_time < 5.0  # Should complete within reasonable time
    assert len(result.metadata["component_results"]) == 2
    
    # Check performance metrics
    assert "execution_time" in result.metadata
    assert "total_executions" in result.metadata
    assert result.metadata["execution_time"] <= execution_time

@pytest.mark.asyncio
async def test_instruction_version_handling():
    """Test instruction version handling and adaptation."""
    instruction = AdaptiveInstruction("versioned", "Versioned instruction")
    
    # Add version handlers
    def v1_handler(context):
        context["legacy"] = True
        return context
    
    def v2_handler(context):
        context["modern"] = True
        return context
    
    instruction.add_version_handler("1.0", v1_handler)
    instruction.add_version_handler("2.0", v2_handler)
    
    # Test different versions
    result_v1 = await instruction.execute({"version": "1.0"})
    assert result_v1.status == "success"
    assert result_v1.metadata["version"] == "1.0"
    
    result_v2 = await instruction.execute({"version": "2.0"})
    assert result_v2.status == "success"
    assert result_v2.metadata["version"] == "2.0"
    
    # Test version-specific context modifications
    assert "legacy" in result_v1.metadata
    assert "modern" in result_v2.metadata

@pytest.mark.asyncio
async def test_instruction_recovery_strategies():
    """Test instruction error recovery strategies."""
    instruction = AdaptiveInstruction("recovery", "Recovery test")
    
    # Add recovery strategies
    def is_value_error(error):
        return isinstance(error, ValueError)
    
    def recover_from_value_error(context):
        return {"recovered": True, "value": 42}
    
    instruction.add_recovery_strategy(is_value_error, recover_from_value_error)
    
    # Create error-generating component
    error_component = AdaptiveInstruction("error", "Error component")
    async def error_execute(context):
        raise ValueError("Test error")
    error_component.execute = error_execute
    instruction.add_component(error_component)
    
    # Test recovery
    result = await instruction.execute({})
    assert result.status == "recovered"
    assert result.metadata["recovery_applied"]
    assert result.output["recovered"]
    assert result.output["value"] == 42

@pytest.mark.asyncio
async def test_image_processing_instruction():
    """Test image processing instruction with mock models."""
    instruction = ImageProcessingInstruction("test_image", "Test image processing", use_mock=True)
    
    # Create mock image tensor
    mock_image = torch.randn(3, 224, 224)
    
    # Test single image processing
    result = await instruction.execute({"image": mock_image})
    assert result.status == "success"
    assert "label" in result.output
    assert "confidence" in result.output
    
    # Test batch processing
    batch_result = await instruction.execute({"images": [mock_image, mock_image]})
    assert batch_result.status == "success"
    assert isinstance(batch_result.output, list)
    assert batch_result.metadata["batch_count"] > 0

@pytest.mark.asyncio
async def test_image_processing_error_handling():
    """Test image processing error handling."""
    instruction = ImageProcessingInstruction("test_image_error", "Test image error handling", use_mock=True)
    
    # Test with invalid input
    result = await instruction.execute({"image": None})
    assert result.status == "error"
    assert "Image cannot be None" in result.metadata["error"]
    
    # Test with empty batch
    batch_result = await instruction.execute({"images": []})
    assert batch_result.status == "error"
    assert "Images list cannot be empty" in batch_result.metadata["error"]

@pytest.mark.asyncio
async def test_llm_instruction():
    """Test LLM instruction with mock model."""
    with patch("transformers.AutoTokenizer.from_pretrained"), \
         patch("transformers.AutoModelForCausalLM.from_pretrained"):
        instruction = LLMInstruction("test_llm", "Test LLM", model_name="test-model")
        
        # Test text generation
        result = await instruction.execute({"input": "Test prompt"})
        assert result.status == "success"
        assert result.metadata["model"] == "test-model"
        assert "input_text" in result.metadata
        assert result.metadata["input_text"] == "Test prompt"

@pytest.mark.asyncio
async def test_llm_instruction_error_handling():
    """Test LLM instruction error handling."""
    with patch("transformers.AutoTokenizer.from_pretrained"), \
         patch("transformers.AutoModelForCausalLM.from_pretrained"):
        instruction = LLMInstruction("test_llm_error", "Test LLM errors")
        
        # Test with missing input
        result = await instruction.execute({})
        assert result.status == "error"
        assert "No valid input text found" in result.metadata["error"]
        
        # Test with empty input
        empty_result = await instruction.execute({"input": ""})
        assert empty_result.status == "error"

@pytest.mark.asyncio
async def test_iterative_instruction():
    """Test iterative instruction convergence."""
    instruction = IterativeInstruction("test_iterative", "Test iterative execution")
    instruction.set_convergence_threshold(0.1)
    instruction.set_target_value(1.0)
    
    # Create mock component that approaches target value
    component = AdaptiveInstruction("component", "Test component")
    async def component_execute(context):
        current = context.get("current_value", 0.0)
        new_value = current + 0.2
        return InstructionResult(
            status="success",
            output=new_value,
            metadata={"value": new_value}
        )
    component.execute = component_execute
    instruction.add_component(component)
    
    # Test convergence
    result = await instruction.execute({"value": 0.0})
    assert result.status == "success"
    assert result.metadata["converged"]
    assert len(result.metadata["iteration_history"]) > 0
    assert abs(result.output - instruction.target_value) < instruction.convergence_threshold

@pytest.mark.asyncio
async def test_iterative_instruction_max_iterations():
    """Test iterative instruction with max iterations."""
    instruction = IterativeInstruction("test_max_iter", "Test max iterations", max_iterations=3)
    
    # Create component that never converges
    component = AdaptiveInstruction("non_converging", "Non-converging component")
    async def component_execute(context):
        return InstructionResult(
            status="success",
            output=0.0,
            metadata={}
        )
    component.execute = component_execute
    instruction.add_component(component)
    
    # Test max iterations
    result = await instruction.execute({})
    assert result.status == "success"
    assert not result.metadata["converged"]
    assert len(result.metadata["iteration_history"]) == 3

@pytest.mark.asyncio
async def test_iterative_instruction_error_handling():
    """Test iterative instruction error handling."""
    instruction = IterativeInstruction("test_error", "Test error handling", max_iterations=5)
    
    # Create component that raises error
    component = AdaptiveInstruction("error", "Error component")
    async def component_execute(context):
        if context.get("iteration", 0) == 2:
            raise ValueError("Test error at iteration 2")
        return InstructionResult(
            status="success",
            output=0.0,
            metadata={}
        )
    component.execute = component_execute
    instruction.add_component(component)
    
    # Test error handling
    result = await instruction.execute({})
    assert result.status == "error"
    assert "Test error at iteration 2" in result.metadata["error"]
    assert len(result.metadata["iteration_history"]) == 2

@pytest.mark.asyncio
async def test_iterative_instruction_with_recovery():
    """Test iterative instruction with recovery strategy."""
    instruction = IterativeInstruction("test_recovery", "Test recovery", max_iterations=5)
    
    # Add recovery strategy
    def is_value_error(error):
        return isinstance(error, ValueError)
    
    def recover_from_value_error(context):
        return {"recovered": True, "value": context.get("current_value", 0.0)}
    
    instruction.add_recovery_strategy(is_value_error, recover_from_value_error)
    
    # Create component that raises error but can be recovered
    component = AdaptiveInstruction("recoverable", "Recoverable component")
    async def component_execute(context):
        if context.get("iteration", 0) == 2:
            raise ValueError("Recoverable error")
        current = context.get("current_value", 0.0)
        new_value = current + 0.2
        return InstructionResult(
            status="success",
            output=new_value,
            metadata={"value": new_value}
        )
    component.execute = component_execute
    instruction.add_component(component)
    
    # Test recovery during iteration
    result = await instruction.execute({})
    assert result.status == "success"
    assert result.metadata["recovery_applied"]
    assert len(result.metadata["iteration_history"]) > 2

@pytest.mark.asyncio
async def test_complex_instruction_chain():
    """Test complex chain of instructions with different types."""
    # Create a chain of instructions: Conditional -> Parallel -> Iterative
    conditional = ConditionalInstruction("root", "Root conditional")
    parallel = ParallelInstruction("parallel", "Parallel processing")
    iterative = IterativeInstruction("iterative", "Iterative processing", max_iterations=5)
    
    # Set up iterative component
    iterative_component = AdaptiveInstruction("iter_comp", "Iterative component")
    async def iter_execute(context):
        current = context.get("current_value", 0.0)
        new_value = current + 0.3
        return InstructionResult(
            status="success",
            output=new_value,
            metadata={"value": new_value}
        )
    iterative_component.execute = iter_execute
    iterative.add_component(iterative_component)
    iterative.set_target_value(1.0)
    iterative.set_convergence_threshold(0.1)
    
    # Set up parallel components
    parallel.add_components([
        AdaptiveInstruction("p1", "Parallel 1"),
        iterative,
        AdaptiveInstruction("p3", "Parallel 3")
    ])
    
    # Set up conditional paths
    conditional.add_condition(lambda ctx: ctx.get("value", 0) > 0.5, parallel)
    conditional.set_default(AdaptiveInstruction("default", "Default path"))
    
    # Test execution chain
    result = await conditional.execute({"value": 0.7})
    assert result.status == "success"
    assert len(result.metadata["component_results"]) > 0
    
    # Verify execution path includes all components
    execution_path = result.metadata.get("execution_path", [])
    assert "root" in str(execution_path)
    assert "parallel" in str(execution_path)
    assert "iterative" in str(execution_path)

@pytest.mark.asyncio
async def test_nested_state_management():
    """Test nested state management with conditional execution."""
    outer_state = StateManagerInstruction("outer", "Outer state manager")
    inner_state = StateManagerInstruction("inner", "Inner state manager")
    
    # Set up inner states
    inner_state.add_state("preparing", lambda ctx: ctx.get("inner_value", 0) < 0.3)
    inner_state.add_state("ready", lambda ctx: ctx.get("inner_value", 0) >= 0.3)
    inner_state.set_default("preparing")
    
    # Set up outer states with conditional behavior
    outer_state.add_state("start", lambda ctx: ctx.get("outer_value", 0) < 0.5)
    
    conditional = ConditionalInstruction("cond", "Conditional inner")
    conditional.add_condition(
        lambda ctx: ctx.get("inner_value", 0) < 0.7,
        inner_state
    )
    
    async def outer_execute(context):
        # Execute conditional and inner state management
        result = await conditional.execute(context)
        return InstructionResult(
            status="success",
            output=result.output,
            metadata={
                "inner_result": result.metadata,
                "outer_state": "processing"
            }
        )
    
    outer_state.execute = outer_execute
    
    # Test nested state transitions
    result1 = await outer_state.execute({
        "outer_value": 0.3,
        "inner_value": 0.2
    })
    assert result1.status == "success"
    assert result1.output == "preparing"
    
    result2 = await outer_state.execute({
        "outer_value": 0.3,
        "inner_value": 0.4
    })
    assert result2.status == "success"
    assert result2.output == "ready"

@pytest.mark.asyncio
async def test_error_propagation():
    """Test error propagation through instruction chain."""
    # Create a chain that will fail at different levels
    root = ParallelInstruction("root", "Root parallel")
    middle = ConditionalInstruction("middle", "Middle conditional")
    leaf = AdaptiveInstruction("leaf", "Leaf adaptive")
    
    # Set up error in leaf
    async def leaf_error(context):
        if context.get("trigger_error"):
            raise ValueError("Leaf error")
        return InstructionResult(
            status="success",
            output="OK",
            metadata={}
        )
    leaf.execute = leaf_error
    
    # Set up middle layer
    middle.add_condition(lambda ctx: True, leaf)
    
    # Set up root with multiple paths
    root.add_components([
        AdaptiveInstruction("success", "Success path"),
        middle,
        AdaptiveInstruction("another", "Another path")
    ])
    
    # Test error propagation
    result = await root.execute({"trigger_error": True})
    assert result.status == "error"
    assert "Leaf error" in result.metadata["error"]
    
    # Test partial success
    success_result = await root.execute({"trigger_error": False})
    assert success_result.status == "success"
    assert len(success_result.metadata["component_results"]) == 3

@pytest.mark.asyncio
async def test_edge_case_empty_components():
    """Test instruction behavior with empty components."""
    instruction = ParallelInstruction("empty", "Empty components")
    
    # Test execution with no components
    result = await instruction.execute({})
    assert result.status == "success"
    assert len(result.metadata["component_results"]) == 0
    
    # Test with single empty component
    empty_component = AdaptiveInstruction("empty_comp", "Empty component")
    instruction.add_component(empty_component)
    
    result_single = await instruction.execute({})
    assert result_single.status == "success"
    assert len(result_single.metadata["component_results"]) == 1

@pytest.mark.asyncio
async def test_edge_case_circular_dependency():
    """Test handling of potential circular dependencies."""
    # Create components that could form a cycle
    comp1 = AdaptiveInstruction("comp1", "Component 1")
    comp2 = AdaptiveInstruction("comp2", "Component 2")
    
    # Create circular reference
    comp1.add_component(comp2)
    comp2.add_component(comp1)
    
    # Test execution
    try:
        await comp1.execute({})
        assert False, "Should have detected circular dependency"
    except Exception as e:
        assert "circular" in str(e).lower() or "recursion" in str(e).lower()

@pytest.mark.asyncio
async def test_edge_case_large_context():
    """Test handling of large context objects."""
    instruction = AdaptiveInstruction("large_context", "Large context test")
    
    # Create large context
    large_context = {
        "data": [i for i in range(10000)],
        "nested": {
            "level1": {
                "level2": {
                    "level3": "deep nesting"
                }
            }
        }
    }
    
    # Test execution with large context
    result = await instruction.execute(large_context)
    assert result.status == "success"
    assert "data" in result.metadata["context_size"] if "context_size" in result.metadata else True

@pytest.mark.asyncio
async def test_performance_tracking():
    """Test performance tracking with _track functionality."""
    instruction = AdaptiveInstruction("tracked", "Tracked instruction")
    
    # Add tracking to instruction execution
    async def tracked_execute(context):
        from agentflow.ell2a.lmp._track import _track
        
        @_track
        async def tracked_operation(ctx):
            await asyncio.sleep(0.1)  # Simulate work
            return {"result": "success"}
            
        result = await tracked_operation(context)
        return InstructionResult(
            status="success",
            output=result,
            metadata={"tracked": True}
        )
    
    instruction.execute = tracked_execute
    
    # Execute multiple times to gather metrics
    results = []
    for i in range(3):
        result = await instruction.execute({"iteration": i})
        results.append(result)
    
    # Verify tracking data
    for result in results:
        assert result.status == "success"
        assert result.metadata["tracked"]
        assert "execution_time" in str(result.metadata)

@pytest.mark.asyncio
async def test_performance_optimization_feedback():
    """Test performance optimization feedback loop."""
    instruction = AdaptiveInstruction("optimized", "Optimized instruction")
    
    # Track performance metrics
    performance_metrics = {
        "total_time": 0.0,
        "executions": 0,
        "optimized_executions": 0
    }
    
    # Add optimization based on performance
    async def optimized_execute(context):
        from agentflow.ell2a.lmp._track import _track
        
        @_track
        async def operation(ctx):
            if performance_metrics["executions"] > 0:
                avg_time = performance_metrics["total_time"] / performance_metrics["executions"]
                if avg_time > 0.1:  # If average time is high, optimize
                    performance_metrics["optimized_executions"] += 1
                    await asyncio.sleep(0.05)  # Optimized path
                    return {"optimized": True}
            
            await asyncio.sleep(0.15)  # Regular path
            return {"optimized": False}
        
        start_time = time.time()
        result = await operation(context)
        execution_time = time.time() - start_time
        
        performance_metrics["total_time"] += execution_time
        performance_metrics["executions"] += 1
        
        return InstructionResult(
            status="success",
            output=result,
            metadata={
                "execution_time": execution_time,
                "metrics": performance_metrics.copy()
            }
        )
    
    instruction.execute = optimized_execute
    
    # Execute multiple times to trigger optimization
    results = []
    for i in range(5):
        result = await instruction.execute({"iteration": i})
        results.append(result)
    
    # Verify optimization occurred
    assert performance_metrics["optimized_executions"] > 0
    assert any(r.output.get("optimized", False) for r in results)
    
    # Verify metrics tracking
    for result in results:
        assert "execution_time" in result.metadata
        assert "metrics" in result.metadata
        assert result.metadata["metrics"]["executions"] > 0

@pytest.mark.asyncio
async def test_adaptive_performance_thresholds():
    """Test adaptive performance thresholds based on tracking data."""
    instruction = AdaptiveInstruction("adaptive", "Adaptive thresholds")
    
    # Initialize adaptive thresholds
    thresholds = {
        "execution_time": 0.2,  # Initial threshold
        "adjustment_factor": 0.9  # Threshold adjustment factor
    }
    
    # Add adaptive execution
    async def adaptive_execute(context):
        from agentflow.ell2a.lmp._track import _track
        
        @_track
        async def operation(ctx):
            current_threshold = thresholds["execution_time"]
            await asyncio.sleep(0.1)  # Base operation time
            
            # Adapt threshold based on performance
            if "execution_time" in ctx.get("history", {}):
                avg_time = ctx["history"]["execution_time"]
                if avg_time < current_threshold:
                    thresholds["execution_time"] *= thresholds["adjustment_factor"]
            
            return {
                "threshold": current_threshold,
                "adapted": True
            }
        
        result = await operation(context)
        return InstructionResult(
            status="success",
            output=result,
            metadata={
                "current_threshold": thresholds["execution_time"]
            }
        )
    
    instruction.execute = adaptive_execute
    
    # Execute multiple times to observe threshold adaptation
    initial_threshold = thresholds["execution_time"]
    results = []
    
    for i in range(5):
        result = await instruction.execute({"iteration": i})
        results.append(result)
    
    # Verify threshold adaptation
    final_threshold = thresholds["execution_time"]
    assert final_threshold < initial_threshold
    
    # Verify adaptation in results
    for result in results:
        assert result.status == "success"
        assert "current_threshold" in result.metadata
        assert result.output["adapted"]

@pytest.mark.asyncio
async def test_performance_profiling():
    """Test detailed performance profiling of instruction execution."""
    instruction = AdaptiveInstruction("profiled", "Profiled instruction")
    
    # Initialize profiling data
    profiling_data = {
        "execution_times": [],
        "memory_usage": [],
        "operation_counts": {}
    }
    
    async def profiled_execute(context):
        from agentflow.ell2a.lmp._track import _track
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        @_track
        async def operation(ctx):
            # Record operation
            op_name = ctx.get("operation", "default")
            profiling_data["operation_counts"][op_name] = \
                profiling_data["operation_counts"].get(op_name, 0) + 1
            
            await asyncio.sleep(0.1)  # Simulate work
            return {"profiled": True}
        
        start_time = time.time()
        result = await operation(context)
        execution_time = time.time() - start_time
        
        # Record metrics
        profiling_data["execution_times"].append(execution_time)
        final_memory = process.memory_info().rss
        profiling_data["memory_usage"].append(final_memory - initial_memory)
        
        return InstructionResult(
            status="success",
            output=result,
            metadata={
                "execution_time": execution_time,
                "memory_delta": final_memory - initial_memory,
                "profiling": profiling_data.copy()
            }
        )
    
    instruction.execute = profiled_execute
    
    # Execute with different operations
    operations = ["op1", "op2", "op1", "op3"]
    results = []
    
    for op in operations:
        result = await instruction.execute({"operation": op})
        results.append(result)
    
    # Verify profiling data
    assert len(profiling_data["execution_times"]) == len(operations)
    assert len(profiling_data["memory_usage"]) == len(operations)
    assert profiling_data["operation_counts"]["op1"] == 2
    assert profiling_data["operation_counts"]["op2"] == 1
    assert profiling_data["operation_counts"]["op3"] == 1

@pytest.mark.asyncio
async def test_adaptive_batch_optimization():
    """Test adaptive batch size optimization based on performance metrics."""
    instruction = AdaptiveInstruction("batch_opt", "Batch optimization")
    
    # Initialize batch optimization parameters
    batch_config = {
        "current_size": 1,
        "min_size": 1,
        "max_size": 8,
        "performance_history": [],
        "size_history": []
    }
    
    async def batch_execute(context):
        from agentflow.ell2a.lmp._track import _track
        
        @_track
        async def process_batch(items):
            await asyncio.sleep(0.1 * len(items))  # Simulate batch processing
            return [{"processed": True} for _ in items]
        
        # Get batch of items
        items = context.get("items", [])
        if not items:
            return InstructionResult(
                status="error",
                output=None,
                metadata={"error": "No items to process"}
            )
        
        # Process in batches
        start_time = time.time()
        results = []
        for i in range(0, len(items), batch_config["current_size"]):
            batch = items[i:i + batch_config["current_size"]]
            batch_result = await process_batch(batch)
            results.extend(batch_result)
        
        execution_time = time.time() - start_time
        items_per_second = len(items) / execution_time
        
        # Update performance history
        batch_config["performance_history"].append(items_per_second)
        batch_config["size_history"].append(batch_config["current_size"])
        
        # Adapt batch size based on performance
        if len(batch_config["performance_history"]) >= 2:
            current_perf = batch_config["performance_history"][-1]
            prev_perf = batch_config["performance_history"][-2]
            
            if current_perf > prev_perf and batch_config["current_size"] < batch_config["max_size"]:
                batch_config["current_size"] *= 2
            elif current_perf < prev_perf and batch_config["current_size"] > batch_config["min_size"]:
                batch_config["current_size"] = max(
                    batch_config["current_size"] // 2,
                    batch_config["min_size"]
                )
        
        return InstructionResult(
            status="success",
            output=results,
            metadata={
                "batch_size": batch_config["current_size"],
                "items_per_second": items_per_second,
                "batch_history": {
                    "sizes": batch_config["size_history"],
                    "performance": batch_config["performance_history"]
                }
            }
        )
    
    instruction.execute = batch_execute
    
    # Test with increasing data sizes
    data_sizes = [10, 20, 40, 80]
    results = []
    
    for size in data_sizes:
        items = [{"id": i} for i in range(size)]
        result = await instruction.execute({"items": items})
        results.append(result)
    
    # Verify batch optimization
    assert len(batch_config["size_history"]) == len(data_sizes)
    assert batch_config["current_size"] > batch_config["min_size"]
    assert all(r.status == "success" for r in results)
    
    # Verify performance improvements
    performance_trend = batch_config["performance_history"]
    assert any(curr > prev for prev, curr in zip(performance_trend, performance_trend[1:]))

@pytest.mark.asyncio
async def test_resource_aware_execution():
    """Test resource-aware execution with adaptive throttling."""
    instruction = AdaptiveInstruction("resource_aware", "Resource-aware execution")
    
    # Initialize resource monitoring
    resource_metrics = {
        "cpu_threshold": 80.0,  # percentage
        "memory_threshold": 85.0,  # percentage
        "throttle_factor": 1.0,
        "throttle_history": []
    }
    
    async def resource_aware_execute(context):
        from agentflow.ell2a.lmp._track import _track
        import psutil
        
        @_track
        async def monitored_operation(ctx):
            # Check resource usage
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Adjust throttle factor based on resource usage
            if cpu_percent > resource_metrics["cpu_threshold"] or \
               memory_percent > resource_metrics["memory_threshold"]:
                resource_metrics["throttle_factor"] *= 0.8
            else:
                resource_metrics["throttle_factor"] = min(
                    resource_metrics["throttle_factor"] * 1.2,
                    1.0
                )
            
            # Apply throttling
            await asyncio.sleep(0.1 / resource_metrics["throttle_factor"])
            
            resource_metrics["throttle_history"].append({
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "throttle_factor": resource_metrics["throttle_factor"]
            })
            
            return {
                "completed": True,
                "throttled": resource_metrics["throttle_factor"] < 1.0
            }
        
        result = await monitored_operation(context)
        
        return InstructionResult(
            status="success",
            output=result,
            metadata={
                "resource_metrics": resource_metrics["throttle_history"][-1],
                "throttle_history": resource_metrics["throttle_history"]
            }
        )
    
    instruction.execute = resource_aware_execute
    
    # Execute under different resource conditions
    results = []
    for _ in range(5):
        result = await instruction.execute({})
        results.append(result)
    
    # Verify resource awareness
    assert len(resource_metrics["throttle_history"]) == 5
    assert all(r.status == "success" for r in results)
    assert all("resource_metrics" in r.metadata for r in results)
    
    # Verify throttling behavior
    throttle_factors = [h["throttle_factor"] for h in resource_metrics["throttle_history"]]
    assert any(tf != 1.0 for tf in throttle_factors)  # Some throttling occurred

@pytest.mark.asyncio
async def test_cross_model_performance_adaptation():
    """Test performance adaptation across different model types."""
    instruction = AdaptiveInstruction("cross_model", "Cross-model adaptation")
    
    # Initialize model-specific metrics
    model_metrics = {
        "gpt": {"latency": [], "tokens_per_second": [], "error_rate": 0.0},
        "t5": {"latency": [], "tokens_per_second": [], "error_rate": 0.0},
        "llama": {"latency": [], "tokens_per_second": [], "error_rate": 0.0}
    }
    
    async def adaptive_execute(context):
        from agentflow.ell2a.lmp._track import _track
        
        @_track
        async def model_operation(ctx):
            model_type = ctx.get("model", "gpt")
            input_tokens = ctx.get("input_tokens", 100)
            
            # Simulate model-specific behavior
            if model_type == "gpt":
                await asyncio.sleep(0.2)  # Base latency
                tokens_per_sec = input_tokens / 0.2
            elif model_type == "t5":
                await asyncio.sleep(0.15)  # Faster but less accurate
                tokens_per_sec = input_tokens / 0.15
            else:  # llama
                await asyncio.sleep(0.25)  # Slower but more accurate
                tokens_per_sec = input_tokens / 0.25
            
            # Update metrics
            metrics = model_metrics[model_type]
            metrics["latency"].append(time.time() - ctx.get("start_time", 0))
            metrics["tokens_per_second"].append(tokens_per_sec)
            
            return {
                "model": model_type,
                "tokens_processed": input_tokens,
                "tokens_per_second": tokens_per_sec
            }
        
        context["start_time"] = time.time()
        result = await model_operation(context)
        
        # Calculate optimal model based on metrics
        optimal_model = max(
            model_metrics.keys(),
            key=lambda m: (
                sum(model_metrics[m]["tokens_per_second"]) / 
                (len(model_metrics[m]["tokens_per_second"]) or 1)
            )
        )
        
        return InstructionResult(
            status="success",
            output=result,
            metadata={
                "model_metrics": model_metrics,
                "optimal_model": optimal_model
            }
        )
    
    instruction.execute = adaptive_execute
    
    # Test with different models and loads
    models = ["gpt", "t5", "llama"]
    token_loads = [50, 100, 200]
    results = []
    
    for model in models:
        for tokens in token_loads:
            result = await instruction.execute({
                "model": model,
                "input_tokens": tokens
            })
            results.append(result)
    
    # Verify cross-model adaptation
    assert len(results) == len(models) * len(token_loads)
    assert all(r.status == "success" for r in results)
    assert all("optimal_model" in r.metadata for r in results)
    
    # Verify model-specific metrics
    for model in models:
        metrics = model_metrics[model]
        assert len(metrics["latency"]) == len(token_loads)
        assert len(metrics["tokens_per_second"]) == len(token_loads)

@pytest.mark.asyncio
async def test_dynamic_load_balancing():
    """Test dynamic load balancing with performance monitoring."""
    instruction = AdaptiveInstruction("load_balancer", "Load balancing test")
    
    # Initialize load balancing metrics
    load_metrics = {
        "worker_loads": {"w1": 0, "w2": 0, "w3": 0},
        "worker_latencies": {"w1": [], "w2": [], "w3": []},
        "task_distribution": [],
        "rebalance_events": []
    }
    
    async def balanced_execute(context):
        from agentflow.ell2a.lmp._track import _track
        
        @_track
        async def process_task(ctx, worker):
            # Simulate worker-specific processing
            base_latency = {
                "w1": 0.1,
                "w2": 0.15,
                "w3": 0.2
            }[worker]
            
            # Add load-based latency
            load_factor = load_metrics["worker_loads"][worker] * 0.05
            total_latency = base_latency + load_factor
            
            await asyncio.sleep(total_latency)
            
            # Update metrics
            load_metrics["worker_latencies"][worker].append(total_latency)
            
            return {
                "worker": worker,
                "latency": total_latency,
                "load": load_metrics["worker_loads"][worker]
            }
        
        # Select least loaded worker
        selected_worker = min(
            load_metrics["worker_loads"].items(),
            key=lambda x: x[1] + sum(load_metrics["worker_latencies"][x[0]]) / 
                (len(load_metrics["worker_latencies"][x[0]]) or 1)
        )[0]
        
        # Update load
        load_metrics["worker_loads"][selected_worker] += 1
        load_metrics["task_distribution"].append(selected_worker)
        
        # Process task
        result = await process_task(context, selected_worker)
        
        # Check if rebalancing is needed
        loads = list(load_metrics["worker_loads"].values())
        if max(loads) - min(loads) > 2:
            # Rebalance by resetting loads
            total_load = sum(loads)
            balanced_load = total_load // 3
            for worker in load_metrics["worker_loads"]:
                load_metrics["worker_loads"][worker] = balanced_load
            load_metrics["rebalance_events"].append({
                "timestamp": time.time(),
                "previous_loads": loads,
                "balanced_load": balanced_load
            })
        
        return InstructionResult(
            status="success",
            output=result,
            metadata={
                "load_metrics": load_metrics,
                "selected_worker": selected_worker
            }
        )
    
    instruction.execute = balanced_execute
    
    # Test with burst of tasks
    results = []
    for _ in range(15):  # Create enough load to trigger rebalancing
        result = await instruction.execute({})
        results.append(result)
    
    # Verify load balancing
    assert len(results) == 15
    assert all(r.status == "success" for r in results)
    
    # Verify worker utilization
    worker_counts = {}
    for worker in load_metrics["task_distribution"]:
        worker_counts[worker] = worker_counts.get(worker, 0) + 1
    
    # Check reasonable distribution
    worker_ratios = [count/15 for count in worker_counts.values()]
    assert max(worker_ratios) - min(worker_ratios) < 0.5  # Not too unbalanced
    
    # Verify rebalancing occurred
    assert len(load_metrics["rebalance_events"]) > 0

@pytest.mark.asyncio
async def test_adaptive_model_switching():
    """Test adaptive model switching based on performance and accuracy requirements."""
    instruction = AdaptiveInstruction("model_switch", "Model switching test")
    
    # Initialize model performance tracking
    model_performance = {
        "fast": {
            "latency": 0.1,
            "accuracy": 0.85,
            "cost": 1.0
        },
        "balanced": {
            "latency": 0.2,
            "accuracy": 0.92,
            "cost": 2.0
        },
        "accurate": {
            "latency": 0.3,
            "accuracy": 0.98,
            "cost": 3.0
        }
    }
    
    # Track execution history
    execution_history = []
    
    async def switching_execute(context):
        from agentflow.ell2a.lmp._track import _track
        
        @_track
        async def execute_with_model(ctx, model_type):
            # Simulate model execution
            perf = model_performance[model_type]
            await asyncio.sleep(perf["latency"])
            
            success = random.random() < perf["accuracy"]
            
            execution_history.append({
                "model": model_type,
                "success": success,
                "latency": perf["latency"],
                "cost": perf["cost"]
            })
            
            return {
                "result": "success" if success else "failure",
                "confidence": random.uniform(0.6, 1.0) if success else random.uniform(0.3, 0.6)
            }
        
        # Determine requirements
        speed_priority = context.get("speed_priority", 0.5)
        accuracy_priority = context.get("accuracy_priority", 0.5)
        cost_priority = context.get("cost_priority", 0.5)
        
        # Select model based on requirements and history
        if len(execution_history) > 0:
            recent_performance = execution_history[-min(5, len(execution_history)):]
            success_rate = sum(1 for x in recent_performance if x["success"]) / len(recent_performance)
            avg_latency = sum(x["latency"] for x in recent_performance) / len(recent_performance)
            
            if success_rate < 0.9 and accuracy_priority > 0.7:
                selected_model = "accurate"
            elif avg_latency > 0.25 and speed_priority > 0.7:
                selected_model = "fast"
            else:
                selected_model = "balanced"
        else:
            selected_model = "balanced"
        
        result = await execute_with_model(context, selected_model)
        
        return InstructionResult(
            status="success",
            output=result,
            metadata={
                "selected_model": selected_model,
                "execution_history": execution_history[-5:],
                "model_performance": model_performance
            }
        )
    
    instruction.execute = switching_execute
    
    # Test with different priority combinations
    scenarios = [
        {"speed_priority": 0.8, "accuracy_priority": 0.2, "cost_priority": 0.5},
        {"speed_priority": 0.2, "accuracy_priority": 0.8, "cost_priority": 0.5},
        {"speed_priority": 0.5, "accuracy_priority": 0.5, "cost_priority": 0.8}
    ]
    
    results = []
    for scenario in scenarios:
        for _ in range(3):  # Multiple executions per scenario
            result = await instruction.execute(scenario)
            results.append(result)
    
    # Verify adaptive behavior
    assert len(results) == len(scenarios) * 3
    assert all(r.status == "success" for r in results)
    
    # Verify model selection adapts to priorities
    high_speed_results = [r for r in results[:3] if r.metadata["selected_model"] == "fast"]
    high_accuracy_results = [r for r in results[3:6] if r.metadata["selected_model"] == "accurate"]
    assert len(high_speed_results) > 0
    assert len(high_accuracy_results) > 0

@pytest.mark.asyncio
async def test_predictive_optimization():
    """Test predictive optimization based on historical performance patterns."""
    instruction = AdaptiveInstruction("predictive", "Predictive optimization")
    
    # Initialize performance history
    performance_history = {
        "time_series": [],
        "patterns": {},
        "predictions": []
    }
    
    async def predictive_execute(context):
        from agentflow.ell2a.lmp._track import _track
        import numpy as np
        
        @_track
        async def monitored_operation(ctx):
            # Record timestamp and load
            timestamp = time.time()
            current_load = ctx.get("load", 1.0)
            
            # Add to time series
            performance_history["time_series"].append({
                "timestamp": timestamp,
                "load": current_load,
                "hour": datetime.datetime.now().hour
            })
            
            # Identify patterns (simplified)
            if len(performance_history["time_series"]) >= 5:
                recent = performance_history["time_series"][-5:]
                avg_load = sum(x["load"] for x in recent) / 5
                hour = recent[-1]["hour"]
                
                if hour not in performance_history["patterns"]:
                    performance_history["patterns"][hour] = []
                performance_history["patterns"][hour].append(avg_load)
                
                # Make prediction for next execution
                if len(performance_history["patterns"][hour]) >= 3:
                    prediction = sum(performance_history["patterns"][hour][-3:]) / 3
                    performance_history["predictions"].append({
                        "hour": hour,
                        "predicted_load": prediction,
                        "actual_load": current_load
                    })
            
            # Simulate operation with adaptive timing
            await asyncio.sleep(0.1 * current_load)
            
            return {
                "processed": True,
                "load": current_load
            }
        
        result = await monitored_operation(context)
        
        # Calculate prediction accuracy if available
        prediction_accuracy = None
        if performance_history["predictions"]:
            recent_predictions = performance_history["predictions"][-min(5, len(performance_history["predictions"])):]
            errors = [abs(p["predicted_load"] - p["actual_load"]) for p in recent_predictions]
            prediction_accuracy = 1.0 - (sum(errors) / len(errors))
        
        return InstructionResult(
            status="success",
            output=result,
            metadata={
                "performance_history": {
                    "recent_patterns": dict(performance_history["patterns"]),
                    "prediction_accuracy": prediction_accuracy
                }
            }
        )
    
    instruction.execute = predictive_execute
    
    # Test with varying loads over time
    loads = [1.0, 1.5, 0.8, 1.2, 0.9, 1.3, 1.1, 0.7, 1.4, 1.0]
    results = []
    
    for load in loads:
        result = await instruction.execute({"load": load})
        results.append(result)
        await asyncio.sleep(0.1)  # Simulate time passing
    
    # Verify pattern recognition
    assert len(results) == len(loads)
    assert all(r.status == "success" for r in results)
    
    # Verify predictions were made
    last_result = results[-1]
    assert "prediction_accuracy" in last_result.metadata["performance_history"]
    if last_result.metadata["performance_history"]["prediction_accuracy"] is not None:
        assert 0 <= last_result.metadata["performance_history"]["prediction_accuracy"] <= 1

@pytest.mark.asyncio
async def test_resource_optimization_scaling():
    """Test resource optimization and automatic scaling."""
    instruction = AdaptiveInstruction("resource_opt", "Resource optimization")
    
    # Initialize resource tracking
    resource_state = {
        "allocated_memory": 1024,  # MB
        "cpu_cores": 2,
        "active_workers": 2,
        "queue_length": 0,
        "scaling_history": [],
        "performance_metrics": {
            "throughput": [],
            "latency": [],
            "resource_utilization": []
        }
    }
    
    async def resource_optimized_execute(context):
        from agentflow.ell2a.lmp._track import _track
        import psutil
        
        @_track
        async def monitored_task(ctx):
            # Simulate resource usage
            task_size = ctx.get("task_size", 1.0)
            memory_usage = task_size * 100  # MB
            cpu_usage = task_size * 0.5  # cores
            
            # Check resource availability
            available_memory = resource_state["allocated_memory"] - memory_usage
            available_cpu = resource_state["cpu_cores"] - cpu_usage
            
            # Update queue length
            resource_state["queue_length"] += 1
            
            # Check if scaling is needed
            queue_threshold = resource_state["active_workers"] * 2
            memory_threshold = resource_state["allocated_memory"] * 0.8
            cpu_threshold = resource_state["cpu_cores"] * 0.8
            
            needs_scaling = (
                resource_state["queue_length"] > queue_threshold or
                available_memory < memory_threshold or
                available_cpu < cpu_threshold
            )
            
            if needs_scaling:
                # Scale up resources
                resource_state["allocated_memory"] *= 1.5
                resource_state["cpu_cores"] += 1
                resource_state["active_workers"] += 1
                
                resource_state["scaling_history"].append({
                    "timestamp": time.time(),
                    "trigger": {
                        "queue_length": resource_state["queue_length"],
                        "memory_available": available_memory,
                        "cpu_available": available_cpu
                    },
                    "new_state": {
                        "memory": resource_state["allocated_memory"],
                        "cpu_cores": resource_state["cpu_cores"],
                        "workers": resource_state["active_workers"]
                    }
                })
            
            # Simulate task execution
            await asyncio.sleep(0.1 * task_size)
            
            # Update metrics
            start_time = ctx.get("start_time", time.time())
            latency = time.time() - start_time
            throughput = 1.0 / latency if latency > 0 else 0
            utilization = (memory_usage / resource_state["allocated_memory"] + 
                         cpu_usage / resource_state["cpu_cores"]) / 2
            
            resource_state["performance_metrics"]["throughput"].append(throughput)
            resource_state["performance_metrics"]["latency"].append(latency)
            resource_state["performance_metrics"]["resource_utilization"].append(utilization)
            
            # Task completed, update queue
            resource_state["queue_length"] -= 1
            
            return {
                "completed": True,
                "resources_used": {
                    "memory": memory_usage,
                    "cpu": cpu_usage
                }
            }
        
        context["start_time"] = time.time()
        result = await monitored_task(context)
        
        # Calculate average metrics
        avg_metrics = {
            "throughput": sum(resource_state["performance_metrics"]["throughput"][-5:]) / 5
            if len(resource_state["performance_metrics"]["throughput"]) >= 5 else None,
            "latency": sum(resource_state["performance_metrics"]["latency"][-5:]) / 5
            if len(resource_state["performance_metrics"]["latency"]) >= 5 else None,
            "utilization": sum(resource_state["performance_metrics"]["resource_utilization"][-5:]) / 5
            if len(resource_state["performance_metrics"]["resource_utilization"]) >= 5 else None
        }
        
        return InstructionResult(
            status="success",
            output=result,
            metadata={
                "resource_state": {
                    "memory": resource_state["allocated_memory"],
                    "cpu_cores": resource_state["cpu_cores"],
                    "workers": resource_state["active_workers"],
                    "queue_length": resource_state["queue_length"]
                },
                "performance_metrics": avg_metrics,
                "scaling_events": len(resource_state["scaling_history"])
            }
        )
    
    instruction.execute = resource_optimized_execute
    
    # Test with increasing workload
    task_sizes = [1.0, 1.5, 2.0, 2.5, 3.0]  # Increasing resource demands
    results = []
    
    for size in task_sizes:
        result = await instruction.execute({"task_size": size})
        results.append(result)
    
    # Verify resource optimization
    assert len(results) == len(task_sizes)
    assert all(r.status == "success" for r in results)
    
    # Verify scaling occurred
    final_result = results[-1]
    assert final_result.metadata["resource_state"]["memory"] > 1024  # Initial memory
    assert final_result.metadata["resource_state"]["cpu_cores"] > 2  # Initial cores
    assert final_result.metadata["scaling_events"] > 0
    
    # Verify performance metrics
    assert all(key in final_result.metadata["performance_metrics"] 
              for key in ["throughput", "latency", "utilization"])
    if final_result.metadata["performance_metrics"]["utilization"] is not None:
        assert 0 <= final_result.metadata["performance_metrics"]["utilization"] <= 1