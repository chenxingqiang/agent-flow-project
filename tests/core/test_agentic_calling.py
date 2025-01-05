"""Tests for Agentic Calling functionality."""

import pytest
import asyncio
from typing import Dict, Any
from agentflow.core.agentic_calling import (
    AgenticTransformer,
    FunctionContext,
    AgentInstruction,
    ContextPreservation
)

@pytest.fixture
def sample_function():
    """Create a sample function for testing."""
    async def test_function(x: int, y: int) -> int:
        return x + y
    return test_function

@pytest.fixture
def agentic_transformer():
    """Create an agentic transformer instance."""
    return AgenticTransformer()

@pytest.mark.asyncio
async def test_function_to_agent_transformation(agentic_transformer, sample_function):
    """Test transformation from function to agent instruction."""
    # Transform function
    agent_instruction = agentic_transformer.transform(sample_function)
    
    # Verify transformation
    assert isinstance(agent_instruction, AgentInstruction)
    assert agent_instruction.name == "test_function"
    assert len(agent_instruction.parameters) == 2
    assert agent_instruction.has_context_embedding
    
    # Test execution
    result = await agent_instruction.execute({"x": 1, "y": 2})
    assert result.value == 3
    assert result.preserved_context is not None

@pytest.mark.asyncio
async def test_context_preservation(agentic_transformer):
    """Test context preservation during transformation."""
    # Create context
    context = FunctionContext(
        variables={"test_var": 42},
        memory={"cached": "value"},
        state={"status": "active"}
    )
    
    # Create function with context
    async def context_function():
        return context.variables["test_var"]
    
    # Transform with context preservation
    agent_instruction = agentic_transformer.transform_with_context(
        context_function,
        context
    )
    
    # Execute and verify context preservation
    result = await agent_instruction.execute({})
    assert result.value == 42
    assert result.preserved_context.variables == context.variables
    assert result.preserved_context.memory == context.memory
    assert result.preserved_context.state == context.state

@pytest.mark.asyncio
async def test_semantic_equivalence(agentic_transformer, sample_function):
    """Test semantic equivalence of transformed functions."""
    # Original execution
    original_result = await sample_function(5, 3)
    
    # Transform and execute
    agent_instruction = agentic_transformer.transform(sample_function)
    transformed_result = await agent_instruction.execute({"x": 5, "y": 3})
    
    assert original_result == transformed_result.value
    assert agentic_transformer.verify_semantic_equivalence(
        sample_function,
        agent_instruction
    )

@pytest.mark.asyncio
async def test_cross_model_adaptation(agentic_transformer):
    """Test cross-model adaptation of agent instructions."""
    # Create model-specific function
    async def model_function(prompt: str) -> str:
        return f"GPT processed: {prompt}"
    
    # Transform for different models
    gpt_instruction = agentic_transformer.transform_for_model(
        model_function,
        "gpt-4"
    )
    claude_instruction = agentic_transformer.transform_for_model(
        model_function,
        "claude-v1"
    )
    
    # Test execution on different models
    gpt_result = await gpt_instruction.execute({"prompt": "test"})
    claude_result = await claude_instruction.execute({"prompt": "test"})
    
    assert "GPT processed" in gpt_result.value
    assert "GPT processed" in claude_result.value
    assert gpt_instruction.model_compatibility == ["gpt-4"]
    assert claude_instruction.model_compatibility == ["claude-v1"]

@pytest.mark.asyncio
async def test_context_injection(agentic_transformer):
    """Test context injection in transformed functions."""
    # Create function with injected context
    async def context_aware_function(x: int, context: Dict[str, Any]) -> int:
        return x + context.get("increment", 0)
    
    # Transform with context injection
    agent_instruction = agentic_transformer.transform_with_injection(
        context_aware_function,
        {"increment": 10}
    )
    
    # Test execution
    result = await agent_instruction.execute({"x": 5})
    assert result.value == 15
    assert "increment" in result.preserved_context.variables

@pytest.mark.asyncio
async def test_batch_transformation(agentic_transformer):
    """Test batch transformation of multiple functions."""
    # Create multiple functions
    async def func1(x: int) -> int:
        return x * 2
        
    async def func2(y: int) -> int:
        return y + 5
    
    functions = [func1, func2]
    
    # Batch transform
    instructions = agentic_transformer.batch_transform(functions)
    
    assert len(instructions) == 2
    for instruction in instructions:
        assert isinstance(instruction, AgentInstruction)
    
    # Test execution
    results = await asyncio.gather(*[
        instructions[0].execute({"x": 3}),
        instructions[1].execute({"y": 3})
    ])
    
    assert results[0].value == 6
    assert results[1].value == 8

@pytest.mark.asyncio
async def test_optimization_preservation(agentic_transformer, sample_function):
    """Test preservation of optimization opportunities."""
    # Add optimization hints
    hints = {
        "parallelizable": True,
        "cacheable": True,
        "batch_size": 10
    }
    
    # Transform with optimization hints
    agent_instruction = agentic_transformer.transform_with_optimization(
        sample_function,
        hints
    )
    
    assert agent_instruction.optimization_hints == hints
    assert agent_instruction.supports_parallel_execution
    assert agent_instruction.supports_caching

@pytest.mark.asyncio
async def test_error_handling(agentic_transformer):
    """Test error handling in transformed functions."""
    # Create function with error
    async def error_function():
        raise ValueError("Test error")
    
    # Transform with error handling
    agent_instruction = agentic_transformer.transform_with_error_handling(
        error_function
    )
    
    # Test execution
    result = await agent_instruction.execute({})
    assert result.status == "error"
    assert "Test error" in result.error_message
    assert result.has_recovery_options

@pytest.mark.asyncio
async def test_instruction_composition(agentic_transformer):
    """Test composition of transformed functions."""
    # Create component functions
    async def func1(x: int) -> int:
        return x * 2
        
    async def func2(y: int) -> int:
        return y + 5
    
    # Transform individual functions
    instr1 = agentic_transformer.transform(func1)
    instr2 = agentic_transformer.transform(func2)
    
    # Compose instructions
    composed = agentic_transformer.compose_instructions([instr1, instr2])
    
    # Test execution
    result = await composed.execute({"x": 3})
    assert result.value == 11  # (3 * 2) + 5

@pytest.mark.asyncio
async def test_performance_tracking(agentic_transformer, sample_function):
    """Test performance tracking in transformed functions."""
    # Transform with performance tracking
    agent_instruction = agentic_transformer.transform_with_tracking(sample_function)
    
    # Execute multiple times
    for _ in range(3):
        await agent_instruction.execute({"x": 1, "y": 2})
    
    # Get performance metrics
    metrics = agent_instruction.get_performance_metrics()
    
    assert metrics.total_executions == 3
    assert metrics.average_latency > 0
    assert metrics.success_rate == 1.0 