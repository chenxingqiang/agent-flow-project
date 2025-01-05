"""Tests for instruction set architecture functionality."""

import pytest
import asyncio
from typing import Dict, Any
from agentflow.core.isa.isa_manager import (
    ISAManager, Instruction, InstructionType,
    InstructionStatus, InstructionResult
)

@pytest.fixture
def sample_instruction_set():
    """Create a sample instruction set."""
    return [
        Instruction(
            id="control_flow",
            name="sequence",
            type=InstructionType.CONTROL,
            params={"init_param": "value"},
            description="Control flow instruction"
        ),
        Instruction(
            id="state_mgmt",
            name="store",
            type=InstructionType.STATE,
            params={"key": "value"},
            description="State management instruction"
        ),
        Instruction(
            id="llm_interact",
            name="query",
            type=InstructionType.LLM,
            params={"prompt": "test"},
            description="LLM interaction instruction"
        ),
        Instruction(
            id="resource_mgmt",
            name="allocate",
            type=InstructionType.RESOURCE,
            params={"resource": "memory"},
            description="Resource management instruction"
        )
    ]

@pytest.fixture
def isa_manager(sample_instruction_set):
    """Create an ISA manager instance."""
    manager = ISAManager()
    for instruction in sample_instruction_set:
        manager.register_instruction(instruction)
    return manager

@pytest.mark.asyncio
async def test_instruction_registration(isa_manager, sample_instruction_set):
    """Test instruction registration and retrieval."""
    # Verify all instructions are registered
    for instruction in sample_instruction_set:
        retrieved = isa_manager.get_instruction(instruction.id)
        assert retrieved.id == instruction.id
        assert retrieved.type == instruction.type
        assert retrieved.params == instruction.params

@pytest.mark.asyncio
async def test_instruction_composition(isa_manager):
    """Test instruction composition."""
    # Create composite instruction
    composite = isa_manager.compose_instructions([
        "control_flow",
        "state_mgmt"
    ], "composite_test")
    
    assert composite.id == "composite_test"
    assert len(composite.steps) == 2
    assert composite.steps[0].id == "control_flow"
    assert composite.steps[1].id == "state_mgmt"

@pytest.mark.asyncio
async def test_instruction_execution(isa_manager):
    """Test instruction execution."""
    context = {"test": "value"}
    
    # Execute state management instruction
    result = await isa_manager.execute_instruction(
        isa_manager.get_instruction("state_mgmt"),
        context
    )
    
    assert result.status == InstructionStatus.SUCCESS
    assert "key" in result.context
    assert result.context["key"] == "value"

@pytest.mark.asyncio
async def test_instruction_optimization(isa_manager):
    """Test instruction optimization."""
    # Create a sequence of instructions
    sequence = [
        isa_manager.get_instruction("state_mgmt"),
        isa_manager.get_instruction("state_mgmt"),
        isa_manager.get_instruction("llm_interact")
    ]
    
    # Optimize sequence
    optimized = isa_manager.optimize_sequence(sequence)
    
    # Should combine duplicate state management instructions
    assert len(optimized) < len(sequence)
    assert optimized[-1].id == "llm_interact"

@pytest.mark.asyncio
async def test_instruction_verification(isa_manager):
    """Test instruction verification."""
    instruction = isa_manager.get_instruction("control_flow")
    
    # Verify instruction validity
    assert isa_manager.verify_instruction(instruction)
    
    # Test invalid instruction
    invalid_instruction = Instruction(
        id="invalid",
        name="invalid",
        type=InstructionType.CONTROL,
        params={},  # Missing required params
        description="Invalid instruction"
    )
    
    assert not isa_manager.verify_instruction(invalid_instruction)

@pytest.mark.asyncio
async def test_instruction_pipeline(isa_manager):
    """Test instruction pipeline execution."""
    pipeline = isa_manager.create_pipeline([
        "control_flow",
        "state_mgmt",
        "llm_interact"
    ])
    
    context = {"input": "test"}
    result = await pipeline.execute(context)
    
    assert result.status == InstructionStatus.SUCCESS
    assert len(result.metrics) > 0
    assert "execution_time" in result.metrics

@pytest.mark.asyncio
async def test_instruction_error_handling(isa_manager):
    """Test instruction error handling."""
    # Create failing instruction
    failing_instruction = Instruction(
        id="fail",
        name="fail",
        type=InstructionType.CONTROL,
        params={"should_fail": True},
        description="Failing instruction"
    )
    
    isa_manager.register_instruction(failing_instruction)
    
    # Execute with error handling
    result = await isa_manager.execute_instruction(
        failing_instruction,
        {},
    )
    
    assert result.status == InstructionStatus.SUCCESS
    assert result.metrics is not None

@pytest.mark.asyncio
async def test_instruction_metrics(isa_manager):
    """Test instruction metrics collection."""
    instruction = isa_manager.get_instruction("llm_interact")
    
    # Execute instruction multiple times
    for _ in range(3):
        await isa_manager.execute_instruction(instruction, {})
    
    metrics = isa_manager.get_metrics()
    
    assert "total_instructions" in metrics
    assert metrics["total_instructions"] > 0

@pytest.mark.asyncio
async def test_cross_model_compatibility(isa_manager):
    """Test cross-model instruction compatibility."""
    instruction = isa_manager.get_instruction("llm_interact")
    
    # Test with different model configurations
    models = ["gpt-4", "claude-v1", "llama-2"]
    
    for model in models:
        context = {"model": model}
        result = await isa_manager.execute_instruction(instruction, context)
        assert result.status == InstructionStatus.SUCCESS
        
@pytest.mark.asyncio
async def test_instruction_learning(isa_manager):
    """Test automatic instruction learning."""
    # Simulate user interaction patterns
    interactions = [
        {"input": "research topic", "actions": ["control_flow", "llm_interact", "state_mgmt"]},
        {"input": "process data", "actions": ["state_mgmt", "resource_mgmt", "state_mgmt"]},
        {"input": "evaluate model", "actions": ["llm_interact", "state_mgmt", "resource_mgmt"]}
    ]
    
    # Learn patterns
    learned_instructions = isa_manager.learn_patterns(interactions)
    
    assert len(learned_instructions) > 0
    for instruction in learned_instructions:
        assert isa_manager.verify_instruction(instruction)
        assert instruction.type == InstructionType.CONTROL 