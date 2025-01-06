"""Test cases for advanced instruction set."""

import pytest
import asyncio
from agentflow.core.instructions.base import InstructionStatus
from agentflow.core.instructions import (
    AdvancedInstruction,
    ControlFlowInstruction,
    StateManagerInstruction,
    LLMInteractionInstruction,
    ResourceManagerInstruction,
    DataProcessingInstruction
)

@pytest.fixture
def basic_instruction():
    return AdvancedInstruction("test", "Test instruction")

@pytest.mark.asyncio
async def test_basic_instruction(basic_instruction):
    """Test basic instruction execution."""
    result = await basic_instruction.execute({})
    assert result.status == InstructionStatus.COMPLETED
    assert result.data.get("output") is None

@pytest.mark.asyncio
async def test_control_flow_sequential():
    """Test sequential control flow."""
    instruction = ControlFlowInstruction("seq", "Sequential flow", "sequential")
    comp1 = AdvancedInstruction("comp1", "Component 1")
    comp2 = AdvancedInstruction("comp2", "Component 2")
    instruction.add_component(comp1)
    instruction.add_component(comp2)
    
    result = await instruction.execute({})
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_control_flow_parallel():
    """Test parallel control flow."""
    instruction = ControlFlowInstruction("par", "Parallel flow", "parallel")
    comp1 = AdvancedInstruction("comp1", "Component 1")
    comp2 = AdvancedInstruction("comp2", "Component 2")
    instruction.add_component(comp1)
    instruction.add_component(comp2)
    
    result = await instruction.execute({})
    assert result.status == InstructionStatus.COMPLETED
    assert isinstance(result.data.get("output"), list)
    assert len(result.data.get("output")) == 2

@pytest.mark.asyncio
async def test_state_management():
    """Test state management."""
    instruction = StateManagerInstruction("state", "State manager")
    
    def handle_state(context):
        return {"processed": True}
    
    instruction.add_state("test_state", handle_state)
    result = await instruction.execute({"state": "test_state"})
    
    assert result.status == InstructionStatus.COMPLETED
    assert result.data.get("output") == {"processed": True}
    assert result.data.get("current_state") == "test_state"

@pytest.mark.asyncio
async def test_llm_interaction():
    """Test LLM interaction."""
    instruction = LLMInteractionInstruction("llm", "LLM interaction")
    instruction.set_model_config({"model": "test"})
    
    result = await instruction.execute({"prompt": "test prompt"})
    assert result.status == InstructionStatus.COMPLETED
    assert isinstance(result.data.get("output"), str)
    assert result.data.get("model_config") == {"model": "test"}

@pytest.mark.asyncio
async def test_resource_management():
    """Test resource management."""
    instruction = ResourceManagerInstruction("resource", "Resource manager")
    instruction.set_resource_limits({"memory": 0.9})
    
    result = await instruction.execute({})
    assert result.status == InstructionStatus.COMPLETED
    assert "resource_usage" in result.data
    assert "memory" in result.data["resource_usage"]

@pytest.mark.asyncio
async def test_precondition_postcondition():
    """Test pre/post conditions."""
    instruction = AdvancedInstruction("test", "Test instruction")
    
    def check_input(context):
        return "input" in context
    
    def check_output(context):
        return "output" in context
    
    instruction.add_precondition(check_input)
    instruction.add_postcondition(check_output)
    
    # Should fail precondition
    result = await instruction.execute({})
    assert result.status == InstructionStatus.FAILED
    
    # Should succeed
    result = await instruction.execute({"input": "test", "output": "test"})
    assert result.status == InstructionStatus.COMPLETED