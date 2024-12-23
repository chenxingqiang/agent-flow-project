"""Test agent integration with ISA."""

import pytest
import os
from agentflow.core.agent import Agent, AgentConfig
from agentflow.core.isa.isa_manager import Instruction, InstructionType

@pytest.fixture
def sample_config():
    """Sample agent configuration."""
    return {
        "isa_config_path": os.path.join(os.path.dirname(__file__), "test_isa_config.json"),
        "max_steps": 100,
        "rl_algorithm": "PPO"
    }

@pytest.fixture
def sample_instructions():
    """Sample instructions for testing."""
    return [
        Instruction(
            id="init",
            name="initialize",
            type=InstructionType.CONTROL,
            params={"init_param": "value"},
            description="Initialize system"
        ),
        Instruction(
            id="process",
            name="process_data",
            type=InstructionType.COMPUTATION,
            params={"data_param": "value"},
            description="Process input data"
        ),
        Instruction(
            id="validate",
            name="validate_result",
            type=InstructionType.VALIDATION,
            params={"threshold": 0.9},
            description="Validate results"
        )
    ]

class TestAgentISAIntegration:
    """Test agent integration with ISA."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, sample_config):
        """Test agent initialization with ISA."""
        config = AgentConfig(**sample_config)
        agent = Agent(config)
        await agent.initialize()
        
        assert agent.isa_manager is not None
        assert agent.instruction_selector is not None
        
    @pytest.mark.asyncio
    async def test_instruction_selection(self, sample_config):
        """Test instruction selection based on input."""
        config = AgentConfig(**sample_config)
        agent = Agent(config)
        await agent.initialize()
        
        # Test instruction selection
        input_data = {"task": "process data", "params": {"data": "test"}, "text": "process data"}
        selected = agent.instruction_selector.select_instructions(input_data, agent.isa_manager.instructions)
        
        assert isinstance(selected, list)
        assert all(isinstance(instr, str) for instr in selected)
        
    @pytest.mark.asyncio
    async def test_instruction_execution(self, sample_config):
        """Test executing selected instructions."""
        config = AgentConfig(**sample_config)
        agent = Agent(config)
        await agent.initialize()
        
        # Select and execute instructions
        input_data = {"task": "init and process", "params": {"x": 1, "data": "test"}, "text": "initialize and process data"}
        selected = agent.instruction_selector.select_instructions(input_data, agent.isa_manager.instructions)
        
        for instruction_id in selected:
            instruction = agent.isa_manager.get_instruction(instruction_id)
            result = agent.isa_manager.execute_instruction(instruction)
            assert result is not None
            
    @pytest.mark.asyncio
    async def test_error_handling(self, sample_config):
        """Test error handling in ISA integration."""
        config = AgentConfig(**sample_config)
        agent = Agent(config)
        await agent.initialize()
        
        # Test invalid instruction
        invalid_instruction = Instruction(
            id="invalid",
            name="nonexistent",
            type=InstructionType.CONTROL,
            params={}
        )
        with pytest.raises(ValueError):
            agent.isa_manager.execute_instruction(invalid_instruction)
            
        # Test invalid config path
        config.isa_config_path = "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            agent.isa_manager.load_instructions(config.isa_config_path)
            
    @pytest.mark.asyncio
    async def test_instruction_optimization(self, sample_config):
        """Test RL-based instruction optimization."""
        config = AgentConfig(**sample_config)
        agent = Agent(config)
        await agent.initialize()
        
        # Test optimization
        input_data = {"task": "complex task", "params": {"x": 1, "data": "test"}, "text": "complex task"}
        initial_sequence = agent.instruction_selector.select_instructions(input_data, agent.isa_manager.instructions)
        
        # Skip optimization if no optimizer is configured
        if agent.optimizer is None:
            return
            
        # Optimize sequence
        optimized = agent.optimizer.optimize(initial_sequence, input_data)
