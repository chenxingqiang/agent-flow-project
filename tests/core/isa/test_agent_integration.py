"""Test agent integration with ISA."""

import pytest
import os
from agentflow.agents.agent import Agent, AgentConfig
from agentflow.core.isa.isa_manager import Instruction, InstructionType
from agentflow.core.isa.types import AgentType

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "AGENT": {
            "name": "TestAgent",
            "type": AgentType.RESEARCH.value
        },
        "MODEL": {
            "name": "gpt-4",
            "provider": "openai"
        },
        "WORKFLOW": {
            "max_iterations": 5
        },
        "isa_config_path": "/Users/xingqiangchen/TASK/APOS/tests/core/isa/test_isa_config.json",
        "domain_config": {
            "instruction_set": ["research", "analyze", "summarize"],
            "optimization_strategy": "rl"
        }
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
        config = AgentConfig.from_dict(sample_config)
        agent = Agent(config)
        await agent.initialize()
        
        assert agent.isa_manager is not None
        assert agent.instruction_selector is not None
        
    @pytest.mark.asyncio
    async def test_instruction_selection(self, sample_config):
        """Test instruction selection based on input."""
        config = AgentConfig.from_dict(sample_config)
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
        config = AgentConfig.from_dict(sample_config)
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
        config = AgentConfig.from_dict(sample_config)
        agent = Agent(config)
        await agent.initialize()
        
        # Test invalid instruction handling
        with pytest.raises(ValueError):
            agent.isa_manager.get_instruction("nonexistent")
            
        # Test instruction execution error handling
        instruction = agent.isa_manager.get_instruction("init")
        instruction.params = {"invalid": "params"}
        with pytest.raises(Exception):
            agent.isa_manager.execute_instruction(instruction)
            
    @pytest.mark.asyncio
    async def test_instruction_optimization(self, sample_config):
        """Test RL-based instruction optimization."""
        config = AgentConfig.from_dict(sample_config)
        agent = Agent(config)
        await agent.initialize()
        
        # Test optimization strategy
        input_data = {"task": "complex task", "params": {"data": "test"}, "text": "complex task requiring multiple steps"}
        selected = agent.instruction_selector.select_instructions(input_data, agent.isa_manager.instructions)
        
        # Execute instructions and collect rewards
        rewards = []
        for instruction_id in selected:
            instruction = agent.isa_manager.get_instruction(instruction_id)
            result = agent.isa_manager.execute_instruction(instruction)
            reward = agent.instruction_selector.get_reward(result)
            rewards.append(reward)
            
        assert all(isinstance(r, (int, float)) for r in rewards)
        assert len(rewards) == len(selected)
