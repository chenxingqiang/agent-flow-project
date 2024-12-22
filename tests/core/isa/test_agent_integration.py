"""Tests for ISA integration with Agent."""
import pytest
from agentflow.core.agent import Agent, AgentConfig
from agentflow.core.isa.isa_manager import ISAManager
from agentflow.core.isa.formal import FormalInstruction
from agentflow.core.isa.analyzer import AnalysisResult

@pytest.fixture
def sample_config():
    """Create sample agent configuration."""
    return {
        "isa_config_path": "test_isa_config.json",
        "rl_algorithm": "PPO",
        "max_steps": 100
    }

@pytest.fixture
def sample_instructions():
    """Create sample instructions for testing."""
    return [
        FormalInstruction(id="1", name="init", params={"x": 1}),
        FormalInstruction(id="2", name="process", params={"data": "test"}),
        FormalInstruction(id="3", name="validate", params={"check": True}),
    ]

class TestAgentISAIntegration:
    """Test ISA integration with Agent class."""
    
    def test_isa_manager_initialization(self):
        """Test ISA manager is properly initialized in Agent."""
        config = AgentConfig(sample_config())
        agent = Agent(config)
        
        assert hasattr(agent, 'isa_manager')
        assert isinstance(agent.isa_manager, ISAManager)
        
    def test_instruction_loading(self, sample_instructions):
        """Test loading instructions into agent."""
        config = AgentConfig(sample_config())
        agent = Agent(config)
        
        # Add instructions through ISA manager
        for instruction in sample_instructions:
            agent.isa_manager.add_instruction(instruction)
            
        assert len(agent.isa_manager.instructions) == len(sample_instructions)
        
    def test_instruction_selection(self, sample_instructions):
        """Test instruction selection based on input."""
        config = AgentConfig(sample_config())
        agent = Agent(config)
        
        # Add instructions
        for instruction in sample_instructions:
            agent.isa_manager.add_instruction(instruction)
            
        # Test instruction selection
        input_data = {"task": "process data", "params": {"data": "test"}}
        selected = agent.instruction_selector.select(input_data)
        
        assert isinstance(selected, list)
        assert all(isinstance(instr, FormalInstruction) for instr in selected)
        
    def test_instruction_execution(self, sample_instructions):
        """Test executing selected instructions."""
        config = AgentConfig(sample_config())
        agent = Agent(config)
        
        # Add instructions
        for instruction in sample_instructions:
            agent.isa_manager.add_instruction(instruction)
            
        # Select and execute instructions
        input_data = {"task": "init and process", "params": {"x": 1, "data": "test"}}
        selected = agent.instruction_selector.select(input_data)
        
        for instruction in selected:
            result = agent.isa_manager.execute_instruction(instruction)
            assert result is not None
            
    def test_error_handling(self):
        """Test error handling in ISA integration."""
        config = AgentConfig(sample_config())
        agent = Agent(config)
        
        # Test invalid instruction
        invalid_instruction = FormalInstruction(id="invalid", name="nonexistent", params={})
        with pytest.raises(ValueError):
            agent.isa_manager.execute_instruction(invalid_instruction)
            
        # Test invalid config path
        config.isa_config_path = "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            agent._load_isa_instructions()
            
    def test_instruction_optimization(self, sample_instructions):
        """Test RL-based instruction optimization."""
        config = AgentConfig(sample_config())
        agent = Agent(config)
        
        # Add instructions
        for instruction in sample_instructions:
            agent.isa_manager.add_instruction(instruction)
            
        # Test optimization
        input_data = {"task": "complex task", "params": {"x": 1, "data": "test"}}
        initial_sequence = agent.instruction_selector.select(input_data)
        
        # Optimize sequence
        optimized = agent.rl_optimizer.optimize(initial_sequence, input_data)
        
        assert isinstance(optimized, list)
        assert len(optimized) > 0
        assert all(isinstance(instr, FormalInstruction) for instr in optimized)
