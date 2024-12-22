"""Tests for behavioral pattern analysis in ISA."""
import pytest
from agentflow.core.isa.pattern_mining import BehavioralMiner
from agentflow.core.isa.formal import FormalInstruction
from agentflow.core.isa.analyzer import AnalysisResult

@pytest.fixture
def sample_instructions():
    """Create sample instructions with behavioral patterns."""
    return [
        FormalInstruction(id="1", name="init", params={"x": 1}),
        FormalInstruction(id="2", name="load_data", params={"source": "db"}),
        FormalInstruction(id="3", name="validate", params={"check": True}),
        FormalInstruction(id="4", name="process", params={"data": "test"}),
        FormalInstruction(id="5", name="save", params={"destination": "db"}),
        # Repeated pattern
        FormalInstruction(id="6", name="load_data", params={"source": "api"}),
        FormalInstruction(id="7", name="validate", params={"check": True}),
        FormalInstruction(id="8", name="process", params={"data": "api_data"}),
        FormalInstruction(id="9", name="save", params={"destination": "db"})
    ]

class TestBehavioralPatterns:
    """Test behavioral pattern mining functionality."""
    
    def test_pattern_detection(self, sample_instructions):
        """Test detection of behavioral patterns."""
        config = {
            "min_pattern_length": 2,
            "min_support": 0.4,
            "similarity_threshold": 0.8
        }
        miner = BehavioralMiner(config)
        patterns = miner.find_patterns(sample_instructions)
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        # Should detect the load->validate->process->save pattern
        found_etl_pattern = False
        for pattern in patterns:
            if len(pattern) >= 4:
                steps = [instr.name for instr in pattern]
                if (steps[0] == "load_data" and 
                    steps[1] == "validate" and
                    steps[2] == "process" and
                    steps[3] == "save"):
                    found_etl_pattern = True
                    break
                    
        assert found_etl_pattern, "Failed to detect ETL pattern"
        
    def test_pattern_similarity(self, sample_instructions):
        """Test pattern similarity calculation."""
        config = {"similarity_threshold": 0.8}
        miner = BehavioralMiner(config)
        
        # Extract two similar patterns
        pattern1 = sample_instructions[1:5]  # load->validate->process->save (db)
        pattern2 = sample_instructions[5:9]  # load->validate->process->save (api)
        
        similarity = miner.calculate_pattern_similarity(pattern1, pattern2)
        assert similarity > 0.8, "Similar patterns should have high similarity score"
        
    def test_pattern_frequency(self, sample_instructions):
        """Test pattern frequency calculation."""
        config = {"min_pattern_length": 2}
        miner = BehavioralMiner(config)
        
        # Count frequency of validate->process pattern
        pattern = [
            FormalInstruction(id="1", name="validate", params={"check": True}),
            FormalInstruction(id="2", name="process", params={"data": "test"})
        ]
        
        frequency = miner.calculate_pattern_frequency(pattern, sample_instructions)
        assert frequency == 2, "validate->process pattern should occur twice"
        
    def test_pattern_significance(self, sample_instructions):
        """Test pattern significance calculation."""
        config = {
            "min_pattern_length": 2,
            "min_support": 0.4
        }
        miner = BehavioralMiner(config)
        patterns = miner.find_patterns(sample_instructions)
        
        for pattern in patterns:
            significance = miner.calculate_pattern_significance(pattern, sample_instructions)
            assert isinstance(significance, float)
            assert 0 <= significance <= 1
            
    def test_pattern_optimization(self, sample_instructions):
        """Test pattern-based sequence optimization."""
        config = {
            "min_pattern_length": 2,
            "optimization_metric": "execution_time"
        }
        miner = BehavioralMiner(config)
        
        # Optimize instruction sequence
        optimized = miner.optimize_sequence(sample_instructions)
        
        assert isinstance(optimized, list)
        assert len(optimized) <= len(sample_instructions)
        assert all(isinstance(instr, FormalInstruction) for instr in optimized)
        
    def test_invalid_patterns(self):
        """Test handling of invalid patterns."""
        config = {}
        miner = BehavioralMiner(config)
        
        # Test empty sequence
        patterns = miner.find_patterns([])
        assert len(patterns) == 0
        
        # Test single instruction
        patterns = miner.find_patterns([
            FormalInstruction(id="1", name="single", params={})
        ])
        assert len(patterns) == 0
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            BehavioralMiner({"min_pattern_length": 0})
