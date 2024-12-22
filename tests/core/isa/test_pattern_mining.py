"""Tests for pattern mining functionality."""
import pytest
import numpy as np
from agentflow.core.isa.pattern_mining import (
    SequenceMiner,
    ParallelMiner,
    BehavioralMiner,
    FrequentSequence,
    DependencyGraph
)
from agentflow.core.isa.formal import FormalInstruction
from agentflow.core.isa.analyzer import AnalysisResult, AnalysisType

@pytest.fixture
def sample_instructions():
    """Create sample instructions for testing."""
    return [
        FormalInstruction(id="1", name="init", params={"x": 1}),
        FormalInstruction(id="2", name="process", params={"data": "test"}),
        FormalInstruction(id="3", name="validate", params={"check": True}),
        FormalInstruction(id="4", name="store", params={"key": "result"})
    ]

@pytest.fixture
def sample_analysis():
    """Create sample analysis result."""
    return AnalysisResult(
        type=AnalysisType.BEHAVIOR,
        metrics={"accuracy": 0.9, "f1_score": 0.8},
        insights=["High accuracy", "Low F1 score"],
        recommendations=["Improve model", "Collect more data"],
        confidence=0.8
    )

class TestSequenceMiner:
    """Test sequence mining functionality."""
    
    def test_find_sequences(self, sample_instructions):
        """Test finding frequent sequences."""
        config = {
            "min_support": 0.1,
            "min_confidence": 0.8
        }
        miner = SequenceMiner(config)
        sequences = miner.find_sequences(sample_instructions)
        
        assert isinstance(sequences, list)
        assert all(isinstance(seq, FrequentSequence) for seq in sequences)
        
        if sequences:
            seq = sequences[0]
            assert isinstance(seq.support, float)
            assert isinstance(seq.confidence, float)
            assert isinstance(seq.frequency, int)
            assert len(seq.instructions) >= 2
            
    def test_sequence_validation(self, sample_instructions):
        """Test sequence validation logic."""
        config = {"min_support": 0.5, "min_confidence": 0.9}
        miner = SequenceMiner(config)
        
        # Test with invalid sequence length
        with pytest.raises(ValueError):
            miner._generate_candidates(sample_instructions, 0)
            
        # Test with sequence length > instructions
        candidates = miner._generate_candidates(
            sample_instructions,
            len(sample_instructions) + 1
        )
        assert len(candidates) == 0

class TestParallelMiner:
    """Test parallel mining functionality."""
    
    def test_find_parallel_groups(self, sample_instructions):
        """Test finding parallel instruction groups."""
        config = {}
        miner = ParallelMiner(config)
        groups = miner.find_parallel_groups(sample_instructions)
        
        assert isinstance(groups, list)
        for group in groups:
            assert isinstance(group, list)
            assert all(isinstance(instr, FormalInstruction) for instr in group)
            
    def test_dependency_graph(self, sample_instructions):
        """Test dependency graph construction."""
        config = {}
        miner = ParallelMiner(config)
        graph = miner._build_dependency_graph(sample_instructions)
        
        assert isinstance(graph, DependencyGraph)
        assert len(graph.nodes) == len(sample_instructions)
        assert isinstance(graph.edges, list)
        assert isinstance(graph.weights, dict)
        
        # Verify node IDs
        for i, instr in enumerate(sample_instructions):
            assert str(i) in graph.nodes
            
class TestBehavioralMiner:
    """Test behavioral mining functionality."""
    
    def test_find_behavioral_patterns(self, sample_instructions, sample_analysis):
        """Test finding behavioral patterns."""
        config = {}
        miner = BehavioralMiner(config)
        patterns = miner.find_behavioral_patterns(
            sample_instructions,
            sample_analysis
        )
        
        assert isinstance(patterns, list)
        
    def test_feature_extraction(self, sample_instructions, sample_analysis):
        """Test behavioral feature extraction."""
        config = {}
        miner = BehavioralMiner(config)
        features = miner._extract_behavioral_features(
            sample_instructions,
            sample_analysis
        )
        
        assert isinstance(features, dict)
        assert len(features) == len(sample_instructions)
        
        # Verify feature vectors
        for instr_id, feature_vector in features.items():
            assert isinstance(feature_vector, np.ndarray)
            assert feature_vector.ndim == 1
            
    def test_behavior_clustering(self, sample_instructions, sample_analysis):
        """Test behavior clustering."""
        config = {}
        miner = BehavioralMiner(config)
        
        # Extract features
        features = miner._extract_behavioral_features(
            sample_instructions,
            sample_analysis
        )
        
        # Test clustering
        clusters = miner._cluster_behaviors(features)
        assert isinstance(clusters, list)
        
        # Verify cluster contents
        all_instrs = set()
        for cluster in clusters:
            assert isinstance(cluster, list)
            all_instrs.update(cluster)
            
        # Verify all instructions are clustered
        assert len(all_instrs) == len(sample_instructions)
