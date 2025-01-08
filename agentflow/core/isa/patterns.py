"""Advanced pattern mining and analysis for instruction optimization."""
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .formal import FormalInstruction
from .analyzer import AnalysisResult

class PatternType(Enum):
    """Types of instruction patterns."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    COMPOSITIONAL = "compositional"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    RESOURCE = "resource"

@dataclass
class PatternMetrics:
    """Metrics for evaluating patterns."""
    frequency: int
    confidence: float
    support: float
    significance: float = 0.0
    lift: float = 1.0
    conviction: float = 1.0
    leverage: float = 0.0
    coverage: float = 1.0
    stability: float = 1.0
    exec_time_impact: float = 0.0  # Impact on execution time
    resource_impact: float = 0.0  # Impact on resource usage

@dataclass
class Pattern:
    """Represents an instruction pattern."""
    type: PatternType
    instructions: List[FormalInstruction]
    metrics: PatternMetrics
    context: Dict[str, Any] = None
    constraints: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __len__(self):
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)

class PatternMiner:
    """Advanced pattern mining system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_support = config.get("min_support", 0.1)
        self.min_confidence = config.get("min_confidence", 0.8)
        self.max_pattern_length = config.get("max_pattern_length", 10)
        self.temporal_window = config.get("temporal_window", 5)
        self.patterns: Dict[str, Pattern] = {}
        
    def mine_patterns(self,
                     instructions: List[FormalInstruction],
                     analysis: AnalysisResult) -> List[Pattern]:
        """Mine patterns from instructions."""
        patterns = []
        
        # Mine different types of patterns
        patterns.extend(self._mine_sequential_patterns(instructions, analysis))
        patterns.extend(self._mine_parallel_patterns(instructions, analysis))
        patterns.extend(self._mine_conditional_patterns(instructions, analysis))
        patterns.extend(self._mine_iterative_patterns(instructions, analysis))
        patterns.extend(self._mine_compositional_patterns(instructions, analysis))
        patterns.extend(self._mine_temporal_patterns(instructions, analysis))
        patterns.extend(self._mine_behavioral_patterns(instructions, analysis))
        patterns.extend(self._mine_resource_patterns(instructions, analysis))
        
        # Filter and rank patterns
        filtered_patterns = self._filter_patterns(patterns)
        ranked_patterns = self._rank_patterns(filtered_patterns)
        
        return ranked_patterns
    
    def _mine_sequential_patterns(self,
                                instructions: List[FormalInstruction],
                                analysis: AnalysisResult) -> List[Pattern]:
        """Mine sequential instruction patterns."""
        patterns = []
        n = len(instructions)
        
        # Use sliding window to find sequences
        for length in range(2, min(self.max_pattern_length, n + 1)):
            for i in range(n - length + 1):
                sequence = instructions[i:i+length]
                
                # Check if sequence forms a pattern
                if self._is_valid_sequence(sequence, analysis):
                    pattern = self._create_pattern(
                        PatternType.SEQUENTIAL,
                        sequence,
                        analysis
                    )
                    patterns.append(pattern)
                    
        return patterns
    
    def _mine_parallel_patterns(self,
                              instructions: List[FormalInstruction],
                              analysis: AnalysisResult) -> List[Pattern]:
        """Mine parallel instruction patterns."""
        patterns = []
        
        # Group independent instructions
        groups = self._group_independent_instructions(instructions)
        
        for group in groups:
            if len(group) > 1:
                pattern = self._create_pattern(
                    PatternType.PARALLEL,
                    group,
                    analysis
                )
                patterns.append(pattern)
                
        return patterns
    
    def _mine_conditional_patterns(self,
                                 instructions: List[FormalInstruction],
                                 analysis: AnalysisResult) -> List[Pattern]:
        """Mine conditional instruction patterns."""
        patterns = []
        
        for i, instr in enumerate(instructions[:-1]):
            # Look for condition-action pairs
            if self._is_conditional(instr):
                consequent = instructions[i+1]
                if self._are_conditionally_related(instr, consequent):
                    pattern = self._create_pattern(
                        PatternType.CONDITIONAL,
                        [instr, consequent],
                        analysis
                    )
                    patterns.append(pattern)
                    
        return patterns
    
    def _mine_iterative_patterns(self,
                               instructions: List[FormalInstruction],
                               analysis: AnalysisResult) -> List[Pattern]:
        """Mine iterative instruction patterns."""
        patterns = []
        
        # Find repeated sequences
        for length in range(2, self.max_pattern_length):
            sequences = self._find_repeated_sequences(
                instructions,
                length
            )
            
            for sequence in sequences:
                pattern = self._create_pattern(
                    PatternType.ITERATIVE,
                    sequence,
                    analysis
                )
                patterns.append(pattern)
                
        return patterns
    
    def _mine_compositional_patterns(self,
                                   instructions: List[FormalInstruction],
                                   analysis: AnalysisResult) -> List[Pattern]:
        """Mine compositional instruction patterns."""
        patterns = []
        
        # Find instruction compositions
        compositions = self._find_compositions(instructions)
        
        for comp in compositions:
            pattern = self._create_pattern(
                PatternType.COMPOSITIONAL,
                comp,
                analysis
            )
            patterns.append(pattern)
            
        return patterns
    
    def _mine_temporal_patterns(self,
                              instructions: List[FormalInstruction],
                              analysis: AnalysisResult) -> List[Pattern]:
        """Mine temporal instruction patterns."""
        patterns = []
        
        # Use sliding window for temporal analysis
        window = self.temporal_window
        for i in range(len(instructions) - window + 1):
            temporal_group = instructions[i:i+window]
            
            if self._has_temporal_pattern(temporal_group):
                pattern = self._create_pattern(
                    PatternType.TEMPORAL,
                    temporal_group,
                    analysis
                )
                patterns.append(pattern)
                
        return patterns
    
    def _mine_behavioral_patterns(self,
                                instructions: List[FormalInstruction],
                                analysis: AnalysisResult) -> List[Pattern]:
        """Mine behavioral instruction patterns."""
        patterns = []
        
        # Group instructions by behavior
        behavior_groups = self._group_by_behavior(instructions)
        
        for group in behavior_groups:
            pattern = self._create_pattern(
                PatternType.BEHAVIORAL,
                group,
                analysis
            )
            patterns.append(pattern)
            
        return patterns
    
    def _mine_resource_patterns(self,
                              instructions: List[FormalInstruction],
                              analysis: AnalysisResult) -> List[Pattern]:
        """Mine resource usage patterns."""
        patterns = []
        
        # Group instructions by resource usage
        resource_groups = self._group_by_resource_usage(instructions)
        
        for group in resource_groups:
            pattern = self._create_pattern(
                PatternType.RESOURCE,
                group,
                analysis
            )
            patterns.append(pattern)
            
        return patterns
    
    def _create_pattern(self,
                       type: PatternType,
                       instructions: List[FormalInstruction],
                       analysis: AnalysisResult) -> Pattern:
        """Create a pattern with metrics."""
        context = self._extract_context(instructions, analysis)
        constraints = self._extract_constraints(instructions, analysis)
        metrics = self._calculate_metrics(instructions, analysis)
        metadata = self._extract_metadata(instructions, analysis)
        
        return Pattern(
            type=type,
            instructions=instructions,
            context=context if context else None,
            constraints=constraints if constraints else None,
            metrics=metrics,
            metadata=metadata if metadata else None
        )
    
    def _calculate_metrics(self,
                         instructions: List[FormalInstruction],
                         analysis: AnalysisResult) -> PatternMetrics:
        """Calculate pattern metrics."""
        return PatternMetrics(
            frequency=self._calculate_frequency(instructions),
            confidence=self._calculate_confidence(instructions),
            support=self._calculate_support(instructions),
            significance=self._calculate_significance(instructions),
            lift=self._calculate_lift(instructions),
            conviction=self._calculate_conviction(instructions),
            leverage=self._calculate_leverage(instructions),
            coverage=self._calculate_coverage(instructions),
            stability=self._calculate_stability(instructions),
            exec_time_impact=self._calculate_exec_time_impact(instructions),
            resource_impact=self._calculate_resource_impact(instructions)
        )
    
    def _filter_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Filter patterns based on metrics."""
        return [
            pattern for pattern in patterns
            if self._is_significant_pattern(pattern)
        ]
    
    def _rank_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Rank patterns by importance."""
        return sorted(
            patterns,
            key=lambda p: self._calculate_importance(p),
            reverse=True
        )
    
    def _calculate_importance(self, pattern: Pattern) -> float:
        """Calculate pattern importance score."""
        return (
            pattern.metrics.support *
            pattern.metrics.confidence *
            pattern.metrics.lift *
            pattern.metrics.stability *
            (1 - pattern.metrics.exec_time_impact) *
            (1 - pattern.metrics.resource_impact)
        )
