"""Adaptive instruction learning and optimization module."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict

from agentflow.core.agentic_calling import AgentInstruction, FunctionContext, ExecutionResult
from agentflow.core.optimization import OptimizationMetrics

@dataclass
class InstructionPattern:
    """Pattern discovered in instruction execution."""
    sequence: List[str]
    frequency: int
    success_rate: float
    avg_latency: float
    context_dependencies: Set[str]

@dataclass
class AdaptiveMetrics:
    """Metrics for adaptive learning."""
    pattern_discoveries: int = 0
    successful_adaptations: int = 0
    failed_adaptations: int = 0
    total_optimizations: int = 0
    learning_cycles: int = 0

class PatternMatcher:
    """Pattern matching for instruction sequences."""
    
    def __init__(self, min_frequency: int = 3, min_success_rate: float = 0.8):
        self.min_frequency = min_frequency
        self.min_success_rate = min_success_rate
        self.sequence_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.sequence_metrics: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)
    
    def add_sequence(
        self,
        sequence: List[str],
        success: bool,
        latency: float,
        context_vars: Set[str]
    ):
        """Add a sequence observation."""
        seq_tuple = tuple(sequence)
        self.sequence_counts[seq_tuple] += 1
        
        # Update metrics
        metrics = self.sequence_metrics[seq_tuple]
        metrics["successes"] = metrics.get("successes", 0) + (1 if success else 0)
        metrics["total"] = metrics.get("total", 0) + 1
        metrics["total_latency"] = metrics.get("total_latency", 0) + latency
        metrics["context_vars"] = metrics.get("context_vars", set()) | context_vars
    
    def find_patterns(self) -> List[InstructionPattern]:
        """Find significant patterns in sequences."""
        patterns = []
        
        for seq_tuple, count in self.sequence_counts.items():
            if count >= self.min_frequency:
                metrics = self.sequence_metrics[seq_tuple]
                success_rate = metrics["successes"] / metrics["total"]
                
                if success_rate >= self.min_success_rate:
                    patterns.append(InstructionPattern(
                        sequence=list(seq_tuple),
                        frequency=count,
                        success_rate=success_rate,
                        avg_latency=metrics["total_latency"] / metrics["total"],
                        context_dependencies=metrics["context_vars"]
                    ))
        
        return sorted(patterns, key=lambda p: p.frequency * p.success_rate, reverse=True)

class AdaptiveLearner:
    """Adaptive learning for instruction optimization."""
    
    def __init__(self, min_frequency: int = 1, min_success_rate: float = 0.8):
        self.pattern_matcher = PatternMatcher(min_frequency=min_frequency, min_success_rate=min_success_rate)
        self.learned_patterns: Dict[str, InstructionPattern] = {}
        self.instruction_cache: Dict[str, AgentInstruction] = {}
        self.metrics = AdaptiveMetrics()
        self.optimization_history: List[Dict[str, Any]] = []
    
    async def learn_from_execution(
        self,
        instruction: AgentInstruction,
        result: ExecutionResult,
        context: FunctionContext
    ):
        """Learn from instruction execution."""
        # Extract execution information
        success = result.status == "success"
        latency = result.metrics.get("execution_time", 0.0) if result.metrics else 0.0
        context_vars = set(context.variables.keys())
        
        # Add to pattern matcher
        self.pattern_matcher.add_sequence(
            [instruction.name],
            success,
            latency,
            context_vars
        )
        
        # Update metrics
        self.metrics.pattern_discoveries += 1
        
        # Discover new patterns
        await self._discover_patterns()
    
    async def optimize_instruction(
        self,
        instruction: AgentInstruction
    ) -> Optional[AgentInstruction]:
        """Optimize instruction based on learned patterns."""
        try:
            # Find applicable patterns
            patterns = [
                p for p in self.learned_patterns.values()
                if instruction.name in p.sequence
            ]
            
            if not patterns:
                return None
            
            # Select best pattern
            best_pattern = max(patterns, key=lambda p: p.success_rate * (1 / p.avg_latency))
            
            # Create optimized instruction
            optimized = await self._create_optimized_instruction(instruction, best_pattern)
            
            # Update metrics
            self.metrics.successful_adaptations += 1
            self.metrics.total_optimizations += 1
            
            return optimized
        except Exception as e:
            self.metrics.failed_adaptations += 1
            return None
    
    async def _discover_patterns(self):
        """Discover new instruction patterns."""
        patterns = self.pattern_matcher.find_patterns()
        
        for pattern in patterns:
            pattern_id = "_".join(pattern.sequence)
            if pattern_id not in self.learned_patterns:
                self.learned_patterns[pattern_id] = pattern
                
                # Record optimization opportunity
                self.optimization_history.append({
                    "timestamp": datetime.now(),
                    "pattern": pattern_id,
                    "frequency": pattern.frequency,
                    "success_rate": pattern.success_rate,
                    "avg_latency": pattern.avg_latency
                })
        
        self.metrics.learning_cycles += 1
    
    async def _create_optimized_instruction(
        self,
        instruction: AgentInstruction,
        pattern: InstructionPattern
    ) -> AgentInstruction:
        """Create an optimized version of the instruction."""
        # Create optimization hints
        hints = {
            "pattern_optimized": True,
            "original_name": instruction.name,
            "pattern_sequence": pattern.sequence,
            "expected_latency": pattern.avg_latency,
            "context_dependencies": list(pattern.context_dependencies)
        }
        
        # Create optimized function
        async def optimized_func(**kwargs):
            context = kwargs.get("context", FunctionContext({}, {}, {}))
            
            # Verify context dependencies
            missing_deps = pattern.context_dependencies - set(context.variables.keys())
            if missing_deps:
                raise ValueError(f"Missing context dependencies: {missing_deps}")
            
            # Execute with optimization
            return await instruction.func(**kwargs)
        
        # Create new instruction
        optimized = AgentInstruction(
            name=f"{instruction.name}_optimized",
            func=optimized_func,
            parameters=instruction.parameters,
            optimization_hints=hints
        )
        
        # Cache optimized instruction
        self.instruction_cache[instruction.name] = optimized
        
        return optimized

class AdaptiveOptimizer:
    """Optimizer that uses adaptive learning."""
    
    def __init__(self):
        self.learner = AdaptiveLearner()
        self.optimization_metrics = OptimizationMetrics()
        self.active_optimizations: Dict[str, AgentInstruction] = {}
        self.optimization_history: List[Dict[str, Any]] = []
    
    async def optimize_instruction_sequence(
        self,
        instructions: List[AgentInstruction]
    ) -> List[AgentInstruction]:
        """Optimize a sequence of instructions."""
        start_time = datetime.now()
        optimized_instructions = []
        
        for instruction in instructions:
            # Try to optimize instruction
            optimized = await self.learner.optimize_instruction(instruction)
            
            if optimized:
                optimized_instructions.append(optimized)
                self.active_optimizations[instruction.name] = optimized
                
                # Record optimization
                self.optimization_history.append({
                    "timestamp": datetime.now(),
                    "instruction": instruction.name,
                    "optimized_name": optimized.name,
                    "optimization_hints": optimized.optimization_hints
                })
            else:
                optimized_instructions.append(instruction)
        
        # Update metrics
        self.optimization_metrics.optimization_time = (
            datetime.now() - start_time
        ).total_seconds()
        self.optimization_metrics.optimization_success = True
        
        return optimized_instructions
    
    async def learn_from_sequence(
        self,
        instructions: List[AgentInstruction],
        results: List[ExecutionResult],
        contexts: List[FunctionContext]
    ):
        """Learn from a sequence of instruction executions."""
        for instruction, result, context in zip(instructions, results, contexts):
            await self.learner.learn_from_execution(instruction, result, context)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "total_patterns": len(self.learner.learned_patterns),
            "active_optimizations": len(self.active_optimizations),
            "learning_cycles": self.learner.metrics.learning_cycles,
            "optimization_history": self.optimization_history,
            "metrics": {
                "pattern_discoveries": self.learner.metrics.pattern_discoveries,
                "successful_adaptations": self.learner.metrics.successful_adaptations,
                "failed_adaptations": self.learner.metrics.failed_adaptations,
                "total_optimizations": self.learner.metrics.total_optimizations
            }
        }
    
    def reset_optimization_state(self):
        """Reset the optimization state."""
        self.active_optimizations.clear()
        self.optimization_history.clear()
        self.optimization_metrics = OptimizationMetrics()
        self.learner = AdaptiveLearner() 