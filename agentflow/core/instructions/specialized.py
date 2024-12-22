"""Specialized instruction implementations"""

from typing import Dict, Any, List, Optional
import logging
from .base import OptimizableInstruction, CacheableInstruction
import ray
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqGeneration
import ast
import astroid
from radon.complexity import cc_visit
from radon.metrics import h_visit
import numpy as np

logger = logging.getLogger(__name__)

@ray.remote
class TextSummarizationInstruction(OptimizableInstruction):
    """Summarize text content"""
    
    def __init__(self):
        super().__init__(
            name="summarize_text",
            description="Generate concise summary of text content"
        )
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(self.model_name)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
        
        # Add optimization rules
        self.add_optimization_rule(self._should_chunk_text)
        self.add_optimization_rule(self._should_use_extractive)
    
    def _should_chunk_text(self, context: Dict[str, Any]) -> bool:
        """Check if text should be chunked"""
        text = context.get("text", "")
        return len(text.split()) > 500
    
    def _should_use_extractive(self, context: Dict[str, Any]) -> bool:
        """Check if extractive summarization should be used"""
        text = context.get("text", "")
        return len(text.split()) > 1000
    
    async def _optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization strategies"""
        text = context.get("text", "")
        
        if self._should_chunk_text(context):
            # Split into chunks with overlap
            words = text.split()
            chunk_size = 400
            overlap = 50
            chunks = []
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)
            
            context["text_chunks"] = chunks
            context["use_chunks"] = True
        
        if self._should_use_extractive(context):
            context["use_extractive"] = True
        
        return context
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute summarization"""
        if context.get("use_chunks", False):
            # Summarize chunks and combine
            chunks = context["text_chunks"]
            summaries = []
            
            for chunk in chunks:
                summary = self.summarizer(chunk, max_length=130, min_length=30)[0]["summary_text"]
                summaries.append(summary)
            
            # Combine chunk summaries
            final_summary = " ".join(summaries)
            if len(final_summary.split()) > 150:
                # Re-summarize if too long
                final_summary = self.summarizer(final_summary, max_length=130, min_length=30)[0]["summary_text"]
        else:
            text = context.get("text", "")
            final_summary = self.summarizer(text, max_length=130, min_length=30)[0]["summary_text"]
        
        return {
            "summary": final_summary,
            "length": len(final_summary.split())
        }

@ray.remote
class CodeAnalysisInstruction(CacheableInstruction):
    """Analyze code structure and complexity"""
    
    def __init__(self):
        super().__init__(
            name="analyze_code",
            description="Analyze code structure, complexity, and quality"
        )
    
    def _analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity"""
        try:
            # Parse code
            tree = ast.parse(code)
            
            # Calculate cyclomatic complexity
            cc_results = cc_visit(code)
            
            # Calculate Halstead metrics
            h_results = h_visit(code)
            
            # Analyze with astroid for more detailed info
            astroid_tree = astroid.parse(code)
            
            # Collect metrics
            metrics = {
                "cyclomatic_complexity": {
                    func.name: func.complexity
                    for func in cc_results
                },
                "halstead_metrics": {
                    "volume": h_results.total.volume,
                    "difficulty": h_results.total.difficulty,
                    "effort": h_results.total.effort
                },
                "structure_metrics": {
                    "num_functions": len([n for n in astroid_tree.nodes_of_class(astroid.FunctionDef)]),
                    "num_classes": len([n for n in astroid_tree.nodes_of_class(astroid.ClassDef)]),
                    "num_imports": len([n for n in astroid_tree.nodes_of_class(astroid.Import)])
                }
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {"error": str(e)}
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code analysis"""
        code = context.get("code", "")
        if not code:
            return {"error": "No code provided"}
        
        # Analyze code
        metrics = self._analyze_complexity(code)
        
        # Calculate quality score
        if "error" not in metrics:
            # Weight different factors
            cc_score = min(1.0, 10.0 / max(metrics["cyclomatic_complexity"].values(), default=1))
            h_score = min(1.0, 1000.0 / metrics["halstead_metrics"]["effort"])
            
            quality_score = (cc_score * 0.4 + h_score * 0.6)
            
            metrics["quality_score"] = quality_score
        
        return metrics

@ray.remote
class ABTestingInstruction(OptimizableInstruction):
    """A/B testing for instruction variants"""
    
    def __init__(self):
        super().__init__(
            name="ab_test",
            description="Run A/B tests on instruction variants"
        )
        self.variants = {}
        self.variant_stats = {}
    
    def register_variant(self, name: str, instruction: ray.actor.ActorHandle):
        """Register a variant for testing"""
        self.variants[name] = instruction
        self.variant_stats[name] = {
            "executions": 0,
            "success_rate": 0.0,
            "avg_latency": 0.0,
            "reward": 0.0
        }
    
    def _update_stats(self, variant: str, success: bool, latency: float, reward: float):
        """Update variant statistics"""
        stats = self.variant_stats[variant]
        n = stats["executions"]
        
        # Update using moving averages
        stats["executions"] += 1
        stats["success_rate"] = (stats["success_rate"] * n + float(success)) / (n + 1)
        stats["avg_latency"] = (stats["avg_latency"] * n + latency) / (n + 1)
        stats["reward"] = (stats["reward"] * n + reward) / (n + 1)
    
    def _select_variant(self) -> str:
        """Select variant using Thompson sampling"""
        if not self.variants:
            raise ValueError("No variants registered")
        
        # Calculate beta parameters for each variant
        beta_params = {}
        for name, stats in self.variant_stats.items():
            successes = stats["success_rate"] * stats["executions"]
            failures = stats["executions"] - successes
            # Add small constants to avoid zero parameters
            beta_params[name] = (successes + 1, failures + 1)
        
        # Sample from beta distributions
        samples = {
            name: np.random.beta(a, b)
            for name, (a, b) in beta_params.items()
        }
        
        # Select variant with highest sample
        return max(samples.items(), key=lambda x: x[1])[0]
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute A/B test"""
        if not self.variants:
            return {"error": "No variants registered"}
        
        # Select variant
        selected_variant = self._select_variant()
        variant = self.variants[selected_variant]
        
        try:
            # Execute variant
            start_time = time.time()
            result = await variant.execute.remote(context)
            latency = time.time() - start_time
            
            # Calculate reward (example: combine success and speed)
            success = "error" not in result
            reward = success / (1 + latency)  # Simple reward function
            
            # Update statistics
            self._update_stats(selected_variant, success, latency, reward)
            
            return {
                "variant": selected_variant,
                "result": result,
                "metrics": {
                    "latency": latency,
                    "success": success,
                    "reward": reward
                }
            }
        except Exception as e:
            logger.error(f"Error executing variant {selected_variant}: {e}")
            self._update_stats(selected_variant, False, 0.0, 0.0)
            return {"error": str(e)}
