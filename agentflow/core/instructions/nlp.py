"""Natural Language Processing Instructions"""

from typing import Dict, Any, List
import logging
from .base import CacheableInstruction, OptimizableInstruction
import spacy
from transformers import pipeline
import ray

logger = logging.getLogger(__name__)

@ray.remote
class SentimentAnalysisInstruction(CacheableInstruction):
    """Analyze sentiment of text content"""
    
    def __init__(self):
        super().__init__(
            name="analyze_sentiment",
            description="Analyze sentiment of text content",
            cache_ttl=3600
        )
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sentiment analysis"""
        text = context.get("text", "")
        if not text:
            return {"sentiment": "neutral", "score": 0.0}
        
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                "sentiment": result["label"],
                "score": float(result["score"])
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "error", "score": 0.0}

@ray.remote
class EntityExtractionInstruction(OptimizableInstruction):
    """Extract named entities from text"""
    
    def __init__(self):
        super().__init__(
            name="extract_entities",
            description="Extract named entities from text"
        )
        self.nlp = spacy.load("en_core_web_sm")
        
        # Add optimization rules
        self.add_optimization_rule(self._should_optimize_long_text)
    
    def _should_optimize_long_text(self, context: Dict[str, Any]) -> bool:
        """Check if text is long enough to warrant optimization"""
        text = context.get("text", "")
        return len(text) > 1000
    
    async def _optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize by splitting long text into chunks"""
        text = context.get("text", "")
        chunk_size = 500
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        context["text_chunks"] = chunks
        return context
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute entity extraction"""
        if "text_chunks" in context:
            # Process chunks in parallel
            entities = []
            for chunk in context["text_chunks"]:
                doc = self.nlp(chunk)
                chunk_entities = [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    }
                    for ent in doc.ents
                ]
                entities.extend(chunk_entities)
        else:
            # Process single text
            text = context.get("text", "")
            doc = self.nlp(text)
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ]
        
        return {
            "entities": entities,
            "entity_count": len(entities)
        }

@ray.remote
class ContentProcessingInstruction(OptimizableInstruction):
    """Process content using multiple sub-instructions"""
    
    def __init__(self):
        super().__init__(
            name="process_content",
            description="Process content using multiple sub-instructions"
        )
        self.sentiment_analyzer = SentimentAnalysisInstruction.remote()
        self.entity_extractor = EntityExtractionInstruction.remote()
        
        # Add optimization rules
        self.add_optimization_rule(self._should_parallelize)
    
    def _should_parallelize(self, context: Dict[str, Any]) -> bool:
        """Check if content should be processed in parallel"""
        return len(context.get("text", "")) > 500
    
    async def _optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize by enabling parallel processing"""
        context["enable_parallel"] = True
        return context
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content processing"""
        if context.get("enable_parallel", False):
            # Execute sub-instructions in parallel
            futures = [
                self.sentiment_analyzer.execute.remote(context),
                self.entity_extractor.execute.remote(context)
            ]
            results = await ray.get(futures)
            
            # Combine results
            combined = {}
            for result in results:
                combined.update(result)
        else:
            # Execute sequentially
            sentiment = await ray.get(self.sentiment_analyzer.execute.remote(context))
            entities = await ray.get(self.entity_extractor.execute.remote(context))
            combined = {**sentiment, **entities}
        
        return combined
