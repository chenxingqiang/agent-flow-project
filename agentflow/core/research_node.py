"""Research node implementation."""

from typing import Dict, Any, Optional
import logging
import asyncio

from .node import AgentNode
from ..ell2a.integration import ELL2AIntegration
from ..ell2a.types.message import Message, MessageRole

logger = logging.getLogger(__name__)

class ResearchNode(AgentNode):
    """Node for executing research tasks."""
    
    def __init__(self, name: str, description: str, model_config: Optional[Dict[str, Any]] = None):
        """Initialize research node.
        
        Args:
            name: Node name
            description: Node description
            model_config: Configuration for the language model
        """
        super().__init__(name, description)
        self.model_config = model_config or {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
        self._ell2a = None
    
    async def _initialize_impl(self) -> None:
        """Initialize ELL2A integration."""
        if not self._ell2a:
            self._ell2a = ELL2AIntegration()
            self._ell2a.configure({
                "model": self.model_config,
                "system_prompt": "You are a research assistant helping with academic research."
            })
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a research task.
        
        Args:
            task: Research task to process
            
        Returns:
            Research findings
        """
        try:
            # Extract research parameters
            topic = task.get("RESEARCH_TOPIC", "")
            academic_level = task.get("ACADEMIC_LEVEL", "")
            language = task.get("LANGUAGE", {})
            template = task.get("TEMPLATE", "")
            
            # Create research prompt
            prompt = f"""
            Conduct research on the topic: {topic}
            Academic Level: {academic_level}
            Language: {language.get('TYPE', 'English')}
            Style: {language.get('STYLE', 'Academic')}
            Template: {template}
            
            Please provide:
            1. Key findings and insights
            2. Relevant academic sources
            3. Structured analysis
            """
            
            # Create message object
            message = Message(
                role=MessageRole.USER,
                content=prompt,
                metadata={
                    "type": "research",
                    "topic": topic,
                    "academic_level": academic_level
                }
            )
            
            # Process with ELL2A
            response = await self._ell2a.process_message(message)
            
            return {
                "findings": response.content,
                "metadata": {
                    "topic": topic,
                    "academic_level": academic_level,
                    "language": language,
                    "template": template
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing research task: {e}")
            raise
    
    async def _cleanup_impl(self) -> None:
        """Clean up ELL2A resources."""
        if self._ell2a:
            await self._ell2a.cleanup()
            self._ell2a = None 