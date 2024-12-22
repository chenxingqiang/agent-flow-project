"""Research Agent Module"""

from typing import Dict, Any, Optional, Union, List
import logging
from .agent import Agent
from .config import AgentConfig

class ResearchAgent(Agent):
    """
    Specialized agent for research-oriented tasks.
    Extends the base Agent with research-specific capabilities.
    """
    def __init__(
        self,
        config: Union[Dict[str, Any], 'AgentConfig'],
        agent_config_path: Optional[str] = None
    ):
        """
        Initialize ResearchAgent
        
        Args:
            config: Agent configuration dictionary or object
            agent_config_path: Optional path to configuration file
        """
        super().__init__(config, agent_config_path)
        self.research_context = {}
        self.citations = []
        self.methodology = None
        
        # Initialize research-specific attributes
        self.initialize_research_context()
    
    def initialize_research_context(self):
        """Initialize research-specific context and attributes"""
        self.research_context = {
            'domain': self.config.get('domain', 'general'),
            'methodology': self.config.get('methodology', 'default'),
            'citations': [],
            'references': []
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process research-specific input data
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processed research output
        """
        # Add research context to input
        input_data['research_context'] = self.research_context
        
        # Process with base implementation
        result = await super().process(input_data)
        
        # Extract and store citations
        if 'citations' in result:
            self.citations.extend(result['citations'])
        
        return result
    
    def add_citation(self, citation: Dict[str, Any]):
        """
        Add a citation to the research context
        
        Args:
            citation: Citation information
        """
        self.citations.append(citation)
        self.research_context['citations'].append(citation)
    
    def set_methodology(self, methodology: str):
        """
        Set the research methodology
        
        Args:
            methodology: Research methodology to use
        """
        self.methodology = methodology
        self.research_context['methodology'] = methodology
    
    def get_citations(self) -> List[Dict[str, Any]]:
        """
        Get all collected citations
        
        Returns:
            List of citations
        """
        return self.citations
    
    def get_research_context(self) -> Dict[str, Any]:
        """
        Get the current research context
        
        Returns:
            Research context dictionary
        """
        return self.research_context
