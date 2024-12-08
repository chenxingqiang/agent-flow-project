"""
AgentFlow Academic Research Workflow Example

This example demonstrates how to create a comprehensive academic research workflow
using AgentFlow's dynamic configuration and workflow execution capabilities.
"""

import asyncio
from typing import List, Dict, Any

from agentflow.core.config_manager import (
    AgentConfig, 
    ModelConfig, 
    WorkflowConfig
)
from agentflow.core.workflow_executor import WorkflowExecutor
from agentflow.core.templates import WorkflowTemplate, TemplateParameter

class AcademicResearchAgent:
    """Specialized agent for academic research tasks"""
    def __init__(self, research_domain: str, academic_level: str):
        self.research_domain = research_domain
        self.academic_level = academic_level
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate academic research workflow stages"""
        stage = input_data.get('stage', 'literature_review')
        
        research_stages = {
            'literature_review': self._literature_review,
            'methodology': self._research_methodology,
            'data_analysis': self._data_analysis,
            'paper_writing': self._paper_writing
        }
        
        return await research_stages.get(stage, self._literature_review)(input_data)
    
    async def _literature_review(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct literature review for the research domain"""
        return {
            "stage": "literature_review",
            "domain": self.research_domain,
            "sources": [
                f"Academic paper on {self.research_domain} - Source 1",
                f"Academic paper on {self.research_domain} - Source 2"
            ],
            "summary": f"Comprehensive literature review for {self.research_domain}"
        }
    
    async def _research_methodology(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Define research methodology based on literature review"""
        return {
            "stage": "methodology",
            "methodology_type": "Qualitative" if self.academic_level == "PhD" else "Mixed",
            "research_questions": [
                f"What are the key challenges in {self.research_domain}?",
                f"How can we innovate in {self.research_domain}?"
            ]
        }
    
    async def _data_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate data analysis process"""
        return {
            "stage": "data_analysis",
            "analysis_method": "Statistical Analysis",
            "key_findings": [
                "Significant trend observed",
                "Novel insights discovered"
            ]
        }
    
    async def _paper_writing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate academic paper draft"""
        return {
            "stage": "paper_writing",
            "paper_sections": [
                "Introduction",
                "Literature Review",
                "Methodology",
                "Results",
                "Conclusion"
            ],
            "draft_quality": "High-quality academic draft"
        }

def create_academic_research_workflow_template() -> WorkflowTemplate:
    """Create a dynamic workflow template for academic research"""
    return WorkflowTemplate(
        id="academic-research-workflow",
        name="Academic Research Workflow Template",
        description="Dynamic workflow for conducting academic research",
        parameters=[
            TemplateParameter(
                name="research_domains",
                description="Research domains to explore",
                type="list",
                required=True
            ),
            TemplateParameter(
                name="academic_level",
                description="Academic research level",
                type="string",
                options=["Undergraduate", "Master", "PhD"],
                default="Master"
            )
        ],
        workflow=WorkflowConfig(
            id="academic-research-workflow",
            name="Multi-Domain Academic Research Workflow",
            agents=[
                AgentConfig(
                    id="research-agent-{{ domain }}",
                    name="Research Agent for {{ domain }}",
                    type="research",
                    model=ModelConfig(
                        name="gpt-4",
                        provider="openai"
                    ),
                    system_prompt="Conduct {{ academic_level }} level research on {{ domain }}"
                ) for domain in "{{ research_domains }}"
            ],
            processors=[
                {
                    "id": "filter-processor",
                    "type": "filter",
                    "config": {
                        "conditions": [
                            {"field": "stage", "operator": "exists"}
                        ]
                    }
                },
                {
                    "id": "transform-processor",
                    "type": "transform",
                    "config": {
                        "transformations": {
                            "research_summary": "domain + ': ' + stage"
                        }
                    }
                }
            ]
        )
    )

async def main():
    """Main function to demonstrate academic research workflow"""
    # Create workflow template
    research_template = create_academic_research_workflow_template()
    
    # Instantiate workflow
    workflow_config = research_template.instantiate_template(
        "academic-research-workflow", 
        {
            "research_domains": ["AI Ethics", "Quantum Computing"],
            "academic_level": "PhD"
        }
    )
    
    # Create workflow executor
    executor = WorkflowExecutor(workflow_config)
    
    # Patch agents with custom implementation
    for agent_config in workflow_config.agents:
        domain = agent_config.id.split('-')[-1]
        academic_level = workflow_config.parameters.get('academic_level', 'Master')
        custom_agent = AcademicResearchAgent(domain, academic_level)
        executor.nodes[agent_config.id].agent = custom_agent
    
    # Execute workflow
    results = await executor.execute()
    
    # Print results
    for result in results:
        print(f"Research Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())