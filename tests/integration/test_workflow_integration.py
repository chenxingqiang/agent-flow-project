"""
Integration tests for AgentFlow workflow components
"""

import pytest
import asyncio
from typing import List, Dict, Any

from agentflow.core.config_manager import (
    ConfigManager, ModelConfig
)
from agentflow.core.workflow_executor import WorkflowExecutor, WorkflowManager
from agentflow.core.templates import (
    TemplateManager, 
    WorkflowTemplate, 
    TemplateParameter, 
    WorkflowConfig, 
    AgentConfig
)
from agentflow.core.processors.transformers import (
    FilterProcessor, 
    TransformProcessor, 
    AggregateProcessor
)

class MockResearchAgent:
    """Mock agent for research workflow simulation"""
    def __init__(self, research_domain: str):
        self.research_domain = research_domain
        self.processed_data = []
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate research data processing"""
        research_result = {
            "domain": self.research_domain,
            "input": input_data,
            "insights": f"Insights for {self.research_domain}"
        }
        self.processed_data.append(research_result)
        return research_result

@pytest.fixture
def config_manager():
    """Create a configuration manager for integration tests"""
    return ConfigManager()

@pytest.fixture
def template_manager():
    """Create a template manager for integration tests"""
    return TemplateManager()

@pytest.mark.asyncio
async def test_research_workflow_integration(config_manager, template_manager):
    """Integration test for a research workflow"""
    # Create research workflow template
    research_template = WorkflowTemplate(
        id="research-workflow",
        name="Research Workflow Template",
        description="A template for conducting research across multiple domains",
        parameters=[
            TemplateParameter(
                name="domains",
                description="Research domains",
                type="list",
                required=True
            ),
            TemplateParameter(
                name="depth",
                description="Research depth",
                type="string",
                options=["shallow", "medium", "deep"],
                default="medium"
            )
        ],
        workflow=WorkflowConfig(
            id="research-workflow",
            name="Multi-Domain Research Workflow",
            description="Research workflow for {{ domains }}",
            agents=[
                AgentConfig(
                    id="research-agent-{{ domain }}",
                    name="Research Agent for {{ domain }}",
                    type="research",
                    description="Agent for {{ domain }} research",
                    model=ModelConfig(
                        name="research-model",
                        provider="openai"
                    ),
                    system_prompt="Conduct {{ depth }} research on {{ domain }}"
                ) for domain in "{{ domains }}"
            ],
            processors=[
                {
                    "id": "filter-processor",
                    "type": "processor",
                    "processor": "agentflow.core.processors.transformers.FilterProcessor",
                    "config": {
                        "conditions": [
                            {"field": "insights", "operator": "exists"}
                        ]
                    }
                },
                {
                    "id": "transform-processor",
                    "type": "processor",
                    "processor": "agentflow.core.processors.transformers.TransformProcessor",
                    "config": {
                        "transformations": {
                            "research_summary": "domain + ': ' + insights"
                        }
                    }
                }
            ],
            connections=[]
        )
    )
    
    # Save template
    template_manager.save_template(research_template)
    
    # Instantiate workflow
    workflow_config = template_manager.instantiate_template(
        "research-workflow", 
        {
            "domains": ["AI", "Robotics", "Quantum Computing"],
            "depth": "deep"
        }
    )
    
    # Create workflow executor
    executor = WorkflowExecutor(workflow_config)
    
    # Mock research agents
    mock_agents = {
        agent_config.id: MockResearchAgent(agent_config.id.split('-')[-1])
        for agent_config in workflow_config.agents
    }
    
    # Patch agent initialization
    for agent_id, mock_agent in mock_agents.items():
        executor.nodes[agent_id].agent = mock_agent
    
    # Execute workflow
    await executor.execute()
    
    # Verify workflow execution
    for agent_id, mock_agent in mock_agents.items():
        assert len(mock_agent.processed_data) > 0
        assert all('insights' in result for result in mock_agent.processed_data)

@pytest.mark.asyncio
async def test_data_processing_workflow_integration():
    """Integration test for complex data processing workflow"""
    # Create sample input data
    input_data = [
        {"category": "tech", "value": 100, "source": "A"},
        {"category": "finance", "value": 50, "source": "B"},
        {"category": "tech", "value": 200, "source": "C"},
        {"category": "finance", "value": 75, "source": "D"}
    ]
    
    # Create workflow with multiple processors
    workflow_config = WorkflowConfig(
        id="data-processing-workflow",
        name="Data Processing Workflow",
        description="Complex data processing with multiple stages",
        agents=[],
        processors=[
            {
                "id": "filter-tech",
                "type": "processor",
                "processor": "agentflow.core.processors.transformers.FilterProcessor",
                "config": {
                    "conditions": [
                        {"field": "category", "operator": "eq", "value": "tech"}
                    ]
                }
            },
            {
                "id": "transform-tech",
                "type": "processor",
                "processor": "agentflow.core.processors.transformers.TransformProcessor",
                "config": {
                    "transformations": {
                        "normalized_value": "value / 100",
                        "source_type": "source + '-tech'"
                    }
                }
            },
            {
                "id": "aggregate-tech",
                "type": "processor",
                "processor": "agentflow.core.processors.transformers.AggregateProcessor",
                "config": {
                    "group_by": "source_type",
                    "aggregations": {
                        "total_value": {
                            "type": "sum",
                            "field": "normalized_value"
                        },
                        "count": {
                            "type": "count",
                            "field": "normalized_value"
                        }
                    }
                }
            }
        ],
        connections=[]
    )
    
    # Create workflow executor
    executor = WorkflowExecutor(workflow_config)
    
    # Process input data through processors
    filter_processor = FilterProcessor({"conditions": workflow_config.processors[0].config.get("conditions")})
    transform_processor = TransformProcessor({"transformations": workflow_config.processors[1].config.get("transformations")})
    aggregate_processor = AggregateProcessor({"group_by": workflow_config.processors[2].config.get("group_by"), "aggregations": workflow_config.processors[2].config.get("aggregations")})
    
    # Process data through workflow
    filtered_data = []
    for item in input_data:
        filter_result = await filter_processor.process(item)
        if filter_result.output:
            filtered_data.append(filter_result.output)
    
    transformed_data = []
    for item in filtered_data:
        transform_result = await transform_processor.process(item)
        transformed_data.append(transform_result.output)
    
    # Accumulate aggregate results
    aggregate_results = None
    for item in transformed_data:
        aggregate_result = await aggregate_processor.process(item)
        if aggregate_result.output:
            aggregate_results = aggregate_result

    # Verify processing results
    assert len(filtered_data) == 2  # Only tech category
    assert all(item['category'] == 'tech' for item in filtered_data)

    assert len(transformed_data) == 2
    assert all('normalized_value' in item for item in transformed_data)
    assert all('source_type' in item for item in transformed_data)

    assert aggregate_results is not None
    assert len(aggregate_results.output) > 0  # Ensure non-empty output
    
    # Print out aggregate results for debugging
    print("Aggregate Results:", aggregate_results.output)
