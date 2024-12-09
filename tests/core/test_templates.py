"""
Tests for workflow template management
"""

import os
import pytest
import tempfile

from agentflow.core.templates import (
    TemplateManager, 
    WorkflowTemplate, 
    TemplateParameter
)
from agentflow.core.config_manager import (
    WorkflowConfig, 
    AgentConfig, 
    ModelConfig
)

@pytest.fixture
def template_manager():
    """Create a temporary template manager"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TemplateManager(tmpdir)
        yield manager

def test_template_creation_and_saving(template_manager):
    """Test creating and saving workflow templates"""
    # Create a workflow template
    template = WorkflowTemplate(
        id="research-workflow",
        name="Research Workflow",
        description="A template for research workflows",
        parameters=[
            TemplateParameter(
                name="topic",
                description="Research topic",
                type="string",
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
            id="research-workflow-template",
            name="Research Workflow Template",
            description="Template for research workflows",
            agents=[
                AgentConfig(
                    id="researcher-agent",
                    name="Researcher Agent",
                    description="Agent for conducting research on {{ topic }}",
                    type="research",
                    model=ModelConfig(
                        name="research-model",
                        provider="openai"
                    ),
                    system_prompt="Conduct research on {{ topic }} with {{ depth }} depth"
                )
            ],
            processors=[],
            connections=[]
        )
    )
    
    # Save template
    template_manager.save_template(template)
    
    # Load template
    loaded_template = template_manager.load_template("research-workflow")
    
    assert loaded_template is not None
    assert loaded_template.id == "research-workflow"
    assert loaded_template.name == "Research Workflow"
    assert len(loaded_template.parameters) == 2

def test_template_instantiation(template_manager):
    """Test instantiating workflow from template"""
    # Create a workflow template
    template = WorkflowTemplate(
        id="research-workflow",
        name="Research Workflow",
        description="A template for research workflows",
        parameters=[
            TemplateParameter(
                name="topic",
                description="Research topic",
                type="string",
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
            id="research-workflow-template",
            name="Research Workflow Template",
            description="Template for research on {{ topic }}",
            agents=[
                AgentConfig(
                    id="researcher-agent",
                    name="Researcher Agent",
                    description="Agent for conducting research on {{ topic }}",
                    type="research",
                    model=ModelConfig(
                        name="research-model",
                        provider="openai"
                    ),
                    system_prompt="Conduct research on {{ topic }} with {{ depth }} depth"
                )
            ],
            processors=[],
            connections=[]
        )
    )
    
    # Save template
    template_manager.save_template(template)
    
    # Instantiate template
    workflow = template_manager.instantiate_template(
        "research-workflow", 
        {
            "topic": "AI Ethics",
            "depth": "deep"
        }
    )
    
    assert workflow is not None
    assert workflow.id == "research-workflow-template"
    
    # Check parameter substitution
    researcher_agent = workflow.agents[0]
    assert "AI Ethics" in researcher_agent.description
    assert "Conduct research on AI Ethics with deep depth" in researcher_agent.system_prompt

def test_template_parameter_validation(template_manager):
    """Test template parameter validation"""
    # Create a workflow template
    template = WorkflowTemplate(
        id="research-workflow",
        name="Research Workflow",
        description="A template for research workflows",
        parameters=[
            TemplateParameter(
                name="topic",
                description="Research topic",
                type="string",
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
            id="research-workflow-template",
            name="Research Workflow Template",
            description="Template for research workflows",
            agents=[],
            processors=[],
            connections=[]
        )
    )
    
    # Save template
    template_manager.save_template(template)
    
    # Test valid instantiation
    workflow = template_manager.instantiate_template(
        "research-workflow", 
        {
            "topic": "Machine Learning",
            "depth": "deep"
        }
    )
    assert workflow is not None
    
    # Test missing required parameter
    with pytest.raises(ValueError, match="Missing required parameter: topic"):
        template_manager.instantiate_template(
            "research-workflow", 
            {}
        )
    
    # Test invalid parameter value
    with pytest.raises(ValueError, match="Invalid value for parameter depth"):
        template_manager.instantiate_template(
            "research-workflow", 
            {
                "topic": "AI",
                "depth": "invalid-depth"
            }
        )

def test_template_listing_and_deletion(template_manager):
    """Test listing and deleting templates"""
    # Create multiple templates
    templates = [
        WorkflowTemplate(
            id=f"workflow-{i}",
            name=f"Workflow {i}",
            description=f"Description for workflow {i}",
            parameters=[],
            workflow=WorkflowConfig(
                id=f"workflow-template-{i}",
                name=f"Workflow Template {i}",
                description=f"Description for workflow template {i}",
                agents=[],
                processors=[],
                connections=[]
            )
        ) for i in range(3)
    ]
    
    # Save templates
    for template in templates:
        template_manager.save_template(template)
    
    # List templates
    listed_templates = template_manager.list_templates()
    assert len(listed_templates) == 3
    
    # Delete a template
    result = template_manager.delete_template("workflow-1")
    assert result is True
    
    # Verify deletion
    listed_templates = template_manager.list_templates()
    assert len(listed_templates) == 2
    assert all(template.id != "workflow-1" for template in listed_templates)
