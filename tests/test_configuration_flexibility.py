import json
import os
import tempfile
from typing import Dict, Any

import pytest
from pydantic import ValidationError, Field
from agentflow.agents.agent_types import AgentType, AgentMode, AgentConfig, ModelConfig
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStepType, WorkflowStep, StepConfig
import pydantic

@pytest.fixture
def base_config():
    """Set up test environment with sample configurations."""
    return {
        "AGENT": {
            "type": AgentType.DATA_SCIENCE.value,
            "name": "TestResearchAgent",
            "version": "1.2.0",
            "description": "Advanced research agent for scientific workflows",
            "workflow_path": "test_workflow.yaml",
            "mode": AgentMode.SEQUENTIAL.value
        },
        "MODEL": {
            "provider": "default",
            "name": "default",
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "WORKFLOW": {
            "id": "test-workflow-1",
            "name": "test_workflow",
            "max_iterations": 15,
            "timeout": 600,
            "steps": [
                {
                    "id": "step-1",
                    "name": "Research Step",
                    "type": WorkflowStepType.RESEARCH_EXECUTION.value,
                    "description": "Perform research on the given topic",
                    "config": {
                        "strategy": "research",
                        "params": {
                            "max_retries": 3,
                            "retry_delay": 1.0,
                            "retry_backoff": 2.0,
                            "timeout": 30.0
                        }
                    },
                    "dependencies": [],
                    "required": True,
                    "optional": False,
                    "is_distributed": False
                }
            ]
        },
        "TRANSFORMATIONS": {
            "input": [
                {
                    "type": "outlier_removal",
                    "params": {
                        "method": "z_score",
                        "threshold": 3.0
                    }
                }
            ],
            "preprocessing": [
                {
                    "type": "feature_engineering",
                    "params": {
                        "strategy": "polynomial",
                        "degree": 2
                    }
                }
            ]
        },
        "DOMAIN_CONFIG": {
            "research_domains": ["machine learning", "data science"]
        }
    }

def test_configuration_schema_creation(base_config):
    """Test creating configuration schema from dictionary."""
    agent_config = AgentConfig.model_validate(base_config["AGENT"])
    
    # Create WorkflowStep instances
    workflow_data = base_config["WORKFLOW"].copy()
    workflow_data["steps"] = [WorkflowStep(**step) for step in workflow_data["steps"]]
    workflow_config = WorkflowConfig(**workflow_data)
    
    # Validate basic properties
    assert agent_config.name == "TestResearchAgent"
    assert agent_config.type == AgentType.DATA_SCIENCE
    assert workflow_config.max_iterations == 15
    
    # Validate transformations
    assert len(base_config["TRANSFORMATIONS"]['input']) == 1

def test_configuration_validation(base_config):
    """Test comprehensive configuration validation."""
    # Valid configuration
    agent_config = AgentConfig.model_validate(base_config["AGENT"])
    
    # Create WorkflowStep instances
    workflow_data = base_config["WORKFLOW"].copy()
    workflow_data["steps"] = [WorkflowStep(**step) for step in workflow_data["steps"]]
    workflow_config = WorkflowConfig(**workflow_data)
    
    model_config = ModelConfig.model_validate(base_config["MODEL"])
    
    assert agent_config.name == "TestResearchAgent"
    assert workflow_config.max_iterations == 15
    assert model_config.temperature == 0.7
    
    # Invalid configuration - invalid type
    invalid_config = {
        "name": "test_agent",
        "type": "invalid_type",  # This should trigger a validation error
        "mode": "sequential",
        "version": "1.0.0"
    }
    
    # This should raise a ValidationError because the type is invalid
    with pytest.raises(ValidationError) as exc_info:
        AgentConfig.model_validate(invalid_config)
    
    assert "type" in str(exc_info.value)  # Verify the error message mentions the invalid type

def test_configuration_merging(base_config):
    """Test configuration merging with priority handling."""
    agent_config = AgentConfig.model_validate(base_config["AGENT"])
    
    # Create WorkflowStep instances
    workflow_data = base_config["WORKFLOW"].copy()
    workflow_data["steps"] = [WorkflowStep(**step) for step in workflow_data["steps"]]
    workflow_config = WorkflowConfig(**workflow_data)
    
    # Create override configuration
    override_config = {
        "name": "OverrideAgent",
        "version": "2.0.0",
        "mode": AgentMode.PARALLEL.value
    }
    
    # Merge configurations
    merged_config = AgentConfig.model_validate({**base_config["AGENT"], **override_config})
    
    # Validate merged configuration
    assert merged_config.name == "OverrideAgent"
    assert merged_config.version == "2.0.0"
    assert merged_config.mode == AgentMode.PARALLEL
    assert merged_config.type == agent_config.type

def test_configuration_export_and_import(base_config):
    """Test configuration export and import functionality."""
    agent_config = AgentConfig.model_validate(base_config["AGENT"])
    
    # Create WorkflowStep instances
    workflow_data = base_config["WORKFLOW"].copy()
    workflow_data["steps"] = [WorkflowStep(**step) for step in workflow_data["steps"]]
    workflow_config = WorkflowConfig(**workflow_data)
    
    # Export to temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump(base_config, temp_file)
        temp_path = temp_file.name
    
    # Import from file
    with open(temp_path, 'r') as f:
        imported_config = json.load(f)
    
    imported_agent_config = AgentConfig.model_validate(imported_config["AGENT"])
    
    # Create WorkflowStep instances for imported config
    imported_workflow_data = imported_config["WORKFLOW"].copy()
    imported_workflow_data["steps"] = [WorkflowStep(**step) for step in imported_workflow_data["steps"]]
    imported_workflow_config = WorkflowConfig(**imported_workflow_data)
    
    # Validate imported configuration
    assert imported_agent_config.name == agent_config.name
    assert imported_agent_config.type == agent_config.type
    assert imported_workflow_config.max_iterations == workflow_config.max_iterations
    
    # Clean up
    os.unlink(temp_path)

def test_flexible_configuration_loading(base_config):
    """Test loading configurations from various sources."""
    test_scenarios = [
        # Dictionary configuration
        base_config,
        
        # JSON string configuration
        json.dumps(base_config),
        
        # Partial configuration with required fields
        {
            "AGENT": {
                "name": "PartialAgent",
                "workflow_path": "test_workflow.yaml",
                "type": AgentType.RESEARCH.value,
                "mode": AgentMode.SEQUENTIAL.value
            },
            "MODEL": {
                "provider": "default",
                "name": "default"
            },
            "WORKFLOW": {
                "id": "test-workflow-1",
                "name": "partial_workflow",
                "max_iterations": 5,
                "steps": [
                    {
                        "id": "step-1",
                        "name": "Research Step",
                        "type": WorkflowStepType.RESEARCH_EXECUTION.value,
                        "config": StepConfig(
                            strategy="research",
                            params={}
                        )
                    }
                ]
            }
        }
    ]
    
    for scenario in test_scenarios:
        if isinstance(scenario, str):
            loaded_config = json.loads(scenario)
            loaded_agent_config = AgentConfig.model_validate(loaded_config["AGENT"])
            
            # Create WorkflowStep instances
            workflow_data = loaded_config["WORKFLOW"].copy()
            workflow_data["steps"] = [WorkflowStep(**step) for step in workflow_data["steps"]]
            loaded_workflow_config = WorkflowConfig(**workflow_data)
        else:
            loaded_agent_config = AgentConfig.model_validate(scenario["AGENT"])
            
            # Create WorkflowStep instances
            workflow_data = scenario["WORKFLOW"].copy()
            workflow_data["steps"] = [WorkflowStep(**step) for step in workflow_data["steps"]]
            loaded_workflow_config = WorkflowConfig(**workflow_data)
        
        assert loaded_agent_config.name is not None
        assert loaded_workflow_config.max_iterations > 0

def test_domain_specific_configuration(base_config):
    """Test domain-specific configuration support."""
    # Create configuration with domain-specific settings
    domain_config = {
        **base_config,
        "DOMAIN_CONFIG": {
            "research_domains": ["quantum computing", "artificial intelligence"],
            "citation_styles": ["APA", "IEEE"],
            "publication_targets": {
                "conferences": ["NeurIPS", "ICML"],
                "journals": ["Nature", "Science"]
            }
        }
    }
    
    agent_config = AgentConfig.model_validate(domain_config["AGENT"])
    
    # Create WorkflowStep instances
    workflow_data = domain_config["WORKFLOW"].copy()
    workflow_data["steps"] = [WorkflowStep(**step) for step in workflow_data["steps"]]
    workflow_config = WorkflowConfig(**workflow_data)
    
    assert agent_config.type == AgentType.DATA_SCIENCE
    assert workflow_config.max_iterations == 15

def test_advanced_transformation_configuration(base_config):
    """Test advanced transformation configuration scenarios."""
    advanced_config = {
        **base_config,
        "TRANSFORMATIONS": {
            "input": [
                {
                    "type": "outlier_removal",
                    "params": {
                        "method": "modified_z_score",
                        "threshold": 3.5,
                        "window_size": 10
                    }
                },
                {
                    "type": "feature_engineering",
                    "params": {
                        "strategy": "polynomial",
                        "degree": 3,
                        "interaction_only": True
                    }
                }
            ],
            "preprocessing": [
                {
                    "type": "time_series",
                    "params": {
                        "strategy": "moving_average",
                        "window_size": 5
                    }
                }
            ]
        }
    }
    
    # Create agent configuration with all required fields
    agent_config = AgentConfig.model_validate(advanced_config["AGENT"])
    
    # Create WorkflowStep instances
    workflow_data = advanced_config["WORKFLOW"].copy()
    workflow_data["steps"] = [WorkflowStep(**step) for step in workflow_data["steps"]]
    workflow_config = WorkflowConfig(**workflow_data)
    
    assert agent_config.type == AgentType.DATA_SCIENCE
    assert workflow_config.max_iterations == 15
    assert len(advanced_config["TRANSFORMATIONS"]["input"]) == 2
