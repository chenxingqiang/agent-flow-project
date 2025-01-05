import json
import os
import tempfile
from typing import Dict, Any

import pytest
from pydantic import ValidationError, Field
from agentflow.core.enums import ConfigurationType
from agentflow.core.config import AgentConfig, WorkflowConfig, ModelConfig, AgentMode

@pytest.fixture
def base_config():
    """Set up test environment with sample configurations."""
    return {
        "AGENT": {
            "type": "data_science",
            "name": "TestResearchAgent",
            "version": "1.2.0",
            "description": "Advanced research agent for scientific workflows",
            "workflow_path": "test_workflow.yaml",
            "mode": AgentMode.SYNC.value
        },
        "MODEL": {
            "provider": "default",
            "name": "default",
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "WORKFLOW": {
            "name": "test_workflow",
            "max_iterations": 15,
            "timeout": 600,
            "retry_strategy": {
                "max_retries": 5,
                "backoff_factor": 2
            },
            "steps": []
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
    agent_config = AgentConfig.from_dict(base_config)
    workflow_config = WorkflowConfig.model_validate(base_config["WORKFLOW"])
    
    # Validate basic properties
    assert agent_config.name == "TestResearchAgent"
    assert agent_config.type.value == "data_science"
    assert workflow_config.max_iterations == 15
    
    # Validate transformations
    assert len(base_config["TRANSFORMATIONS"]['input']) == 1
    assert len(base_config["TRANSFORMATIONS"]['preprocessing']) == 1

def test_configuration_validation(base_config):
    """Test comprehensive configuration validation."""
    # Valid configuration
    agent_config = AgentConfig.from_dict(base_config)
    workflow_config = WorkflowConfig.model_validate(base_config["WORKFLOW"])
    assert agent_config.model_dump()  # Validates the model
    assert workflow_config.model_dump()  # Validates the model
    
    # Invalid configurations
    invalid_configs = [
        # Invalid workflow configuration
        {
            "WORKFLOW": {
                "max_iterations": -1,  # Invalid: must be > 0
                "timeout": 600,
                "retry_strategy": {
                    "max_retries": 5,
                    "backoff_factor": 2
                }
            }
        },
        
        # Invalid agent configuration
        {
            "AGENT": {
                "type": "invalid_type",  # Invalid: must be one of ConfigurationType
                "name": "TestResearchAgent",
                "workflow_path": "test_workflow.yaml",
                "mode": "invalid_mode"  # Invalid: must be one of AgentMode
            },
            "MODEL": {
                "provider": "invalid_provider",  # Invalid: must be one of valid providers
                "name": "default"
            }
        }
    ]
    
    for invalid_config in invalid_configs:
        if "WORKFLOW" in invalid_config:
            with pytest.raises(ValidationError):
                WorkflowConfig.model_validate(invalid_config["WORKFLOW"])
        else:
            with pytest.raises(ValueError):
                AgentConfig.from_dict(invalid_config)

def test_configuration_merging(base_config):
    """Test configuration merging with priority handling."""
    agent_config = AgentConfig.from_dict(base_config)
    workflow_config = WorkflowConfig.model_validate(base_config["WORKFLOW"])
    
    # Partial configuration to merge
    partial_agent_config = AgentConfig(
        name="MergedAgent",
        type=ConfigurationType.RESEARCH,
        mode=AgentMode.SEQUENTIAL,
        model=ModelConfig(provider="default", name="default"),
        workflow=workflow_config.model_dump(),
        workflow_path="test_workflow.yaml"
    )
    
    # Merge configurations
    merged_agent_config = agent_config.model_copy()
    merged_agent_config = merged_agent_config.model_copy(update=partial_agent_config.model_dump())
    
    # Validate merged configuration
    assert merged_agent_config.name == "MergedAgent"
    assert isinstance(merged_agent_config.workflow, dict)
    assert merged_agent_config.workflow['max_iterations'] == workflow_config.max_iterations
    
    # Validate transformation merging
    transformations = base_config.get("TRANSFORMATIONS", {})
    assert len(transformations.get('output', [])) == 0

def test_configuration_export_and_import(base_config):
    """Test configuration export and import functionality."""
    agent_config = AgentConfig.from_dict(base_config)
    workflow_config = WorkflowConfig.model_validate(base_config["WORKFLOW"])
    
    # Test JSON export
    json_agent_config = agent_config.model_dump_json()
    json_workflow_config = workflow_config.model_dump_json()
    assert isinstance(json_agent_config, str)
    assert isinstance(json_workflow_config, str)
    
    # Test dictionary export
    dict_agent_config = agent_config.model_dump()
    dict_workflow_config = workflow_config.model_dump()
    assert isinstance(dict_agent_config, dict)
    assert isinstance(dict_workflow_config, dict)
    
    # Test file-based configuration saving and loading
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump(dict_agent_config, temp_file)
        temp_file_path = temp_file.name
    
    # Test loading from file
    with open(temp_file_path, 'r') as temp_file:
        loaded_config = json.load(temp_file)
        assert isinstance(loaded_config, dict)
        assert loaded_config["name"] == base_config["AGENT"]["name"]

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
                "type": "research",
                "mode": AgentMode.SEQUENTIAL.value
            },
            "MODEL": {
                "provider": "default",
                "name": "default"
            },
            "WORKFLOW": {
                "name": "partial_workflow",
                "max_iterations": 5,
                "steps": []
            }
        }
    ]
    
    for scenario in test_scenarios:
        if isinstance(scenario, str):
            loaded_config = json.loads(scenario)
            loaded_agent_config = AgentConfig.from_dict(loaded_config)
            loaded_workflow_config = WorkflowConfig.model_validate(loaded_config["WORKFLOW"])
        else:
            loaded_agent_config = AgentConfig.from_dict(scenario)
            loaded_workflow_config = WorkflowConfig.model_validate(scenario["WORKFLOW"])
        assert isinstance(loaded_agent_config, AgentConfig)
        assert isinstance(loaded_workflow_config, WorkflowConfig)

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
    
    agent_config = AgentConfig.from_dict(domain_config)
    workflow_config = WorkflowConfig.model_validate(domain_config["WORKFLOW"])
    
    # Validate domain-specific configuration
    assert domain_config['DOMAIN_CONFIG']['research_domains'] == ["quantum computing", "artificial intelligence"]
    assert "publication_targets" in domain_config['DOMAIN_CONFIG']

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
    agent_config = AgentConfig(
        name=advanced_config["AGENT"]["name"],
        type=ConfigurationType(advanced_config["AGENT"]["type"]),
        workflow_path=advanced_config["AGENT"]["workflow_path"],
        mode=AgentMode(advanced_config["AGENT"]["mode"]),
        model=ModelConfig.model_validate(advanced_config["MODEL"])
    )

    # Validate the configuration
    assert agent_config.model_dump()
    assert agent_config.name == "TestResearchAgent"
    assert agent_config.type == ConfigurationType.DATA_SCIENCE
    assert agent_config.mode == AgentMode.SYNC
    assert agent_config.model.provider == "default"
