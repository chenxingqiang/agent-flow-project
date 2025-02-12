# AgentFlow Documentation

## Overview

AgentFlow is a powerful and flexible AI Agent Workflow Management System that enables the creation, configuration, and management of intelligent agents. It provides a comprehensive framework for building agent-based applications with features like dynamic configuration, workflow management, and extensive testing capabilities.

## Latest Version

Current version: v0.1.1
- Fixed workflow transform functions to handle step and context parameters
- Added feature engineering and outlier removal transforms
- Improved test suite and type hints
- Enhanced error handling and validation

## Table of Contents

1. [Installation](#installation)
2. [Core Components](#core-components)
3. [Workflow Management](#workflow-management)
4. [Testing Framework](#testing-framework)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Development](#development)

## Installation

```bash
# Clone the repository
git clone https://github.com/chenxingqiang/agentflow.git

# Install dependencies
cd agentflow
pip install -r requirements.txt
```

## Core Components

### Agent Configuration

The system uses a flexible configuration system for agents:

```python
from agentflow.core.config import AgentConfig
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType

# Create workflow configuration
workflow_config = WorkflowConfig(
    id="test-workflow",
    name="test_workflow",
    steps=[
        WorkflowStep(
            id="step-1",
            name="transform_step",
            type=WorkflowStepType.TRANSFORM,
            description="Data transformation step",
            config={
                "strategy": "standard",
                "params": {
                    "method": "standard",
                    "with_mean": True,
                    "with_std": True
                }
            }
        )
    ]
)

# Create agent configuration
agent_config = AgentConfig(
    name="test_agent",
    type="data_science",
    workflow=workflow_config
)
```

## Workflow Management

### Transform Functions

Transform functions are a key component of workflow steps. They must accept both step and context parameters:

```python
async def feature_engineering_transform(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
    """Feature engineering transform function.
    
    Args:
        step: The workflow step being executed
        context: The execution context containing the data
        
    Returns:
        Dict containing the transformed data
    """
    data = context["data"]
    scaler = StandardScaler(
        with_mean=step.config.params["with_mean"],
        with_std=step.config.params["with_std"]
    )
    transformed_data = scaler.fit_transform(data)
    return {"data": transformed_data}
```

### Workflow Execution

```python
from agentflow.core.workflow_executor import WorkflowExecutor

# Create executor
executor = WorkflowExecutor(workflow_config)
await executor.initialize()

# Execute workflow
result = await executor.execute({"data": your_data})
```

## Testing Framework

### Unit Tests

```python
import pytest
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep

@pytest.mark.asyncio
async def test_workflow_execution():
    """Test basic workflow execution."""
    config = WorkflowConfig(
        id="test-workflow",
        name="test_workflow",
        steps=[
            WorkflowStep(
                id="step-1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="standard",
                    params={"execute": your_transform_function}
                )
            )
        ]
    )
    executor = WorkflowExecutor(config)
    await executor.initialize()
    result = await executor.execute({"data": test_data})
    assert result["status"] == "success"
```

### Performance Tests

```python
@pytest.mark.asyncio
async def test_workflow_performance():
    """Test workflow performance."""
    workflow = WorkflowConfig(
        id="perf-workflow",
        name="performance_workflow",
        steps=[
            WorkflowStep(
                id="step-1",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="feature_engineering",
                    params={
                        "method": "standard",
                        "with_mean": True,
                        "with_std": True,
                        "execute": feature_engineering_transform
                    }
                )
            )
        ]
    )
    start_time = time.time()
    result = await workflow.execute({"data": large_dataset})
    execution_time = time.time() - start_time
    assert execution_time < 5  # Should complete within 5 seconds
```

## API Reference

### Core Components

- `AgentConfig`: Configuration for agents
- `WorkflowConfig`: Configuration for workflows
- `WorkflowStep`: Individual workflow step
- `WorkflowExecutor`: Executes workflows
- `Agent`: Base agent class

### Transform Types

- `feature_engineering_transform`: Feature engineering transform
- `outlier_removal_transform`: Outlier removal transform
- `text_transform`: Text processing transform

## Examples

### Basic Workflow

```python
from agentflow import Agent, AgentConfig, WorkflowConfig

# Create configuration
config = AgentConfig(
    name="example_agent",
    type="data_science",
    workflow=WorkflowConfig(
        id="example-workflow",
        steps=[
            WorkflowStep(
                id="transform",
                type=WorkflowStepType.TRANSFORM,
                config=StepConfig(
                    strategy="feature_engineering",
                    params={"execute": feature_engineering_transform}
                )
            )
        ]
    )
)

# Create agent
agent = Agent(config)
await agent.initialize()

# Process data
result = await agent.execute({"data": your_data})
```

## Development

### Project Structure

```
agentflow/
├── core/
│   ├── base_types.py
│   ├── config.py
│   ├── workflow.py
│   └── workflow_executor.py
├── transformations/
│   └── text.py
└── tests/
    ├── unit/
    ├── core/
    └── performance/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/core/
pytest tests/performance/

# Run with coverage
pytest --cov=agentflow tests/
```

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
