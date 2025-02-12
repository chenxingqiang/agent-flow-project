# AgentFlow

AgentFlow is a flexible and extensible framework for building and managing AI agents and workflows. It provides a modular architecture for creating, configuring, and orchestrating AI agents with advanced features for workflow management and testing.

## Latest Version

Current version: v0.1.1
- Fixed workflow transform functions to handle step and context parameters
- Added feature engineering and outlier removal transforms
- Improved test suite and type hints
- Enhanced error handling and validation

## Project Structure

```
agentflow/
├── agents/             # Agent implementations
│   ├── agent.py       # Base agent implementation
│   └── agent_types.py # Agent type definitions
├── core/              # Core framework components
│   ├── base_types.py  # Base type definitions
│   ├── config.py      # Configuration management
│   ├── workflow.py    # Workflow engine
│   └── workflow_executor.py # Workflow execution
├── ell2a/             # ELL2A integration
│   └── types/         # Message and content types
├── transformations/   # Data transformation tools
│   └── text.py       # Text processing utilities
└── tests/            # Comprehensive test suite
    ├── unit/         # Unit tests
    ├── core/         # Core component tests
    └── performance/  # Performance tests
```

## Features

- **Flexible Agent Architecture**
  - Configurable agent types and behaviors
  - Support for research and data science agents
  - Extensible agent factory system

- **Advanced Workflow Management**
  - Step-based workflow execution
  - Dependency management
  - Error handling and retry policies
  - Performance monitoring

- **Robust Testing Framework**
  - Unit and integration tests
  - Performance testing
  - Test-driven development support

- **Data Transformation Tools**
  - Feature engineering
  - Outlier removal
  - Text processing utilities

- **Type Safety**
  - Comprehensive type hints
  - Pydantic model validation
  - Runtime type checking

## Installation

```bash
pip install agentflow
```

## Quick Start

```python
from agentflow import Agent, AgentConfig, WorkflowConfig, WorkflowStep, WorkflowStepType

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

# Create and initialize agent
agent = Agent(config=agent_config)
await agent.initialize()

# Process data
result = await agent.execute({"data": your_data})
```

## Configuration

Agents and workflows can be configured using Python dictionaries or YAML files:

```yaml
AGENT:
  name: data_science_agent
  type: data_science
  version: 1.0.0
  mode: sequential

MODEL:
  provider: openai
  name: gpt-4
  temperature: 0.7
  max_tokens: 4096

WORKFLOW:
  id: transform-workflow
  name: Data Transformation Workflow
  max_iterations: 5
  timeout: 30
  steps:
    - id: step-1
      name: feature_engineering
      type: transform
      description: Feature engineering step
      config:
        strategy: standard
        params:
          method: standard
          with_mean: true
          with_std: true
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/core/
pytest tests/performance/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for your changes
4. Implement your changes
5. Run the test suite
6. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
