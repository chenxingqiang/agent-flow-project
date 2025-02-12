# AgentFlow Examples

This directory contains examples demonstrating how to use AgentFlow for various tasks.

## Latest Version

Current version: v0.1.1
- Fixed workflow transform functions to handle step and context parameters
- Added feature engineering and outlier removal transforms
- Improved test suite and type hints
- Enhanced error handling and validation

## Examples Overview

1. **Basic Workflow**
   - Simple workflow creation and execution
   - Transform function implementation
   - Error handling

2. **Data Science Workflow**
   - Feature engineering transforms
   - Outlier removal
   - Performance optimization

3. **Research Workflow**
   - Document processing
   - Multi-step workflows
   - Dependency management

## Running Examples

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run examples:
```bash
# Basic workflow example
python examples/basic_workflow.py

# Data science workflow example
python examples/data_science_workflow.py

# Research workflow example
python examples/research_workflow.py
```

## Example Details

### Basic Workflow

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

### Data Science Workflow

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

async def outlier_removal_transform(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
    """Outlier removal transform function.
    
    Args:
        step: The workflow step being executed
        context: The execution context containing the data
        
    Returns:
        Dict containing the transformed data
    """
    data = context["data"]
    iso_forest = IsolationForest(
        contamination=step.config.params["threshold"],
        random_state=42
    )
    predictions = iso_forest.fit_predict(data)
    filtered_data = data[predictions == 1]
    return {"data": filtered_data}

# Create workflow with multiple steps
workflow_config = WorkflowConfig(
    id="ds-workflow",
    name="data_science_workflow",
    steps=[
        WorkflowStep(
            id="step-1",
            name="feature_engineering",
            type=WorkflowStepType.TRANSFORM,
            description="Feature engineering step",
            config=StepConfig(
                strategy="feature_engineering",
                params={
                    "method": "standard",
                    "with_mean": True,
                    "with_std": True,
                    "execute": feature_engineering_transform
                }
            )
        ),
        WorkflowStep(
            id="step-2",
            name="outlier_removal",
            type=WorkflowStepType.TRANSFORM,
            description="Outlier removal step",
            dependencies=["step-1"],
            config=StepConfig(
                strategy="outlier_removal",
                params={
                    "method": "isolation_forest",
                    "threshold": 0.1,
                    "execute": outlier_removal_transform
                }
            )
        )
    ]
)
```

### Research Workflow

```python
# Create research workflow
workflow_config = WorkflowConfig(
    id="research-workflow",
    name="research_workflow",
    steps=[
        WorkflowStep(
            id="step-1",
            name="research_step",
            type=WorkflowStepType.RESEARCH_EXECUTION,
            description="Execute research step",
            config=StepConfig(
                strategy="standard",
                params={"protocol": "federated"}
            )
        ),
        WorkflowStep(
            id="step-2",
            name="document_step",
            type=WorkflowStepType.DOCUMENT_GENERATION,
            description="Generate documentation",
            dependencies=["step-1"],
            config=StepConfig(
                strategy="standard",
                params={
                    "format": "markdown",
                    "sections": ["introduction", "methodology", "results", "conclusion"]
                }
            )
        )
    ]
)
```

## Example Output

1. Feature Engineering Output:
   - Standardized feature values
   - Feature importance scores
   - Transformation statistics

2. Outlier Removal Output:
   - Number of outliers detected
   - Cleaned dataset
   - Outlier indices

3. Research Output:
   - Research findings
   - Generated documentation
   - Analysis results

## Best Practices

1. **Transform Functions**
   - Always accept both `step` and `context` parameters
   - Return dictionary with transformed data
   - Include proper type hints and docstrings

2. **Workflow Configuration**
   - Use descriptive step names and IDs
   - Set appropriate dependencies
   - Configure error handling and retries

3. **Testing**
   - Write unit tests for transform functions
   - Test workflow execution
   - Verify error handling
