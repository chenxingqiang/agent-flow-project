# AgentFlow v0.1.1 Release Notes

## 🚀 Major Updates

### 1. Workflow Transform Functions

#### Core Updates
- ✨ Updated transform functions to accept both step and context parameters
- 🔧 Added feature engineering transform with StandardScaler support
- 🛠️ Added outlier removal transform with IsolationForest support

#### Testing Improvements
- ✅ Added comprehensive unit tests for transform functions
- 📊 Added performance tests for workflow execution
- 🔍 Enhanced test coverage and error handling tests

### 2. Documentation Updates
- 📚 Updated all README files with latest version information
- 💡 Added detailed examples for transform functions
- 📖 Improved API documentation and usage guides

### 3. Error Handling
- ⚡ Enhanced validation for transform function parameters
- 🔔 Improved error messages and debugging information
- 🔄 Added retry policies for workflow steps

## 🔄 Breaking Changes

Transform functions now require both `step` and `context` parameters:

```python
async def your_transform(
    step: WorkflowStep,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Transform function with new parameter structure.
    
    Args:
        step: The workflow step being executed
        context: The execution context containing the data
        
    Returns:
        Dict containing the transformed data
    """
    return {"data": transformed_data}
```

## 📋 Migration Guide

1. Update your transform functions:
```python
# Before
async def old_transform(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"data": process(data)}

# After
async def new_transform(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
    data = context["data"]
    return {"data": process(data)}
```

2. Update your workflow configurations:
```python
# Before
workflow_config = WorkflowConfig(
    steps=[
        WorkflowStep(
            id="step-1",
            type=WorkflowStepType.TRANSFORM,
            config={"strategy": "standard"}
        )
    ]
)

# After
workflow_config = WorkflowConfig(
    steps=[
        WorkflowStep(
            id="step-1",
            type=WorkflowStepType.TRANSFORM,
            config=StepConfig(
                strategy="standard",
                params={"execute": new_transform}
            )
        )
    ]
)
```

## 🎯 Examples

### Feature Engineering Transform
```python
async def feature_engineering_transform(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
    data = context["data"]
    scaler = StandardScaler(
        with_mean=step.config.params["with_mean"],
        with_std=step.config.params["with_std"]
    )
    transformed_data = scaler.fit_transform(data)
    return {"data": transformed_data}
```

### Outlier Removal Transform
```python
async def outlier_removal_transform(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
    data = context["data"]
    iso_forest = IsolationForest(
        contamination=step.config.params["threshold"],
        random_state=42
    )
    predictions = iso_forest.fit_predict(data)
    filtered_data = data[predictions == 1]
    return {"data": filtered_data}
```

## 📝 Additional Notes

- All tests have been updated to reflect the new parameter structure
- Documentation has been updated with new examples
- Error messages now provide more context about parameter requirements
