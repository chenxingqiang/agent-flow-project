# AgentFlow Examples

This directory contains examples demonstrating how to use AgentFlow for various tasks.

## Latest Version

Current version: v0.2.0
- Added OpenAI integration with GPT-4 support
- Enhanced workflow capabilities with async/await support
- Added new chat and prompt caching examples
- Improved error handling and validation
- Added support for PDF report generation

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
   - PDF report generation with figures

4. **Chat and Interaction**
   - Quick chat simulation with AI personalities
   - OpenAI prompt caching
   - Audio processing capabilities
   - Interactive conversations

## Running Examples

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

3. Run examples:
```bash
# Basic workflow example
python examples/basic_workflow.py

# Data science workflow example
python examples/data_science_workflow.py

# Research workflow example
python examples/research_workflow.py

# Quick chat simulation
python examples/quick_chat.py

# OpenAI prompt caching
python examples/openai_prompt_caching.py

# Audio processing
python examples/openai_audio.py
```

## New Example Details

### Quick Chat Simulation

```python
# Create AI personalities with backstories
@ell2a.with_ell2a(mode="simple")
async def create_personality() -> str:
    """Create a personality with a backstory."""
    name = random.choice(names_list)
    # ... personality creation logic ...

# Generate chat responses
@ell2a.with_ell2a(mode="simple")
async def chat(message_history: List[Tuple[str, str]], personality: str) -> str:
    """Generate a chat response based on personality and history."""
    # ... chat generation logic ...
```

### OpenAI Prompt Caching

```python
@ell2a.with_ell2a(mode="simple")
async def cached_chat(history: List[str], new_message: str) -> str:
    """Chat with caching support for better performance."""
    # ... cached chat implementation ...
```

### Research Report Generation

```python
def generate_pdf_report(content: str, output_path: str) -> str:
    """Generate a PDF report with figures and formatting."""
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    # ... report generation logic ...
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
   - PDF reports with figures

4. Chat Simulation Output:
   - AI-generated personalities
   - Natural conversation flow
   - Cached responses
   - Audio processing results

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

4. **OpenAI Integration**
   - Always handle API errors gracefully
   - Implement proper rate limiting
   - Use environment variables for API keys
   - Cache responses when possible

5. **Report Generation**
   - Include clear visualizations
   - Maintain consistent formatting
   - Handle large datasets efficiently
   - Provide proper error handling
