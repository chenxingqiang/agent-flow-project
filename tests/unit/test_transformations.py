import pytest
from agentflow.agents.agent import TransformationPipeline
from agentflow.transformations.text import TextTransformationStrategy

def test_transformation_pipeline():
    """Test transformation pipeline functionality."""
    pipeline = TransformationPipeline()
    
    # Create text transformation strategies
    lowercase_strategy = TextTransformationStrategy({'lowercase': True})
    strip_strategy = TextTransformationStrategy({'strip': True})
    replace_strategy = TextTransformationStrategy({
        'replacements': {'hello': 'hi', 'world': 'earth'}
    })
    
    # Add strategies to pipeline
    pipeline.add_strategy(lowercase_strategy)
    pipeline.add_strategy(strip_strategy)
    pipeline.add_strategy(replace_strategy)
    
    # Test pipeline transformation
    input_text = "  HELLO WORLD!  "
    expected = "hi earth!"
    result = pipeline.fit_transform(input_text)
    assert result == expected

def test_text_transformation():
    """Test text transformation strategy."""
    config = {
        'lowercase': True,
        'strip': True,
        'replacements': {'hello': 'hi', 'world': 'earth'}
    }
    strategy = TextTransformationStrategy(config)
    
    # Test basic transformation
    input_text = "  HELLO WORLD!  "
    expected = "hi earth!"
    result = strategy.transform(input_text)
    assert result == expected
    
    # Test invalid input
    with pytest.raises(ValueError):
        strategy.transform(123)
        
def test_empty_pipeline():
    """Test empty transformation pipeline."""
    pipeline = TransformationPipeline()
    input_text = "test"
    result = pipeline.fit_transform(input_text)
    assert result == input_text 