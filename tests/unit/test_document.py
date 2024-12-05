import pytest
from pathlib import Path
from agentflow.core.document import DocumentGenerator

def test_document_generation(temp_dir):
    """Test document generation"""
    generator = DocumentGenerator({})
    content = {"title": "Test", "content": "Test content"}
    
    output_path = Path(temp_dir) / "test.md"
    result = generator.generate(content, "markdown", str(output_path))
    assert result == str(output_path)
    assert output_path.exists()

@pytest.mark.parametrize("output_format", ["docx", "markdown"])
def test_multiple_formats(temp_dir, output_format):
    """Test generation of different formats"""
    generator = DocumentGenerator({})
    content = {"title": "Test", "content": "Test content"}
    
    output_path = Path(temp_dir) / f"test.{output_format}"
    result = generator.generate(content, output_format, str(output_path))
    assert result == str(output_path)
    assert output_path.exists()

def test_template_rendering(temp_dir):
    """Test template rendering"""
    generator = DocumentGenerator({})
    content = {"title": "Test", "content": "Test content"}
    
    output_path = Path(temp_dir) / "test.md"
    result = generator.generate(content, "markdown", str(output_path))
    
    with open(output_path) as f:
        rendered = f.read()
    assert "Test" in rendered
    assert "Test content" in rendered

def test_error_handling(temp_dir):
    """Test error handling in document generation"""
    generator = DocumentGenerator({})
    output_path = Path(temp_dir) / "test.txt"
    
    with pytest.raises(ValueError):
        generator.generate({}, "invalid_format", str(output_path))