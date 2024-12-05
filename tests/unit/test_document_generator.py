import os
import json
import yaml
import pytest
from pydantic import ValidationError
from agentflow.core.document_generator import (
    DocumentGenerator, 
    ContentParser, 
    DocumentSchema
)

def test_content_parser_validation():
    """Test document content validation"""
    parser = ContentParser()
    
    # Valid document
    valid_doc = {
        'title': 'Research Report',
        'summary': 'Comprehensive study of AI technologies',
        'author': 'Research Team',
        'tags': ['AI', 'Machine Learning']
    }
    
    # Validate successful case
    validated_doc = parser.validate_document(valid_doc)
    assert 'title' in validated_doc
    assert 'tags' in validated_doc
    
    # Test invalid documents
    invalid_docs = [
        # Too long title
        {'title': 'A' * 250},
        # Empty title
        {'title': ''},
        {'title': '   '},
        # Invalid tags
        {'title': 'Test', 'tags': ['', '  ']}
    ]
    
    for invalid_doc in invalid_docs:
        try:
            parser.validate_document(invalid_doc)
            pytest.fail(f"Expected validation failure for {invalid_doc}")
        except ValueError as e:
            assert 'Document validation failed' in str(e)

def test_content_parser_metadata_extraction():
    """Test metadata extraction from different content types"""
    parser = ContentParser()
    
    # Markdown content
    markdown_content = """title: Project Proposal
author: AI Research Team
tags: [AI, Research]

## Introduction
Project background details

## Methodology
Research approach description"""
    
    # Extract all metadata
    extracted_md = parser.extract_metadata(markdown_content)
    assert extracted_md['title'] == 'Project Proposal'
    assert extracted_md['author'] == 'AI Research Team'
    assert extracted_md['tags'] == ['AI', 'Research']
    
    # Extract specific metadata
    specific_md = parser.extract_metadata(markdown_content, keys=['title', 'tags'])
    assert 'title' in specific_md
    assert 'tags' in specific_md
    assert 'author' not in specific_md

def test_content_parser_parsing():
    """Test advanced parsing capabilities"""
    parser = ContentParser()
    
    # Test markdown parsing with list
    markdown_content = """title: Advanced Research
tags: [Machine Learning, Deep Learning]

## Overview
Research summary"""
    
    parsed_md = parser.parse_markdown(markdown_content)
    assert parsed_md['title'] == 'Advanced Research'
    assert parsed_md['tags'] == ['Machine Learning', 'Deep Learning']
    
    # Test YAML parsing
    yaml_content = """
title: YAML Document
author: Research Team
tags: 
  - AI
  - Automation"""
    
    parsed_yaml = parser.parse_yaml(yaml_content)
    assert parsed_yaml['title'] == 'YAML Document'
    assert parsed_yaml['tags'] == ['AI', 'Automation']

def test_document_generator_advanced_features():
    """Test advanced document generation features"""
    generator = DocumentGenerator()
    
    # Test document with comprehensive metadata
    comprehensive_doc = {
        'title': 'Comprehensive AI Research Report',
        'summary': 'In-depth analysis of AI technologies',
        'author': 'Research Team',
        'date': '2023-09-15',
        'tags': ['AI', 'Machine Learning', 'Research'],
        'sections': {
            'Introduction': 'Background and motivation',
            'Methodology': 'Research approach and techniques'
        }
    }
    
    # Generate document with validation
    markdown_doc = generator.generate(comprehensive_doc, format='markdown')
    assert 'Comprehensive AI Research Report' in markdown_doc
    assert 'Research Team' in markdown_doc
    
    # Test with custom template
    research_doc = generator.generate(
        comprehensive_doc, 
        format='markdown', 
        template='research_report.j2'
    )
    assert 'Machine Learning' in research_doc
    assert 'Research approach' in research_doc

def test_content_parser():
    """Test advanced content parsing"""
    parser = ContentParser()
    
    # Test markdown parsing
    markdown_content = """title: Research Report
author: AI Agent

## Introduction
This is an introduction section.

## Methodology
Research methodology details."""
    
    parsed_markdown = parser.parse_markdown(markdown_content)
    assert parsed_markdown['title'] == 'Research Report'
    assert parsed_markdown['author'] == 'AI Agent'
    assert 'Introduction' in parsed_markdown['sections']
    assert 'Methodology' in parsed_markdown['sections']

def test_document_generator_advanced():
    """Test advanced document generation features"""
    generator = DocumentGenerator()
    
    # Test parsing different content types
    markdown_content = """title: Project Proposal
summary: Innovative project overview

## Research Questions
- What is the problem?
- How can we solve it?

## Proposed Solution
Detailed solution description."""
    
    # Generate from markdown
    markdown_doc = generator.generate(markdown_content, format='markdown')
    assert 'Project Proposal' in markdown_doc
    
    # Generate from markdown with custom template
    research_doc = generator.generate(
        markdown_content, 
        format='markdown', 
        template='research_report.j2'
    )
    assert 'Research Questions' in research_doc
    
    # Test JSON generation
    json_content = {
        'title': 'AI Research Project',
        'sections': {
            'Background': 'Detailed background information',
            'Goals': 'Project objectives and scope'
        }
    }
    json_doc = generator.generate(json_content, format='json')
    json_data = json.loads(json_doc)
    assert 'content' in json_data
    
    # Test YAML generation
    yaml_doc = generator.generate(json_content, format='yaml')
    parsed_yaml = yaml.safe_load(yaml_doc)
    assert 'content' in parsed_yaml

def test_document_generator_formats(tmp_path):
    """Test document generation with different formats"""
    generator = DocumentGenerator()
    
    test_content = {
        'title': 'Project Proposal',
        'summary': 'Innovative project overview',
        'sections': {
            'Introduction': 'Project background and motivation',
            'Methodology': 'Detailed research approach'
        }
    }
    
    # Test markdown generation
    markdown_doc = generator.generate(test_content, format='markdown')
    assert 'Project Proposal' in markdown_doc
    
    # Test HTML generation
    html_doc = generator.generate(test_content, format='html')
    assert '<h1>Project Proposal</h1>' in html_doc
    
    # Test JSON generation
    json_doc = generator.generate(test_content, format='json')
    json_data = json.loads(json_doc)
    assert 'content' in json_data
    
    # Test YAML generation
    yaml_doc = generator.generate(test_content, format='yaml')
    parsed_yaml = yaml.safe_load(yaml_doc)
    assert 'content' in parsed_yaml
    
    # Test file output for different formats
    formats = ['markdown', 'html', 'txt', 'json', 'yaml']
    for fmt in formats:
        output_path = os.path.join(tmp_path, f'test_doc.{fmt}')
        saved_path = generator.generate(test_content, format=fmt, output_path=output_path)
        
        assert saved_path == output_path
        assert os.path.exists(output_path)

def test_document_generator_templates():
    """Test template listing and custom template usage"""
    generator = DocumentGenerator()
    
    # List available templates
    templates = generator.list_templates()
    assert 'default.j2' in templates
    assert 'research_report.j2' in templates
    
    # Test custom template
    test_content = {
        'title': 'Custom Template Test',
        'research_questions': [
            'What is the research problem?',
            'How can we solve it?'
        ]
    }
    
    custom_doc = generator.generate(
        test_content,
        template='research_report.j2',
        format='markdown'
    )
    assert 'Custom Template Test' in custom_doc
    assert 'What is the research problem?' in custom_doc

def test_document_generator_error_handling():
    """Test error handling for unsupported formats and parsing"""
    generator = DocumentGenerator()
    
    test_content = {'title': 'Error Test'}
    
    # Test unsupported format
    with pytest.raises(ValueError, match='Unsupported format'):
        generator.generate(test_content, format='unsupported')
    
    # Test unsupported parsing method
    with pytest.raises(ValueError, match='Unsupported parsing method'):
        generator.generate('Test content', parse_method='invalid')
