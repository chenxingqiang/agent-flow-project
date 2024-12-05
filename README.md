

### Project Structure

```
agentflow/
│
├── agentflow/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── workflow.py
│   │   └── document.py
│   ├── templates/
│   │   └── base.json
│   └── examples/
│       ├── academic/
│       │   ├── agent.json
│       │   └── templates/
│       ├── customer_service/
│       │   ├── agent.json
│       │   └── templates/
│       └── data_analysis/
│           ├── agent.json
│           └── templates/
│
├── tests/
│   └── test_agentflow.py
│
├── setup.py
├── README.md
├── LICENSE
└── requirements.txt
```

### setup.py

```python
from setuptools import setup, find_packages

setup(
    name='agentflow',
    version='0.1.0',
    description='A flexible framework for building and managing AI agent workflows',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Chen Xingqiang',
    author_email='chenxingqiang@gmail.com',
    url='https://github.com/chenxingqiang/agentflow',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'ell',
        'jinja2',
        'markdown',
        'python-docx',
        'pandas'
    ],
    entry_points={
        'console_scripts': [
            'agentflow=agentflow.core.agent:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)
```

### README.md

```markdown
# AgentFlow

[![PyPI version](https://badge.fury.io/py/agentflow.svg)](https://badge.fury.io/py/agentflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AgentFlow is a flexible framework for building and managing AI agent workflows. It provides a structured way to define, execute, and monitor agent-based workflows for various applications.

## Features

- **Flexible Workflow Definition**: Define workflows using JSON configuration
- **Modular Design**: Easy to extend and customize for different use cases
- **Multiple Output Formats**: Support for various output formats including PDF, Word, and Markdown
- **Template System**: Customizable templates for different applications
- **Multilingual Support**: Built-in support for multiple languages including Chinese

## Installation

```bash
pip install agentflow
```

## Quick Start

```python
from agentflow import AgentFlow

# Initialize agent with workflow configuration
agent = AgentFlow('config.json', 'workflows/custom_workflow.json')

# Execute workflow
input_data = {
    'task': 'research',
    'parameters': {
        'topic': 'AI Applications',
        'deadline': '2024-12-31',
        'requirements': 'Focus on practical applications'
    }
}

results = agent.execute_workflow(input_data)
```

## Example Workflows

### Academic Research Workflow

```python
from agentflow.examples.academic import AcademicWorkflow

workflow = AcademicWorkflow()
results = workflow.execute({
    'research_topic': 'AI Applications in Education',
    'deadline': '2024-12-31',
    'academic_level': 'PhD'
})
```

### Customer Service Workflow

```python
from agentflow.examples.customer_service import CustomerServiceWorkflow

workflow = CustomerServiceWorkflow()
results = workflow.execute({
    'customer_query': 'Product return request',
    'customer_id': '12345',
    'priority': 'high'
})
```

## Creating Custom Workflows

1. Define your workflow in JSON:

```json
{
    "AGENT": "Custom_Workflow",
    "CONTEXT": "Your workflow context",
    "WORKFLOW": [
        {
            "step": 1,
            "title": "First Step",
            "description": "Step description",
            "input": ["parameter1", "parameter2"],
            "output": {
                "type": "analysis",
                "format": "json"
            }
        }
    ]
}
```

2. Create your workflow class:

```python
from agentflow import BaseWorkflow

class CustomWorkflow(BaseWorkflow):
    def __init__(self):
        super().__init__('path/to/workflow.json')
        
    def process_step(self, step_number, inputs):
        # Implement step processing
        pass
```

## Documentation

For detailed documentation, visit [https://agentflow.readthedocs.io](https://agentflow.readthedocs.io)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Chen Xingqiang ([@chenxingqiang](https://github.com/chenxingqiang))

## Citation

If you use AgentFlow in your research, please cite:

```bibtex
@software{agentflow2024chen,
  author = {Chen, Xingqiang},
  title = {AgentFlow: A Flexible Framework for AI Agent Workflows},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/chenxingqiang/agentflow}
}
```
```

### LICENSE

```text
MIT License

Copyright (c) 2024 Chen Xingqiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[...]
```