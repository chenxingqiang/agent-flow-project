# AgentFlow Documentation

## Overview

AgentFlow is a powerful and flexible AI Agent Workflow Management System that enables the creation, configuration, and management of intelligent agents. It provides a comprehensive framework for building agent-based applications with features like dynamic configuration, advanced visualization, and real-time monitoring.

## Table of Contents

1. [Installation](#installation)
2. [Core Components](#core-components)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Visualization](#visualization)
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

The system uses a flexible DSL (Domain Specific Language) for configuring agents:

```python
from agentflow.core.config import AgentConfig

agent_config = {
    "agent": {
        "name": "research_agent",
        "version": "1.0.0",
        "type": "research"
    },
    "input_specification": {
        "MODES": ["DIRECT_INPUT", "CONTEXT_INJECTION"],
        "TYPES": {
            "CONTEXT": {
                "sources": ["PREVIOUS_AGENT_OUTPUT"]
            }
        }
    },
    "output_specification": {
        "MODES": ["RETURN", "FORWARD"],
        "STRATEGIES": {
            "RETURN": {
                "options": ["FULL_RESULT"]
            }
        }
    }
}

config = AgentConfig(**agent_config)
```

### Input/Output Processing

The system provides flexible input and output processing:

```python
from agentflow.core.input_processor import InputProcessor
from agentflow.core.output_processor import OutputProcessor

# Process input
input_processor = InputProcessor(config.input_specification)
processed_input = input_processor.process_input(data, mode="DIRECT_INPUT")

# Process output
output_processor = OutputProcessor(config.output_specification)
processed_output = output_processor.process_output(result, mode="RETURN")
```

### Flow Control

Control the flow of data and execution:

```python
from agentflow.core.flow_controller import FlowController

flow_config = {
    "ROUTING_RULES": {
        "DEFAULT_BEHAVIOR": "FORWARD_ALL",
        "CONDITIONAL_ROUTING": {
            "CONDITIONS": [
                {
                    "when": "data.get('type') == 'special'",
                    "action": "TRANSFORM"
                }
            ]
        }
    }
}

controller = FlowController(flow_config)
result = controller.route_data(data)
```

## Visualization

AgentFlow provides powerful visualization capabilities through integration with ell.studio.

### Basic Usage

```python
from agentflow.visualization.service import VisualizationService

# Create visualization service
service = VisualizationService({
    "api_key": "your_ell_studio_api_key",
    "project_id": "your_project_id"
})

# Start service
service.start()
```

### Real-time Visualization

```python
import asyncio
import websockets
import json

async def monitor_agent():
    async with websockets.connect("ws://localhost:8001/live") as websocket:
        while True:
            # Send agent status
            await websocket.send(json.dumps({
                "type": "agent_status",
                "data": {"status": "processing"}
            }))
            
            # Receive updates
            response = await websocket.recv()
            print(json.loads(response))
            
            await asyncio.sleep(1)

# Run monitoring
asyncio.run(monitor_agent())
```

## API Reference

### REST API Endpoints

- `POST /visualize/agent`: Visualize agent configuration
- `POST /visualize/workflow`: Visualize workflow configuration
- `WS /live`: WebSocket endpoint for real-time updates

### Python API

#### Agent Configuration

```python
from agentflow.core.config import AgentConfig

# Create configuration
config = AgentConfig.from_dict({...})

# Convert to dictionary
config_dict = config.to_dict()
```

#### Visualization Components

```python
from agentflow.visualization.components import VisualGraph, VisualNode, VisualEdge

# Create graph
graph = VisualGraph()

# Add nodes and edges
graph.add_node(VisualNode(...))
graph.add_edge(VisualEdge(...))

# Convert to ell.studio format
ell_data = graph.to_ell_format()
```

## Examples

### Basic Agent Setup

```python
from agentflow.core.agent import Agent
from agentflow.core.config import AgentConfig

# Create agent configuration
config = AgentConfig(...)

# Initialize agent
agent = Agent(config)

# Process data
result = agent.process({"input": "data"})
```

### Workflow Visualization

```python
from agentflow.visualization.renderer import AgentVisualizer

# Create visualizer
visualizer = AgentVisualizer()

# Visualize workflow
visual_data = visualizer.visualize_workflow({
    "name": "Research Workflow",
    "steps": [
        {"name": "Data Collection", "type": "input"},
        {"name": "Analysis", "type": "processor"},
        {"name": "Report Generation", "type": "output"}
    ]
})
```

## Development

### Project Structure

```
agentflow/
├── core/
│   ├── agent.py
│   ├── config.py
│   ├── input_processor.py
│   ├── output_processor.py
│   └── flow_controller.py
├── visualization/
│   ├── components.py
│   ├── renderer.py
│   └── service.py
├── api/
│   └── base_service.py
└── examples/
    └── workflow_example.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/core/test_agent.py

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
