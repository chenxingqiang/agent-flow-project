#!/usr/bin/env python3
from agentflow.api.workflow_server import start_server

if __name__ == "__main__":
    start_server(host='0.0.0.0', port=8000)
