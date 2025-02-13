"""Ell Studio integration for AgentFlow monitoring and visualization."""
import os
from typing import Dict, Any, Optional, List

from agentflow.ell2a.tracers import SQLiteTracer
from agentflow.ell2a.callbacks import PromptCallback, CompletionCallback
from datetime import datetime

class EllMonitor:
    """Ell Studio monitoring integration for AgentFlow."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ell monitor.
        
        Args:
            config: Configuration dictionary containing:
                - storage_dir: Directory for storing Ell data
                - project_name: Name of the project
                - version: Version of the project
        """
        self.config = config
        self.storage_dir = config.get('storage_dir', './ell2a_data')
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize Ell
        ell2a.init(
            store=self.storage_dir,
            project=config.get('project_name', 'agentflow'),
            version=config.get('version', '1.0.0')
        )
        
        # Initialize tracer
        self.tracer = SQLiteTracer(
            db_path=os.path.join(self.storage_dir, 'traces.db')
        )
        
        # Initialize callbacks
        self.prompt_callback = PromptCallback()
        self.completion_callback = CompletionCallback()
        
        # Register callbacks
        ell2a.register_callback(self.prompt_callback)
        ell2a.register_callback(self.completion_callback)
        
    def track_agent_execution(self, agent_id: str, prompt: str, completion: str,
                            metadata: Optional[Dict[str, Any]] = None):
        """Track agent execution in Ell.
        
        Args:
            agent_id: ID of the agent
            prompt: Input prompt
            completion: Model completion
            metadata: Additional metadata
        """
        metadata = metadata or {}
        metadata.update({
            'agent_id': agent_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Track with Ell
        with ell2a.trace() as trace:
            trace.log_prompt(prompt)
            trace.log_completion(completion)
            trace.log_metadata(metadata)
            
    def track_workflow_execution(self, workflow_id: str, steps: List[Dict[str, Any]],
                               metadata: Optional[Dict[str, Any]] = None):
        """Track workflow execution in Ell.
        
        Args:
            workflow_id: ID of the workflow
            steps: List of workflow steps
            metadata: Additional metadata
        """
        metadata = metadata or {}
        metadata.update({
            'workflow_id': workflow_id,
            'timestamp': datetime.utcnow().isoformat(),
            'num_steps': len(steps)
        })
        
        # Track with Ell
        with ell2a.trace() as trace:
            for step in steps:
                trace.log_prompt(step.get('input', ''))
                trace.log_completion(step.get('output', ''))
                trace.log_metadata({
                    **metadata,
                    'step_id': step.get('id'),
                    'step_name': step.get('name'),
                    'step_type': step.get('type')
                })
                
    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get metrics for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary containing agent metrics
        """
        traces = self.tracer.get_traces(
            filter_fn=lambda t: t.metadata.get('agent_id') == agent_id
        )
        
        return {
            'total_executions': len(traces),
            'average_latency': sum(t.duration for t in traces) / len(traces) if traces else 0,
            'success_rate': sum(1 for t in traces if t.metadata.get('status') == 'success') / len(traces) if traces else 0
        }
        
    def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get metrics for a specific workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Dictionary containing workflow metrics
        """
        traces = self.tracer.get_traces(
            filter_fn=lambda t: t.metadata.get('workflow_id') == workflow_id
        )
        
        return {
            'total_executions': len(traces),
            'average_steps': sum(t.metadata.get('num_steps', 0) for t in traces) / len(traces) if traces else 0,
            'success_rate': sum(1 for t in traces if t.metadata.get('status') == 'success') / len(traces) if traces else 0
        }
        
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get visualization data for Ell Studio.
        
        Returns:
            Dictionary containing visualization data
        """
        traces = self.tracer.get_traces()
        
        return {
            'traces': [
                {
                    'id': trace.id,
                    'timestamp': trace.metadata.get('timestamp'),
                    'type': 'agent' if trace.metadata.get('agent_id') else 'workflow',
                    'entity_id': trace.metadata.get('agent_id') or trace.metadata.get('workflow_id'),
                    'duration': trace.duration,
                    'status': trace.metadata.get('status'),
                    'prompt': trace.prompts[0] if trace.prompts else None,
                    'completion': trace.completions[0] if trace.completions else None,
                    'metadata': trace.metadata
                }
                for trace in traces
            ]
        }
        
    def launch_studio(self, host: str = "localhost", port: int = 8002):
        """Launch Ell Studio server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        import subprocess
        subprocess.Popen([
            'ell-studio',
            '--storage', self.storage_dir,
            '--host', host,
            '--port', str(port)
        ])
