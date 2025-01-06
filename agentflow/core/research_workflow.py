"""Research workflow module."""

from typing import Dict, Any, Optional, List
import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime

from .workflow import WorkflowEngine, WorkflowInstance
from .types import AgentStatus
from ..agents.agent import Agent

logger = logging.getLogger(__name__)

@dataclass
class ResearchStep:
    """Research step class."""
    id: str
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    
@dataclass
class DistributedConfig:
    """Distributed configuration class."""
    num_workers: int = 1
    batch_size: int = 1
    max_retries: int = 3
    timeout: float = 3600.0

class ResearchDistributedWorkflow(WorkflowEngine):
    """Research distributed workflow class."""
    
    @classmethod
    async def create_remote_workflow(cls, workflow_def: Dict[str, Any], config: Dict[str, Any]) -> Any:
        """Create a remote workflow instance.
        
        Args:
            workflow_def: Workflow definition
            config: Workflow configuration
            
        Returns:
            Remote workflow instance
        """
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        # Create distributed config
        dist_config = DistributedConfig(
            num_workers=config.get("num_workers", 1),
            batch_size=config.get("batch_size", 1),
            max_retries=config.get("max_retries", 3),
            timeout=config.get("timeout", 3600.0)
        )
        
        # Create and return remote actor
        workflow = cls(dist_config)
        remote_workflow = ray.remote(cls).remote(dist_config)
        return remote_workflow
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        """Initialize research distributed workflow.
        
        Args:
            config: Distributed configuration
        """
        super().__init__()
        self.config = config or DistributedConfig()
        self.workers: List[Agent] = []
        self.steps: List[ResearchStep] = []
        self.results: Dict[str, Any] = {}
        
    async def initialize_workers(self, num_workers: Optional[int] = None) -> None:
        """Initialize worker nodes.
        
        Args:
            num_workers: Number of workers to initialize
        """
        if not self._initialized:
            await self.initialize()
            
        n_workers = num_workers or self.config.num_workers
        for _ in range(n_workers):
            worker = Agent()
            await worker.initialize()
            self.workers.append(worker)
            
    async def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of inputs.
        
        Args:
            batch: Batch of inputs to process
            
        Returns:
            List of results
        """
        if not self.workers:
            await self.initialize_workers()
            
        tasks = []
        for i, item in enumerate(batch):
            worker = self.workers[i % len(self.workers)]
            task = asyncio.create_task(worker.process_message(str(item)))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        return [{"result": result} for result in results]
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow.
        
        Args:
            input_data: Input data for workflow
            
        Returns:
            Execution result
            
        Raises:
            ray.exceptions.RayTaskError: If input data is invalid or execution fails
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Validate input data
            if not isinstance(input_data, dict):
                raise ValueError("Input data must be a dictionary")
                
            # Process input data in batches
            batch_size = self.config.batch_size
            batches = [
                input_data.get("items", [])[i:i + batch_size]
                for i in range(0, len(input_data.get("items", [])), batch_size)
            ]
            
            results = []
            for batch in batches:
                batch_results = await self.process_batch(batch)
                results.extend(batch_results)
                
            return {
                "status": "success",
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            import ray
            logger.error(f"Workflow execution failed: {str(e)}")
            raise ray.exceptions.RayTaskError(str(e))
            
    async def cleanup(self) -> None:
        """Clean up workflow resources."""
        if not self._initialized:
            return
            
        # Clean up workers
        for worker in self.workers:
            await worker.cleanup()
        self.workers.clear()
        
        # Clean up base class
        await super().cleanup()
        
    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously.
        
        Args:
            input_data: Input data for workflow
            
        Returns:
            Execution result
        """
        return await self.execute(input_data)
        
    async def set_distributed_steps(self, steps: List[str]) -> None:
        """Set distributed steps.
        
        Args:
            steps: List of step IDs to distribute
        """
        self.distributed_steps = steps
        
    async def get_distributed_steps(self) -> List[str]:
        """Get distributed steps.
        
        Returns:
            List of distributed step IDs
        """
        return getattr(self, "distributed_steps", [])