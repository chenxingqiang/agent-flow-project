"""Research workflow module."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
import logging
from pydantic import BaseModel
from .workflow import Workflow
from .node import AgentNode

logger = logging.getLogger(__name__)

@dataclass
class ResearchStep:
    """Research workflow step."""
    name: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    status: str = "pending"
    error: Optional[str] = None

@dataclass
class DistributedConfig:
    """Configuration for distributed workflow execution."""
    num_workers: int = 4
    batch_size: int = 10
    timeout: float = 30.0
    retry_limit: int = 3
    worker_init_timeout: float = 5.0

class ResearchDistributedWorkflow(Workflow):
    """Distributed workflow implementation for research tasks."""
    
    def __init__(self, name: str, config: Optional[DistributedConfig] = None):
        super().__init__(name)
        self.config = config or DistributedConfig()
        self.worker_pool: List[AgentNode] = []
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.is_running: bool = False
        self.workers_initialized: bool = False
    
    async def initialize_workers(self) -> None:
        """Initialize the worker pool."""
        if self.workers_initialized:
            return
            
        for i in range(self.config.num_workers):
            worker = AgentNode(f"worker_{i}", f"Worker node {i}")
            try:
                await asyncio.wait_for(
                    worker.initialize(),
                    timeout=self.config.worker_init_timeout
                )
                self.worker_pool.append(worker)
            except asyncio.TimeoutError:
                logger.error(f"Worker {i} initialization timed out")
            except Exception as e:
                logger.error(f"Failed to initialize worker {i}: {str(e)}")
        
        self.workers_initialized = True
    
    async def process_batch(self, worker: AgentNode, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of tasks using a worker node."""
        try:
            results = []
            for task in batch:
                result = await asyncio.wait_for(
                    worker.process_task(task),
                    timeout=self.config.timeout
                )
                results.append(result)
            return results
        except asyncio.TimeoutError:
            logger.error(f"Batch processing timed out for worker {worker.name}")
            return [{"status": "error", "error": "timeout"} for _ in batch]
        except Exception as e:
            logger.error(f"Error processing batch on worker {worker.name}: {str(e)}")
            return [{"status": "error", "error": str(e)} for _ in batch]
    
    async def worker_loop(self, worker: AgentNode) -> None:
        """Main loop for a worker node."""
        while self.is_running:
            try:
                # Get batch of tasks
                batch = []
                for _ in range(self.config.batch_size):
                    try:
                        task = await asyncio.wait_for(
                            self.task_queue.get(),
                            timeout=1.0
                        )
                        batch.append(task)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    continue
                
                # Process batch
                results = await self.process_batch(worker, batch)
                
                # Put results in queue
                for result in results:
                    await self.result_queue.put(result)
                    
            except Exception as e:
                logger.error(f"Error in worker loop for {worker.name}: {str(e)}")
    
    async def start(self) -> None:
        """Start the distributed workflow."""
        await self.initialize_workers()
        if not self.worker_pool:
            raise RuntimeError("No workers available")
        
        self.is_running = True
        worker_tasks = [
            asyncio.create_task(self.worker_loop(worker))
            for worker in self.worker_pool
        ]
        
        await asyncio.gather(*worker_tasks)
    
    async def stop(self) -> None:
        """Stop the distributed workflow."""
        self.is_running = False
        for worker in self.worker_pool:
            await worker.cleanup()
        self.worker_pool.clear()
        self.workers_initialized = False
    
    async def submit_task(self, task: Dict[str, Any]) -> None:
        """Submit a task to the workflow."""
        if not self.is_running:
            raise RuntimeError("Workflow is not running")
        await self.task_queue.put(task)
    
    async def get_result(self) -> Dict[str, Any]:
        """Get a result from the workflow."""
        if not self.is_running:
            raise RuntimeError("Workflow is not running")
        return await self.result_queue.get()
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow with the given context."""
        if not self.is_running:
            await self.start()
        
        try:
            await self.submit_task(context)
            result = await self.get_result()
            return result
        except Exception as e:
            logger.error(f"Error executing workflow: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

class ResearchWorkflow:
    """Research workflow implementation."""
    
    def __init__(self, config: BaseModel):
        """Initialize research workflow.
        
        Args:
            config: Workflow configuration
        """
        config_dict = config.model_dump()
        self.name = config_dict.get("name", "default_research")
        self.steps: List[ResearchStep] = []
        self.current_step = 0
        self.status = "initialized"
        self.results = {}
        self.config = config_dict
        
    async def execute(self) -> Dict[str, Any]:
        """Execute research workflow.
        
        Returns:
            Workflow results
        """
        self.status = "running"
        try:
            while self.current_step < len(self.steps):
                step = self.steps[self.current_step]
                try:
                    # Execute step
                    await self._execute_step(step)
                    self.current_step += 1
                except Exception as e:
                    step.status = "error"
                    step.error = str(e)
                    logger.error(f"Error executing step {step.name}: {e}")
                    raise
            
            self.status = "completed"
            return self.results
            
        except Exception as e:
            self.status = "error"
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def _execute_step(self, step: ResearchStep):
        """Execute a single workflow step.
        
        Args:
            step: Workflow step to execute
        """
        logger.info(f"Executing step: {step.name}")
        step.status = "running"
        
        try:
            # Process step inputs
            processed_inputs = await self._process_inputs(step.inputs)
            
            # Execute step logic
            outputs = await self._process_step(step.name, processed_inputs)
            
            # Update step outputs and status
            step.outputs = outputs
            step.status = "completed"
            
            # Update workflow results
            self.results[step.name] = outputs
            
        except Exception as e:
            step.status = "error"
            step.error = str(e)
            raise
    
    async def _process_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process step inputs.
        
        Args:
            inputs: Step inputs
            
        Returns:
            Processed inputs
        """
        processed = {}
        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to previous step output
                step_name = value[1:].split(".")[0]
                if step_name in self.results:
                    processed[key] = self.results[step_name]
                else:
                    raise ValueError(f"Referenced step {step_name} not found in results")
            else:
                processed[key] = value
        return processed
    
    async def _process_step(self, step_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a workflow step.
        
        Args:
            step_name: Name of the step
            inputs: Processed step inputs
            
        Returns:
            Step outputs
        """
        # TODO: Implement actual step processing logic
        return {
            "status": "success",
            "step": step_name,
            "inputs": inputs,
            "output": f"Processed {step_name}"
        }
    
    def add_step(self, name: str, description: str, inputs: Dict[str, Any]):
        """Add a step to the workflow.
        
        Args:
            name: Step name
            description: Step description
            inputs: Step inputs
        """
        step = ResearchStep(
            name=name,
            description=description,
            inputs=inputs,
            outputs={}
        )
        self.steps.append(step)
    
    def get_status(self) -> Dict[str, Any]:
        """Get workflow status.
        
        Returns:
            Workflow status information
        """
        return {
            "name": self.name,
            "status": self.status,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "steps": [
                {
                    "name": step.name,
                    "status": step.status,
                    "error": step.error
                }
                for step in self.steps
            ]
        }
