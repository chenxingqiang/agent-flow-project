from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path
import json

from .workflow import BaseWorkflow
from .config import AgentConfig, ModelConfig, WorkflowConfig
from .research_workflow import ResearchWorkflow
from .document import DocumentGenerator

class Agent:
    """Main agent class for managing workflows"""
    
    def __init__(self, config: Union[AgentConfig, str], workflow_def_path: Optional[str] = None):
        """Initialize Agent with configuration"""
        # Setup logger first to ensure it exists
        self.logger = logging.getLogger(__name__)
        
        try:
            if isinstance(config, str):
                # Load config from file
                with open(config, 'r') as f:
                    config_data = json.load(f)
                
                # Remove any invalid or extra keys
                valid_keys = {'agent_type', 'model', 'workflow'}
                config_data = {k: v for k, v in config_data.items() if k in valid_keys}
                
                # Add default values if missing
                if 'agent_type' not in config_data:
                    config_data['agent_type'] = 'research'
                if 'model' not in config_data:
                    config_data['model'] = {
                        'provider': 'openai',
                        'name': 'gpt-4',
                        'temperature': 0.5
                    }
                if 'workflow' not in config_data:
                    config_data['workflow'] = {
                        'max_iterations': 10,
                        'logging_level': 'INFO'
                    }
                
                # Load workflow definition if provided
                workflow_def = None
                if workflow_def_path:
                    with open(workflow_def_path, 'r') as f:
                        workflow_def = json.load(f)
                
                config = AgentConfig(**config_data)
            else:
                workflow_def = None
            
            self.config = config
            self.state = {}
            
            # Add properties for easier access
            self.agent_type = config.agent_type
            self.model = config.model
            self.workflow = config.workflow
            self.is_distributed = config.workflow.distributed
            self.workflow_def = workflow_def
            
            # Initialize distributed components if needed
            if self.is_distributed:
                import ray
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                self.ray_actor = ray.remote(self.__class__).remote(config)
            else:
                self.ray_actor = None
                
            # Validate config
            self.config.model_dump()
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {str(e)}")
            raise

    def execute_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow synchronously"""
        try:
            if self.config.agent_type == 'research':
                workflow = ResearchWorkflow(
                    config=self.config.model_dump(), 
                    workflow_def=self.workflow_def
                )
            else:
                raise ValueError(f"Unsupported agent type: {self.config.agent_type}")
            
            results = workflow.execute(input_data)
            
            # Format results to match expected output
            formatted_results = {}
            for step_key, step_result in results.items():
                if isinstance(step_result, dict):
                    formatted_results[step_key] = step_result
            
            # Validate results
            self.validate_results(formatted_results)
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise

    def execute_workflow_async(self, input_data: Dict[str, Any]) -> Any:
        """Execute workflow asynchronously
        
        Args:
            input_data: Initial input data for workflow
            
        Returns:
            Ray ObjectRef for async execution
        """
        try:
            if not self.is_distributed:
                # Initialize Ray if not already initialized
                import ray
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                self.ray_actor = ray.remote(self.__class__).remote(self.config)
                self.is_distributed = True

            # Create remote workflow and execute
            return self.ray_actor.execute_workflow.remote(input_data)
            
        except Exception as e:
            self.logger.error(f"Async workflow execution failed: {str(e)}")
            raise

    def generate_output_document(self, content: Dict[str, Any], output_format: str, output_path: str) -> str:
        """Generate output document in specified format
        
        Args:
            content: Content to generate document from
            output_format: Format of the output document
            output_path: Path to save the output document
            
        Returns:
            Path to generated document
        """
        generator = DocumentGenerator(self.config)
        return generator.generate(content, output_format, output_path)

    def validate_results(self, results: Dict[str, Any]) -> None:
        """Validate workflow results"""
        if not results:
            raise ValueError("Empty workflow results")
            
        # Validate research output
        if "research_output" not in results and "step_1" not in results:
            raise ValueError("Missing research output in results")
            
        # If using step format, validate first step
        if "step_1" in results:
            step_result = results["step_1"]
            if not isinstance(step_result, dict):
                raise ValueError("Invalid step result format")
                
            if "result" not in step_result:
                raise ValueError("Missing result in step output")

def main():
    """CLI entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Execute Agent workflow')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_data = json.load(f)
        
    with open(args.input, 'r') as f:
        input_data = json.load(f)
        
    config = AgentConfig(**config_data)
    agent = Agent(config)
    results = agent.execute_workflow(input_data)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()