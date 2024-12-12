import os
import sys
import json
import logging
from typing import Dict, Any
from agentflow.core.agent import ResearchAgent
from agentflow.core.config import AgentConfig, ModelConfig

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/xingqiangchen/TASK/APOS/tests/agent_run_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AgentRunTest')

def load_config(config_path: str) -> Dict[str, Any]:
    """Load agent configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return config_data  # Return raw dict instead of AgentConfig
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def run_agent_workflow() -> Dict[str, Any]:
    """Run the research agent workflow"""
    try:
        # Path to the configuration file
        config_path = '/Users/xingqiangchen/TASK/APOS/tests/test_agent_run_config.json'
        
        # Load configuration
        agent_config: Dict[str, Any] = load_config(config_path)
        
        # Print the entire configuration
        print("Full Agent Config:", json.dumps(agent_config, indent=2))
        
        # Create Research Agent
        research_agent = ResearchAgent(AgentConfig(
            **agent_config,
            model=ModelConfig(provider='default', name='default')
        ))
        
        # Prepare initial input
        initial_input = {
            'research_topic': 'AI Applications in Education',
            'deadline': '2024-12-31',
            'academic_level': 'PhD',
            'field': 'Computer Science',
            'special_requirements': 'Focus on machine learning applications',
            'author': 'John Doe'
        }
        
        # Execute workflow
        results = research_agent.execute_workflow(initial_input)
        
        # Generate output documents
        output_dir = '/Users/xingqiangchen/TASK/APOS/tests/agent_outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate comprehensive research report
        report_path = os.path.join(output_dir, 'research_report.md')
        with open(report_path, 'w') as f:
            f.write("# Research Report: AI Applications in Education\n\n")
            
            # Write results from each workflow step
            for step in agent_config.get('WORKFLOW', {}).get('steps', []):
                step_title = step.get('title', f"Step {step.get('step')}")
                step_description = step.get('description', '')
                
                f.write(f"## {step_title}\n")
                f.write(f"{step_description}\n\n")
                
                # Attempt to write step-specific results
                step_results = results.get(step.get('action'), {})
                if step_results:
                    f.write("### Results\n")
                    f.write(json.dumps(step_results, indent=2) + "\n\n")
        
        # Log output paths
        logger.info(f"Generated Research Report: {report_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error during agent workflow execution: {e}")
        raise

def main() -> None:
    try:
        results = run_agent_workflow()
        print("Agent workflow completed successfully.")
        return results
    except Exception as e:
        print(f"Agent workflow failed: {e}")
        return None

if __name__ == '__main__':
    main()
