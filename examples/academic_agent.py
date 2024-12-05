import ell
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import json
import logging
from dataclasses import dataclass
from enum import Enum
from document_generator import DocumentGenerator, DocumentFormat

class OutputFormat(Enum):
    MARKDOWN = "Markdown"
    LATEX = "LaTeX"
    PLAIN = "Plain"

@dataclass
class TemplateVariables:
    student_needs: Dict[str, Any]
    language: Dict[str, Any]
    template: Dict[str, Any]

class AcademicAgent:
    def __init__(self, config_path: str, agent_path: str):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load configurations
        self.config = self._load_json(config_path)
        self.agent = self._load_json(agent_path)
        
        # Initialize ell
        ell.init(store='./logdir', autocommit=True, verbose=True)
        
        # Initialize template variables
        self.variables = self._initialize_variables()
        
        # Initialize workflow state
        self.state = {}
        
        # Initialize document generator
        self.document_generator = DocumentGenerator(self.config)
        
    def _load_json(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading JSON from {path}: {str(e)}")
            raise
            
    def _initialize_variables(self) -> TemplateVariables:
        """Initialize template variables from config"""
        return TemplateVariables(
            student_needs=self.config['template_variables']['STUDENT_NEEDS'],
            language=self.config['template_variables']['LANGUAGE'],
            template=self.config['template_variables']['TEMPLATE']
        )
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data against configuration rules"""
        try:
            # Validate deadline
            if 'deadline' in input_data:
                deadline = datetime.strptime(input_data['deadline'], '%Y-%m-%d')
                days_until = (deadline - datetime.now()).days
                if not (self.config['validation_rules']['deadline']['min_days'] <= 
                       days_until <= 
                       self.config['validation_rules']['deadline']['max_days']):
                    raise ValueError("Deadline out of acceptable range")
                    
            # Validate word counts
            if 'content' in input_data:
                word_count = len(input_data['content'].split())
                if not (self.config['validation_rules']['word_count']['min'] <= 
                       word_count <= 
                       self.config['validation_rules']['word_count']['max']):
                    raise ValueError("Content length out of acceptable range")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False
            
    def update_variables(self, new_data: Dict[str, Any]) -> None:
        """Update template variables with new data"""
        for key, value in new_data.items():
            if hasattr(self.variables, key):
                setattr(self.variables, key, {**getattr(self.variables, key), **value})
                
    def format_output(self, content: str, format_type: OutputFormat) -> str:
        """Format output according to specified format"""
        if format_type == OutputFormat.MARKDOWN:
            return self._format_markdown(content)
        elif format_type == OutputFormat.LATEX:
            return self._format_latex(content)
        return content
        
    def _format_markdown(self, content: str) -> str:
        # Add markdown formatting
        return content
        
    def _format_latex(self, content: str) -> str:
        # Add LaTeX formatting
        return content
        
    @ell.simple(model="gpt-4o")
    def execute_step(self, step_number: int, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        step = next(s for s in self.agent['WORKFLOW'] if s['step'] == step_number)
        
        # Prepare prompt
        prompt = f"""
        {self.agent['CONTEXT']}
        
        Task: {step['description']}
        
        Input Variables:
        {json.dumps(inputs, indent=2)}
        
        Requirements:
        - Output Format: {step['output']['format']}
        - Word Limit: {self.config['word_count_limits'][f'step_{step_number}']}
        - Details Required: {step['output']['details']}
        """
        
        return prompt
        
    def execute_workflow(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete workflow"""
        try:
            # Validate initial input
            if not self.validate_input(initial_input):
                raise ValueError("Invalid initial input")
                
            # Update variables with initial input
            self.update_variables(initial_input)
            
            results = {}
            
            # Execute each workflow step
            for step in self.agent['WORKFLOW']:
                step_number = step['step']
                step_inputs = self._prepare_step_inputs(step, results)
                
                # Execute step
                step_result = self.execute_step(step_number, step_inputs)
                
                # Format output
                formatted_result = self.format_output(
                    step_result,
                    OutputFormat[step['output']['format'].upper()]
                )
                
                # Store result
                results[f'step_{step_number}'] = formatted_result
                
                # Update state
                self.state[f'step_{step_number}_complete'] = True
                
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise
            
    def _prepare_step_inputs(self, step: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for a workflow step"""
        inputs = {}
        for input_var in step['input']:
            if '.' in input_var:
                obj_name, attr = input_var.split('.')
                if obj_name == 'STUDENT_NEEDS':
                    inputs[attr] = self.variables.student_needs.get(attr)
                elif obj_name == 'LANGUAGE':
                    inputs[attr] = self.variables.language.get(attr)
                elif obj_name == 'WORKFLOW':
                    step_num = int(attr[0])
                    inputs[f'step_{step_num}_result'] = previous_results.get(f'step_{step_num}')
            else:
                inputs[input_var] = getattr(self.variables, input_var.lower(), None)
        return inputs 
        
    def generate_output_document(self, 
                               results: Dict[str, Any], 
                               output_format: str, 
                               output_path: str) -> str:
        """Generate final document from workflow results"""
        try:
            # Prepare document content
            document_content = self._prepare_document_content(results)
            
            # Generate document
            return self.document_generator.generate_document(
                document_content,
                output_format,
                output_path
            )
            
        except Exception as e:
            self.logger.error(f"Document generation failed: {str(e)}")
            raise
            
    def _prepare_document_content(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare content for document generation"""
        return {
            'title': self.variables.student_needs['research_topic'],
            'author': self.variables.student_needs.get('author', ''),
            'sections': [
                {
                    'title': 'Abstract',
                    'content': results['step_1']
                },
                {
                    'title': 'Research Ideas',
                    'content': results['step_2']
                },
                {
                    'title': 'Implementation Plan',
                    'content': results['step_3']
                },
                {
                    'title': 'Timeline',
                    'content': results['step_4']
                },
                {
                    'title': 'Recommendations',
                    'content': results['step_5']
                }
            ],
            'bibliography': []  # Add references if needed
        }