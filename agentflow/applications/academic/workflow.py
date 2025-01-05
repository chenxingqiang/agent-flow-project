from typing import Dict, Any
import ell
from ...core.workflow import BaseWorkflow

class AcademicWorkflow(BaseWorkflow):
    """Academic paper workflow implementation"""
    
    @ell2a.simple(model="gpt-4o")
    def process_step(self, step_number: int, inputs: Dict[str, Any]) -> Any:
        """Process academic workflow step"""
        step = next(s for s in self.workflow_def['WORKFLOW'] if s['step'] == step_number)
        
        prompt = f"""
        {self.workflow_def['CONTEXT']}
        
        Task: {step['description']}
        
        Input Variables:
        {inputs}
        
        Requirements:
        - Output Format: {step['output']['format']}
        - Word Limit: {self.config['word_count_limits'][f'step_{step_number}']}
        - Details Required: {step['output']['details']}
        """
        
        return prompt 