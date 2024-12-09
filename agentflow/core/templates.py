"""
Workflow template management
"""

import os
import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .config_manager import WorkflowConfig, AgentConfig, ProcessorConfig

class TemplateParameter(BaseModel):
    """Template parameter definition"""
    name: str
    description: str
    type: str
    default: Optional[str] = None
    required: bool = True
    options: Optional[List[str]] = None
    
class WorkflowTemplate(BaseModel):
    """Workflow template definition"""
    id: str
    name: str
    description: str
    parameters: List[TemplateParameter]
    workflow: WorkflowConfig
    metadata: Dict[str, str] = Field(default_factory=dict)
    
class TemplateManager:
    """Manager for workflow templates"""
    
    def __init__(self, template_dir: str = None):
        """Initialize template manager
        
        Args:
            template_dir: Directory to store templates
        """
        self.template_dir = template_dir or os.path.expanduser("~/.agentflow/templates")
        self._ensure_template_dir()
        
    def _ensure_template_dir(self):
        """Ensure template directory exists"""
        os.makedirs(self.template_dir, exist_ok=True)
        
    def save_template(self, template: WorkflowTemplate):
        """Save workflow template
        
        Args:
            template: Workflow template
        """
        path = os.path.join(self.template_dir, f"{template.id}.json")
        
        with open(path, "w") as f:
            json.dump(template.dict(), f, indent=2)
            
    def load_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Load workflow template
        
        Args:
            template_id: Template ID
            
        Returns:
            Workflow template if found, None otherwise
        """
        path = os.path.join(self.template_dir, f"{template_id}.json")
        
        if not os.path.exists(path):
            return None
            
        with open(path) as f:
            return WorkflowTemplate(**json.load(f))
            
    def list_templates(self) -> List[WorkflowTemplate]:
        """List all workflow templates
        
        Returns:
            List of workflow templates
        """
        templates = []
        for file in os.listdir(self.template_dir):
            if file.endswith(".json"):
                with open(os.path.join(self.template_dir, file)) as f:
                    templates.append(WorkflowTemplate(**json.load(f)))
        return templates
        
    def delete_template(self, template_id: str) -> bool:
        """Delete workflow template
        
        Args:
            template_id: Template ID
            
        Returns:
            True if template was deleted, False otherwise
        """
        path = os.path.join(self.template_dir, f"{template_id}.json")
        
        if not os.path.exists(path):
            return False
            
        os.remove(path)
        return True
        
    def instantiate_template(self, template_id: str, parameters: Dict[str, str]) -> WorkflowConfig:
        """Create workflow instance from template
        
        Args:
            template_id: Template ID
            parameters: Template parameters
            
        Returns:
            Instantiated workflow configuration
        """
        # Load template
        template = self.load_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
    
        # Validate parameters
        for param in template.parameters:
            if param.required and param.name not in parameters:
                raise ValueError(f"Missing required parameter: {param.name}")
    
            if param.name in parameters and param.options:
                if parameters[param.name] not in param.options:
                    raise ValueError(f"Invalid value for parameter {param.name}")
    
        # Create workflow copy
        workflow = template.workflow.model_copy(deep=True)
    
        # Convert workflow to dict for templating
        workflow_dict = workflow.model_dump()
        workflow_str = json.dumps(workflow_dict)
    
        # Replace placeholders with actual values
        for name, value in parameters.items():
            # Convert list to a string representation for JSON replacement
            if isinstance(value, list):
                # For lists used in list comprehensions, convert to a list of strings
                value_str = f"[{', '.join(str(v) for v in value)}]"
            else:
                value_str = str(value)
    
            # Replace placeholders
            placeholder = f"{{{{ {name} }}}}"
            workflow_str = workflow_str.replace(placeholder, value_str)
    
        # Parse back to workflow config
        workflow_dict = json.loads(workflow_str)
        workflow = WorkflowConfig(**workflow_dict)
    
        return workflow
        
template_manager = TemplateManager()
