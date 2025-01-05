"""
Workflow template management
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime

from ..agents.agent_types import AgentConfig
from .workflow_types import WorkflowConfig
from .config_manager import ProcessorConfig

class TemplateParameter(BaseModel):
    """Template parameter model."""
    model_config = ConfigDict(frozen=False, validate_assignment=True)
    
    name: str
    type: str
    description: Optional[str] = None
    default: Optional[Any] = None
    required: bool = True
    options: Optional[List[Any]] = None
    
    def validate(self, value: Any) -> bool:
        """Validate parameter value."""
        if value is None and self.required and self.default is None:
            return False
            
        if value is None and (not self.required or self.default is not None):
            return True
            
        if self.options is not None and value not in self.options:
            return False
            
        if self.type == "string":
            return isinstance(value, (str, int, float, bool))
        elif self.type == "integer":
            try:
                int(str(value))
                return True
            except (ValueError, TypeError):
                return False
        elif self.type == "float":
            try:
                float(str(value))
                return True
            except (ValueError, TypeError):
                return False
        elif self.type == "boolean":
            return isinstance(value, (bool, int, str))
        elif self.type == "list":
            return isinstance(value, (list, tuple))
        elif self.type == "dict":
            return isinstance(value, dict)
        return True
        
    def convert_value(self, value: Any) -> Any:
        """Convert parameter value to correct type."""
        if value is None:
            if self.default is not None:
                return self.default
            elif not self.required:
                return None
            else:
                raise ValueError(f"Required parameter {self.name} has no value or default")
            
        if self.type == "string":
            return str(value)
        elif self.type == "integer":
            return int(str(value))
        elif self.type == "float":
            return float(str(value))
        elif self.type == "boolean":
            if isinstance(value, bool):
                return value
            elif isinstance(value, int):
                return bool(value)
            elif isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)
        elif self.type == "list":
            if isinstance(value, tuple):
                return list(value)
            return value
        return value
        
    def set_default(self, value: Any) -> None:
        """Set default value."""
        if value is not None and not self.validate(value):
            raise ValueError(f"Invalid default value for parameter {self.name}: {value}")
        self.default = value
        self.required = value is None  # Make required if default is None, not required otherwise

class WorkflowTemplate(BaseModel):
    """Workflow template model."""
    model_config = ConfigDict(frozen=False, validate_assignment=True)
    
    id: str
    name: str
    description: Optional[str] = None
    parameters: List[TemplateParameter] = Field(default_factory=list)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    agents: List[Dict[str, Any]] = Field(default_factory=list)
    processors: List[Dict[str, Any]] = Field(default_factory=list)
    connections: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    workflow: Union[Dict[str, Any], WorkflowConfig]
    
    @field_validator('workflow')
    @classmethod
    def validate_workflow(cls, v):
        """Validate workflow field."""
        if isinstance(v, dict):
            return WorkflowConfig(**v)
        return v
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and process template parameters."""
        validated = {}
        missing = []
        
        for param in self.parameters:
            if param.name in params:
                value = params[param.name]
                if param.validate(value):
                    validated[param.name] = param.convert_value(value)
                else:
                    raise ValueError(f"Invalid value for parameter {param.name}: {value}")
            elif param.required and param.default is None:
                missing.append(param.name)
            else:
                validated[param.name] = param.default
                
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
            
        return validated
    
    def _replace_parameters(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Replace parameter placeholders in data.
        
        Args:
            data: Data structure containing parameter placeholders
            params: Parameter values to substitute
            
        Returns:
            Dict with parameter placeholders replaced with values
        """
        result = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = self._replace_parameters(value, params)
            elif isinstance(value, list):
                result[key] = [
                    self._replace_parameters(item, params) if isinstance(item, dict)
                    else self._replace_parameter(item, params) if isinstance(item, str)
                    else item
                    for item in value
                ]
            elif isinstance(value, str):
                result[key] = self._replace_parameter(value, params)
            else:
                result[key] = value
                
        return result
        
    def _replace_parameter(self, value: str, params: Dict[str, Any]) -> Any:
        """Replace a single parameter placeholder.
        
        Args:
            value: String potentially containing parameter placeholder
            params: Parameter values to substitute
            
        Returns:
            Substituted value if placeholder found, original value otherwise
        """
        if value.startswith("$"):
            param_name = value[1:]
            return params.get(param_name, value)
        return value
    
    def create_workflow(self, params: Dict[str, Any]) -> WorkflowConfig:
        """Create workflow from template."""
        validated_params = self.validate_parameters(params)
        workflow_dict = self.workflow.model_dump()
        
        # Replace parameter placeholders in workflow configuration
        workflow_dict = self._replace_parameters(workflow_dict, validated_params)
        
        return WorkflowConfig(**workflow_dict)

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
    
        # Replace placeholders with actual values
        for name, value in parameters.items():
            param = next((p for p in template.parameters if p.name == name), None)
            if param and param.type == "list":
                # For list parameters, replace directly in the workflow dict
                workflow_dict[name] = value
            else:
                # For other parameters, do string replacement
                value_str = str(value)
                placeholder = f"{{{{ {name} }}}}"
                for k, v in workflow_dict.items():
                    if isinstance(v, str):
                        workflow_dict[k] = v.replace(placeholder, value_str)
    
        # Parse back to workflow config
        workflow = WorkflowConfig(**workflow_dict)
    
        return workflow
        
template_manager = TemplateManager()
