"""Configuration classes for AgentFlow."""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pathlib import Path
import logging
import json
import uuid

class AgentMetadata(BaseModel):
    """Agent元数据配置"""
    name: str
    version: str
    type: str

class InputSpec(BaseModel):
    """输入规范定义"""
    types: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)

class OutputSpec(BaseModel):
    """输出规范定义"""
    types: Dict[str, Any] = Field(default_factory=dict)
    format: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)

class DataFlowControl(BaseModel):
    """数据流控制配置"""
    routing_rules: Dict[str, Any] = Field(default_factory=dict)
    error_handling: Dict[str, Any] = Field(default_factory=dict)

class InterfaceContract(BaseModel):
    """接口契约配置"""
    input_contract: Dict[str, List[str]] = Field(default_factory=dict)
    output_contract: Dict[str, List[str]] = Field(default_factory=dict)

class ModelConfig(BaseModel):
    """Configuration for AI model"""
    provider: str = Field(default="openai", description="AI model provider")
    name: str = Field(default="gpt-4", description="Specific model name")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum token limit")
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature value."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0 and 1")
        return v

class AgentConfig(BaseModel):
    """Agent配置模型"""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = "default"
    agent_type: Optional[str] = "default"
    
    # 模型和执行配置
    model: Optional[ModelConfig] = Field(default_factory=ModelConfig)
    system_prompt: Optional[str] = None
    tools: Optional[List[str]] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # 输入输出规范
    input_specification: Optional[InputSpec] = Field(default_factory=InputSpec)
    output_specification: Optional[OutputSpec] = Field(default_factory=OutputSpec)
    
    # 执行策略
    execution_policies: Optional[Dict[str, Any]] = None
    
    # 元数据和额外配置
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """
        从字典创建AgentConfig，支持多种配置格式
        
        :param config_dict: 配置字典
        :return: AgentConfig实例
        """
        # 处理旧版本配置
        if 'agent' in config_dict:
            agent_config = config_dict.get('agent', {})
            config_dict.update(agent_config)
            config_dict.pop('agent', None)
        
        # 处理嵌套的工作流配置
        if 'workflow' in config_dict:
            workflow_config = config_dict.get('workflow', {})
            config_dict['execution_policies'] = workflow_config
            config_dict.pop('workflow', None)
        
        # 处理类型映射
        if 'type' not in config_dict and 'agent_type' in config_dict:
            config_dict['type'] = config_dict['agent_type']
        
        # 处理默认值和可选字段
        config_dict.setdefault('id', str(uuid.uuid4()))
        config_dict.setdefault('type', 'default')
        config_dict.setdefault('agent_type', 'default')
        
        # 处理输入输出规范
        if 'input_specification' in config_dict and not isinstance(config_dict['input_specification'], InputSpec):
            config_dict['input_specification'] = InputSpec(**config_dict['input_specification'])
        
        if 'output_specification' in config_dict and not isinstance(config_dict['output_specification'], OutputSpec):
            config_dict['output_specification'] = OutputSpec(**config_dict['output_specification'])
        
        return cls(**config_dict)

class ProcessorConfig(BaseModel):
    """Processor configuration"""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    type: str = "processor"
    processor: Any  # Can be string (import path) or class
    config: Dict[str, Any] = Field(default_factory=dict)

class StepConfig(BaseModel):
    """Configuration for workflow steps"""
    id: str = Field(..., description="Step ID")
    step: int = Field(..., description="Step number in workflow")
    name: str = Field(..., description="Step name")
    agents: List[str] = Field(default_factory=list, description="List of agent IDs")
    input_type: str = Field(..., description="Input type for the step")
    output_type: str = Field(..., description="Output type for the step")
    input: List[str] = Field(default_factory=list, description="Required input fields")
    output: Dict[str, Any] = Field(default_factory=dict, description="Output configuration")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ExecutionPolicies(BaseModel):
    """Configuration for workflow execution policies"""
    required_fields: List[str] = Field(default_factory=list, description="Required input fields")
    default_status: str = Field(default="initialized", description="Default workflow status")
    error_handling: Dict[str, Any] = Field(default_factory=dict, description="Error handling policies")
    steps: List[StepConfig] = Field(default_factory=list, description="Workflow steps")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class WorkflowConfig(BaseModel):
    """工作流配置模型"""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    
    # 工作流执行参数
    max_iterations: int = 10
    timeout: Optional[int] = None
    logging_level: str = 'INFO'
    
    # 代理和连接配置
    agents: Optional[List[AgentConfig]] = None
    connections: Optional[List[Dict[str, Any]]] = None
    
    # 处理器配置
    processors: Optional[List[ProcessorConfig]] = None
    
    # 执行策略
    execution_policies: Optional[ExecutionPolicies] = None
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, workflow_dict: Dict[str, Any]) -> 'WorkflowConfig':
        """
        从字典创建WorkflowConfig
        
        :param workflow_dict: 工作流配置字典
        :return: WorkflowConfig实例
        """
        # 处理代理配置
        if 'agents' in workflow_dict:
            workflow_dict['agents'] = [
                AgentConfig.from_dict(agent) if isinstance(agent, dict) else agent
                for agent in workflow_dict['agents']
            ]
        
        # 处理处理器配置
        if 'processors' in workflow_dict:
            workflow_dict['processors'] = [
                ProcessorConfig.from_dict(processor) if isinstance(processor, dict) else processor
                for processor in workflow_dict['processors']
            ]
        
        # 处理执行策略
        if 'execution_policies' in workflow_dict:
            workflow_dict['execution_policies'] = ExecutionPolicies(**workflow_dict['execution_policies'])
        
        # 处理默认值
        workflow_dict.setdefault('id', str(uuid.uuid4()))
        workflow_dict.setdefault('max_iterations', 10)
        workflow_dict.setdefault('logging_level', 'INFO')
        
        return cls(**workflow_dict)

class ConfigManager:
    """Configuration manager for handling config files and validation"""
    
    def __init__(self, config_path: str):
        """Initialize config manager
        
        Args:
            config_path: Path to config file
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path) as f:
                config = json.load(f)
                
            # Add variables section if not present
            if 'variables' not in config:
                config['variables'] = {}
                
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise
            
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If config is invalid
        """
        required_sections = {'variables', 'agent_type', 'model', 'workflow'}
        missing = required_sections - set(config.keys())
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
            
        # Validate model config
        model_config = config.get('model', {})
        if not all(k in model_config for k in ['provider', 'name']):
            raise ValueError("Model config missing required fields")
            
        # Validate workflow config
        workflow_config = config.get('workflow', {})
        if not isinstance(workflow_config, dict):
            raise ValueError("Workflow config must be a dictionary")
            
    def extract_variables(self) -> Dict[str, Any]:
        """Extract variables from config
        
        Returns:
            Dictionary of variables
        """
        return self.config.get('variables', {})
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration
        
        Args:
            updates: Configuration updates to apply
            
        Raises:
            ValueError: If updates are invalid
        """
        # Validate updates
        for key, value in updates.items():
            if key not in self.config:
                raise ValueError(f"Invalid config key: {key}")
                
            if key == 'variables':
                if not isinstance(value, dict):
                    raise ValueError("Variables must be a dictionary")
                    
        # Apply updates
        self.config.update(updates)
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

class ConfigValidator:
    @staticmethod
    def validate_workflow_config(config: Dict[str, Any], workflow_def: Dict[str, Any]):
        """Validate workflow configuration"""
        if not config:
            raise ValueError("Empty configuration")
            
        if not workflow_def or 'WORKFLOW' not in workflow_def:
            raise ValueError("Invalid workflow definition")
            
        # Validate step configurations
        for step in workflow_def['WORKFLOW']:
            step_num = step['step']
            step_config_key = f'step_{step_num}_config'
            
            if step_config_key not in config:
                raise ValueError(f"Missing configuration for step {step_num}")
                
            step_config = config[step_config_key]
            if not isinstance(step_config, dict):
                raise ValueError(f"Invalid configuration for step {step_num}")

# Example usage
def main():
    # Create config from dictionary
    config_dict = {
        "agent_type": "research",
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.7
        },
        "workflow": {
            "max_iterations": 5,
            "logging_level": "DEBUG"
        },
        "name": "Academic Research Assistant",
        "description": "AI agent for academic research and paper writing",
        "skill_tags": ["research", "academic_writing", "literature_review"]
    }
    
    agent_config = AgentConfig(**config_dict)
    print(agent_config.model_dump())
    
    # Save and load config
    with open("agent_config.json", "w") as f:
        json.dump(agent_config.to_dict(), f, indent=2)
    
    loaded_config = AgentConfig.from_json("agent_config.json")
    print(loaded_config.model_dump())

if __name__ == "__main__":
    main()
