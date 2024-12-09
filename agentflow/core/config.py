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
    provider: str
    name: str
    temperature: float = 0.5
    max_tokens: Optional[int] = None
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate model provider"""
        valid_providers = ['openai', 'anthropic', 'google', 'default']
        if v not in valid_providers:
            raise ValueError(f"Unsupported model provider: {v}")
        return v
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature value"""
        if v < 0 or v > 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

class WorkflowStep(BaseModel):
    """工作流步骤配置"""
    type: str
    config: Optional[Dict[str, Any]] = None

class ExecutionPolicies(BaseModel):
    """Execution policies configuration"""
    required_fields: List[str] = Field(default_factory=list)
    default_status: str = "initialized"
    error_handling: Dict[str, str] = Field(default_factory=lambda: {
        "missing_field_error": "Missing required fields: {}",
        "missing_input_error": "Empty input data"
    })

class WorkflowConfig(BaseModel):
    """
    工作流配置类，定义工作流执行的关键参数
    """
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    
    # 工作流执行参数
    max_iterations: int = 10  # 默认最大迭代次数为10
    timeout: Optional[int] = None
    logging_level: str = 'INFO'  # 默认日志级别为INFO
    
    # 代理和连接配置
    agents: Optional[List['AgentConfig']] = None
    connections: Optional[List[Dict[str, Any]]] = None
    
    # 处理器配置
    processors: Optional[List['ProcessorConfig']] = None
    
    # 执行策略
    execution_policies: ExecutionPolicies = Field(default_factory=ExecutionPolicies)
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = None
    
    # 步骤配置
    steps: Optional[List[WorkflowStep]] = None  # 工作流步骤，可选
    
    # 协作配置
    collaboration: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        """
        初始化方法，确保默认值被正确设置
        """
        # 如果没有提供steps，设置为空列表
        if 'steps' not in data or data['steps'] is None:
            data['steps'] = []
        
        # 如果没有提供execution_policies，设置为空字典
        if 'execution_policies' not in data or data['execution_policies'] is None:
            data['execution_policies'] = {}
        
        super().__init__(**data)
    
    class Config:
        """
        Pydantic配置类，启用额外属性和不可变性
        """
        extra = 'allow'
        allow_population_by_field_name = True
        frozen = False
    
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
        
        # 处理步骤配置
        if 'steps' in workflow_dict:
            workflow_dict['steps'] = [
                WorkflowStep(**step) if isinstance(step, dict) else step
                for step in workflow_dict['steps']
            ]
        
        # 处理默认值
        workflow_dict.setdefault('id', str(uuid.uuid4()))
        workflow_dict.setdefault('max_iterations', 10)
        workflow_dict.setdefault('logging_level', 'INFO')
        workflow_dict.setdefault('steps', [])
        
        return cls(**workflow_dict)

class AgentConfig(BaseModel):
    """Agent配置类，定义Agent的关键参数和行为"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = 'default'
    agent_type: str = 'default'
    model: Optional[ModelConfig] = None
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    input_specification: Optional[InputSpec] = None
    output_specification: Optional[OutputSpec] = None
    config: Optional[Dict[str, Any]] = None
    execution_policies: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            'max_iterations': 10,
            'logging_level': 'INFO',
            'distributed': False
        }
    )
    workflow: Optional[WorkflowConfig] = None
    
    model_config = ConfigDict(
        extra='allow',
        populate_by_name=True,
        validate_assignment=True
    )
    
    @field_validator('type', 'agent_type')
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        """Validate agent type"""
        valid_types = {'default', 'research', 'test'}
        if v not in valid_types:
            raise ValueError(f"Unsupported agent type: {v}")
        return v
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v: Optional[ModelConfig]) -> Optional[ModelConfig]:
        """Validate model configuration"""
        if v and v.provider not in ['openai', 'anthropic', 'google', 'default']:
            raise ValueError(f"Unsupported model provider: {v.provider}")
        return v
    
    def __init__(self, **data):
        """
        初始化方法，确保默认值和工作流配置被正确设置
        """
        # 处理类型映射
        if 'type' not in data and 'agent_type' in data:
            data['type'] = data['agent_type']
        elif 'agent_type' not in data and 'type' in data:
            data['agent_type'] = data['type']
        
        # 处理默认值
        data.setdefault('id', str(uuid.uuid4()))
        data.setdefault('type', 'default')
        data.setdefault('agent_type', 'default')
        
        # 处理执行策略
        execution_policies = data.get('execution_policies', {})
        if execution_policies is None:
            execution_policies = {}
        
        # 合并默认值和提供的执行策略
        default_policies = {
            'max_iterations': 10,
            'logging_level': 'INFO',
            'distributed': False
        }
        execution_policies = {**default_policies, **execution_policies}
        data['execution_policies'] = execution_policies
        
        # 处理工作流配置
        if 'workflow' in data:
            workflow_data = data['workflow']
            if isinstance(workflow_data, dict):
                # 合并执行策略到工作流配置
                workflow_data.setdefault('max_iterations', execution_policies['max_iterations'])
                workflow_data.setdefault('logging_level', execution_policies['logging_level'])
                workflow_data.setdefault('distributed', execution_policies['distributed'])
                data['workflow'] = WorkflowConfig(**workflow_data)
        else:
            # 创建默认工作流配置
            data['workflow'] = WorkflowConfig(
                max_iterations=execution_policies['max_iterations'],
                logging_level=execution_policies['logging_level'],
                distributed=execution_policies['distributed']
            )
        
        # 处理输入输出规范
        if 'input_specification' in data and not isinstance(data['input_specification'], InputSpec):
            data['input_specification'] = InputSpec(**data['input_specification'])
        
        if 'output_specification' in data and not isinstance(data['output_specification'], OutputSpec):
            data['output_specification'] = OutputSpec(**data['output_specification'])
        
        super().__init__(**data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """
        从字典创建AgentConfig，支持多种配置格式
        
        :param config_dict: 配置字典
        :return: AgentConfig实例
        """
        return cls(**config_dict)
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        序列化配置，包括工作流详情
        """
        data = super().model_dump(**kwargs)
        if self.workflow:
            data['workflow'] = self.workflow.model_dump()
        return data

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
