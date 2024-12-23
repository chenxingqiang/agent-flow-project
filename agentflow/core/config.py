"""Configuration classes for AgentFlow."""
from typing import Dict, Any, List, Optional, FrozenSet, Union
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
        valid_providers = ['openai', 'anthropic', 'google', 'default', 'ray']
        if v.lower() not in valid_providers:
            raise ValueError(f"Unsupported model provider: {v}. Must be one of {valid_providers}")
        return v.lower()
    
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
    default_status: Optional[str] = "initialized"
    error_handling: Dict[str, str] = Field(default_factory=lambda: {
        "missing_field_error": "Missing required fields: {}",
        "missing_input_error": "Empty input data"
    })
    steps: List[Dict[str, Any]] = Field(default_factory=list)

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
    distributed: bool = False  # 是否为分布式工作流
    
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
    
    class Config:
        """
        Pydantic配置类，启用额外属性和不可变性
        """
        extra = 'allow'
        populate_by_name = True
        frozen = False
    
    def __init__(self, **data):
        """
        初始化方法，确保默认值被正确设置
        """
        # 如果没有提供steps，设置为空列表
        if 'steps' not in data or data['steps'] is None:
            data['steps'] = []
        
        # 如果没有提供execution_policies，设置为空字典
        if 'execution_policies' not in data or data['execution_policies'] is None:
            data['execution_policies'] = ExecutionPolicies(
                required_fields=[],
                default_status=None,
                error_handling={},
                steps=[]
            )
        elif not isinstance(data['execution_policies'], ExecutionPolicies):
            data['execution_policies'] = ExecutionPolicies(**data['execution_policies'])
        
        super().__init__(**data)
    
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
    model: Optional[Union[ModelConfig, Dict[str, Any]]] = None
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
    workflow: Optional[Union[WorkflowConfig, Dict[str, Any]]] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')

    def __init__(self, **data):
        """Initialize method to ensure default values and workflow configuration are set correctly"""
        # Handle type mapping
        if 'type' not in data and 'agent_type' in data:
            data['type'] = data['agent_type']
        elif 'agent_type' not in data and 'type' in data:
            data['agent_type'] = data['type']

        # Handle default values
        data.setdefault('id', str(uuid.uuid4()))
        data.setdefault('type', 'default')
        data.setdefault('agent_type', 'default')

        # Handle model configuration
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])

        # Handle workflow configuration
        if 'workflow' in data and isinstance(data['workflow'], dict):
            data['workflow'] = WorkflowConfig(**data['workflow'])

        # Handle execution policies
        execution_policies = data.get('execution_policies', {})
        if execution_policies is None:
            execution_policies = {}

        # Merge default values and provided execution policies
        default_policies = {
            'max_iterations': 10,
            'logging_level': 'INFO',
            'distributed': False
        }
        execution_policies = {**default_policies, **execution_policies}
        data['execution_policies'] = execution_policies

        # Handle input specification
        if 'input_specification' in data:
            if data['input_specification'] is None or (isinstance(data['input_specification'], dict) and not data['input_specification']):
                data['input_specification'] = InputSpec()
            elif not isinstance(data['input_specification'], InputSpec):
                data['input_specification'] = InputSpec(**data['input_specification'])

        super().__init__(**data)

    @field_validator('agent_type')
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        """Validate agent type"""
        valid_types = ['default', 'research', 'data_science', 'generic']
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid agent type: {v}. Must be one of {valid_types}")
        return v.lower()

    @field_validator('model')
    @classmethod
    def validate_model(cls, v: Optional[Union[ModelConfig, Dict[str, Any]]]) -> Optional[ModelConfig]:
        """Validate model configuration"""
        if v is None:
            return None
        if isinstance(v, dict):
            return ModelConfig(**v)
        return v

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """从字典创建AgentConfig，支持多种配置格式"""
        return cls(**config_dict)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """序列化配置，包括工作流详情"""
        data = super().model_dump(**kwargs)
        if self.workflow:
            data['workflow'] = self.workflow.model_dump()
        if self.model:
            data['model'] = self.model.model_dump()
        return data

class ProcessorConfig(BaseModel):
    """Processor configuration"""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    type: str = "processor"
    processor: Any  # Can be string (import path) or class
    config: Dict[str, Any] = Field(default_factory=dict)

class StepConfig(BaseModel):
    """Configuration for workflow steps."""
    name: str = Field(description="Name of the step")
    description: Optional[str] = Field(default=None, description="Description of what the step does")
    input: List[str] = Field(default_factory=list, description="Required input fields")
    output: Dict[str, Any] = Field(default_factory=dict, description="Output configuration")
    depends_on: Optional[FrozenSet[int]] = Field(default_factory=frozenset, description="Set of step numbers this step depends on")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    @field_validator('depends_on', mode='before')
    @classmethod
    def validate_depends_on(cls, v):
        """Convert depends_on to frozenset if needed"""
        if isinstance(v, (list, set)):
            return frozenset(v)
        return v

class AgentMetadataConfig(BaseModel):
    """Agent元数据配置"""
    name: str
    version: str
    type: str
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

class InputSpecConfig(BaseModel):
    """输入规范定义"""
    types: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

class OutputSpecConfig(BaseModel):
    """输出规范定义"""
    types: Dict[str, Any] = Field(default_factory=dict)
    format: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

class DataFlowControlConfig(BaseModel):
    """数据流控制配置"""
    routing_rules: Dict[str, Any] = Field(default_factory=dict)
    error_handling: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

class InterfaceContractConfig(BaseModel):
    """接口契约配置"""
    input_contract: Dict[str, List[str]] = Field(default_factory=dict)
    output_contract: Dict[str, List[str]] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

class ModelConfigConfig(BaseModel):
    """Configuration for AI model"""
    provider: str
    name: str
    temperature: float = 0.5
    max_tokens: Optional[int] = None
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate model provider"""
        valid_providers = ['openai', 'anthropic', 'google', 'default', 'ray']
        if v.lower() not in valid_providers:
            raise ValueError(f"Unsupported model provider: {v}. Must be one of {valid_providers}")
        return v.lower()
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature value"""
        if v < 0 or v > 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

class WorkflowStepConfig(BaseModel):
    """工作流步骤配置"""
    type: str
    config: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

class ExecutionPoliciesConfig(BaseModel):
    """Execution policies configuration"""
    required_fields: List[str] = Field(default_factory=list)
    default_status: Optional[str] = "initialized"
    error_handling: Dict[str, str] = Field(default_factory=lambda: {
        "missing_field_error": "Missing required fields: {}",
        "missing_input_error": "Empty input data"
    })
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

class WorkflowConfigConfig(BaseModel):
    """
    工作流配置类，定义工作流执行的关键参数
    """
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    
    # 工作流执行参数
    max_iterations: int = 10  # 默认最大迭代次数为10
    timeout: Optional[int] = None
    logging_level: str = 'INFO'  # 默认日志级别为INFO
    distributed: bool = False  # 是否为分布式工作流
    
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
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    def __init__(self, **data):
        """
        初始化方法，确保默认值被正确设置
        """
        # 如果没有提供steps，设置为空列表
        if 'steps' not in data or data['steps'] is None:
            data['steps'] = []
        
        # 如果没有提供execution_policies，设置为空字典
        if 'execution_policies' not in data or data['execution_policies'] is None:
            data['execution_policies'] = ExecutionPolicies(
                required_fields=[],
                default_status=None,
                error_handling={},
                steps=[]
            )
        elif not isinstance(data['execution_policies'], ExecutionPolicies):
            data['execution_policies'] = ExecutionPolicies(**data['execution_policies'])
        
        super().__init__(**data)
    
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
        required_sections = {'AGENT', 'MODEL', 'WORKFLOW'}
        missing = required_sections - set(config.keys())
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
            
        # Validate model config
        model_config = config.get('MODEL', {})
        if not all(k in model_config for k in ['provider', 'name']):
            raise ValueError("Model config missing required fields")
            
        # Validate workflow config
        workflow_config = config.get('WORKFLOW', {})
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
                
                # Validate each variable
                for var_name, var_config in value.items():
                    if not isinstance(var_config, dict):
                        raise ValueError(f"Variable {var_name} must be a dictionary")
                    
                    # Check for required keys
                    if 'type' not in var_config:
                        raise ValueError(f"Variable {var_name} must have a 'type' key")
                    
                    # Validate type
                    valid_types = ['string', 'integer', 'float', 'boolean', 'list', 'dict']
                    if var_config['type'] not in valid_types:
                        raise ValueError(f"Invalid type for variable {var_name}. Must be one of {valid_types}")
        
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
