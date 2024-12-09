from typing import Dict, Any, List, Optional, Union
import json
import importlib
import ray
import logging
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import gc
import os
from .base_workflow import BaseWorkflow
import asyncio
from .node import Node, AgentNode, ProcessorNode

logger = logging.getLogger(__name__)

def memory_profiler(func):
    """内存性能分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 启动前内存
        gc.collect()
        start_memory = sys.getsizeof(args) + sys.getsizeof(kwargs)
        
        try:
            result = func(*args, **kwargs)
            
            # 结束后内存
            end_memory = sys.getsizeof(result)
            memory_used = end_memory - start_memory
            
            if memory_used > 1024 * 1024:  # 超过1MB记录日志
                logger.info(f"Function {func.__name__} memory usage: {memory_used / 1024 / 1024:.2f} MB")
            
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} execution error: {e}")
            raise
    return wrapper

class WorkflowEngineError(Exception):
    """WorkflowEngine的自定义异常"""
    pass

class WorkflowExecutor:
    """Executes workflows with configurable timeout"""
    
    def __init__(self, workflow_config: Dict[str, Any], timeout: float = 5.0):
        """
        Initialize workflow executor
        
        Args:
            workflow_config: Workflow configuration
            timeout: Maximum time (in seconds) to wait for workflow execution
        """
        self.workflow_config = workflow_config
        self.timeout = timeout
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow with timeout
        
        Args:
            input_data: Initial input data
            
        Returns:
            Workflow execution result
            
        Raises:
            asyncio.TimeoutError: If workflow execution exceeds timeout
        """
        try:
            # Use asyncio.wait_for to enforce timeout
            result = await asyncio.wait_for(
                self._execute_workflow(input_data), 
                timeout=self.timeout
            )
            return result
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Workflow execution timed out after {self.timeout} seconds")
        
    async def _execute_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal method to execute workflow steps
        
        Args:
            input_data: Initial input data
            
        Returns:
            Workflow execution result
        """
        current_data = input_data.copy()
        
        # Execute agents and processors in order
        for node_config in self.workflow_config.get('agents', []) + self.workflow_config.get('processors', []):
            node = self._create_node(node_config)
            current_data = await node.process(current_data)
        
        return current_data
    
    def _create_node(self, node_config: Dict[str, Any]) -> Node:
        """
        Create node based on configuration
        
        Args:
            node_config: Node configuration
            
        Returns:
            Configured node
        """
        node_type = node_config.get('type')
        
        if node_type == 'agent':
            return AgentNode(node_config)
        elif node_type == 'processor':
            return ProcessorNode(node_config)
        else:
            raise ValueError(f"Unsupported node configuration: {node_config}")

class WorkflowEngine(BaseWorkflow):
    """工作流引擎，支持复杂的Agent协作模式"""
    
    def __init__(self, workflow_config: Dict[str, Any]):
        """
        初始化工作流引擎
        
        :param workflow_config: 工作流配置
        """
        super().__init__(workflow_config)
        self.mode = workflow_config.get('COLLABORATION', {}).get('MODE', 'SEQUENTIAL')
        self.workflow = workflow_config.get('COLLABORATION', {}).get('WORKFLOW', {})
        self.communication_protocol = workflow_config.get('COLLABORATION', {}).get('COMMUNICATION_PROTOCOL', {})
        
        # 性能优化：缓存Agent创建
        self._agent_cache = {}
        self._config_hash_cache = {}
        
        # Validate execution mode
        valid_modes = ['SEQUENTIAL', 'PARALLEL', 'DYNAMIC_ROUTING']
        if self.mode not in valid_modes:
            raise WorkflowEngineError(f"不支持的工作流模式: {self.mode}")

    @memory_profiler
    async def execute(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工作流，支持不同模式和通信协议
        
        :param initial_context: 初始上下文数据
        :return: 执行结果
        """
        # 对于空工作流，直接返回空字典
        if not self.workflow:
            return {}

        try:
            if self.mode == 'SEQUENTIAL':
                results = await self._execute_sequential_workflow(initial_context)
            elif self.mode == 'PARALLEL':
                results = await self._execute_parallel_workflow(initial_context)
            elif self.mode == 'DYNAMIC_ROUTING':
                results = await self._execute_dynamic_routing(initial_context)
            else:
                raise WorkflowEngineError(f"不支持的工作流模式: {self.mode}")

            # 应用通信协议
            return self._apply_communication_protocol(results)
        except Exception as e:
            logger.error(f"工作流执行错误: {e}")
            raise WorkflowEngineError(f"工作流执行失败: {e}") from e
    
    @memory_profiler
    async def _execute_sequential_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        顺序执行工作流
        
        :param context: 上下文数据
        :return: 执行结果
        """
        for agent_config in self.workflow:
            try:
                context = await self.execute_step(agent_config, context)
                
                # 及时清理不再需要的上下文数据
                context = {k: v for k, v in context.items() if v is not None}
            except Exception as e:
                logger.error(f"顺序工作流执行错误: {e}")
                raise WorkflowEngineError(f"Agent执行失败: {agent_config}") from e
        
        return context
    
    @memory_profiler
    async def _execute_parallel_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        并行执行工作流
        
        :param context: 上下文数据
        :return: 执行结果
        """
        tasks = []
        for agent_config in self.workflow:
            tasks.append(self.execute_step(agent_config, context.copy()))
        
        try:
            results = await asyncio.gather(*tasks)
            final_result = {}
            for result in results:
                final_result.update(result)
            return final_result
        except Exception as e:
            logger.error(f"并行工作流执行错误: {e}")
            raise WorkflowEngineError(f"并行执行失败: {e}") from e
    
    @memory_profiler
    async def _execute_dynamic_routing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        动态路由执行工作流
        
        :param context: 上下文数据
        :return: 执行结果
        """
        for agent_id, agent_config in self.workflow.items():
            # 检查依赖和条件
            if not self._check_agent_dependencies(agent_config.get('dependencies', []), context):
                continue
            
            try:
                context = await self.execute_step(agent_config, context)
            except Exception as e:
                logger.error(f"动态路由执行错误: {e}")
                raise WorkflowEngineError(f"Agent {agent_id} 执行失败") from e
        
        return context
    
    def _check_agent_dependencies(self, dependencies: Union[Dict[str, List[str]], List[str]], context: Dict[str, Any]) -> bool:
        """
        检查Agent的依赖是否满足
        
        :param dependencies: 依赖列表或包含依赖的字典
        :param context: 当前上下文
        :return: 依赖是否满足
        """
        # 处理不同输入类型的依赖
        if isinstance(dependencies, dict):
            deps = dependencies.get('dependencies', [])
        elif isinstance(dependencies, list):
            deps = dependencies
        else:
            return False
        
        return all(dep in context and context[dep] for dep in deps)

    def _apply_communication_protocol(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用通信协议处理结果
        
        :param results: 执行结果
        :return: 处理后的结果
        """
        protocol_type = self.communication_protocol.get('TYPE')
        if protocol_type == 'SEMANTIC':
            return self._semantic_message_merge([results])
        elif protocol_type == 'RPC':
            return self._format_rpc_response(results)
        elif protocol_type == 'FEDERATED':
            # 联邦学习协议：对模型参数取平均
            global_model = {}
            model_params_count = 0
            
            # 遍历工作流配置，找到模型参数
            for workflow_item in self.workflow:
                if isinstance(workflow_item, dict) and 'model_params' in workflow_item:
                    model_params = workflow_item['model_params']
                    model_params_count += 1
                    for key, value in model_params.items():
                        global_model[key] = global_model.get(key, 0) + value
            
            # 对模型参数取平均
            if model_params_count > 0:
                for key in global_model:
                    global_model[key] /= model_params_count
            
            return {"global_model": global_model}
        elif protocol_type == 'HIERARCHICAL':
            # 层级合并协议
            result_levels = {}
            
            # 检查工作流配置中的层级信息
            workflow_dict = self.workflow if isinstance(self.workflow, dict) else {}
            
            # 遍历结果，根据层级标记结果
            for result_key, result_value in results.items():
                # 检查是否有处理完成的结果
                if result_key.endswith('_processed'):
                    # 尝试从工作流配置中获取层级信息
                    agent_name = result_key.replace('_processed', '')
                    agent_config = workflow_dict.get(agent_name, {})
                    
                    # 获取层级，默认为0
                    hierarchy_level = agent_config.get('hierarchy_level', 0)
                    
                    # 创建层级标记的结果
                    level_key = f"level_{hierarchy_level}"
                    result_levels[level_key] = result_value
            
            return result_levels
        else:
            return results

    def _semantic_message_merge(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并语义化消息
        
        :param messages: 消息列表
        :return: 合并后的消息
        """
        merged = {}
        for msg in messages:
            for key, value in msg.items():
                if key not in merged or key == 'shared':
                    merged[key] = value
        return merged

    def _format_rpc_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化RPC响应
        
        :param response: 原始响应
        :return: 格式化后的响应
        """
        # 简单实现，可以根据需要扩展
        return response
    
    def _create_agent(self, agent_config: Dict[str, Any]) -> Any:
        """
        创建Agent实例，并缓存
        
        :param agent_config: Agent配置
        :return: Agent实例
        """
        # 使用agent名称作为缓存键
        agent_name = agent_config.get('name')
        
        # 如果缓存中已存在，直接返回
        if agent_name and agent_name in self._agent_cache:
            return self._agent_cache[agent_name]
        
        # 创建配置的哈希值
        def config_hash(config):
            """创建配置的确定性哈希值"""
            # 将配置转换为JSON字符串，这样可以为字典创建唯一的哈希值
            return hash(json.dumps(config, sort_keys=True))
        
        config_hash_value = config_hash(agent_config)
        
        # 如果缓存中存在配置哈希，直接返回
        if config_hash_value in self._config_hash_cache:
            return self._config_hash_cache[config_hash_value]
        
        # 创建新的agent实例
        from agentflow.agents.base_agent import BaseTestAgent
        from agentflow.core.agent import Agent
        from agentflow.core.config import AgentConfig, ModelConfig
        
        try:
            # 对于测试场景，使用BaseTestAgent
            # 如果没有提供模型配置，使用默认配置
            if 'model' not in agent_config:
                agent_config['model'] = {
                    'provider': 'openai',
                    'name': 'gpt-3.5-turbo',
                    'temperature': 0.5
                }
            
            # 如果没有名称，生成一个随机名称
            if 'name' not in agent_config:
                agent_config['name'] = f'test_agent_{hash(config_hash_value)}'
            
            # 确保agent_type被设置
            agent_config['agent_type'] = agent_config.get('agent_type', 'base')
            
            # 创建Agent实例
            agent = BaseTestAgent(agent_config)
        except Exception as e:
            logger.error(f"Agent创建失败: {e}")
            raise WorkflowEngineError(f"无法创建Agent: {agent_config}") from e
        
        # 如果有名称，缓存agent
        if agent_name:
            self._agent_cache[agent_name] = agent
        
        # 缓存配置哈希对应的agent
        self._config_hash_cache[config_hash_value] = agent
        
        return agent

    async def execute_step(self, agent_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工作流的一个步骤
        
        :param agent_config: Agent配置
        :param context: 当前上下文
        :return: 执行结果
        """
        agent = self._create_agent(agent_config)
        return await agent.execute(context)

class Workflow:
    """Simple workflow class for executing agent and processor nodes"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize workflow with configuration
        
        Args:
            config: Dictionary containing workflow configuration
        """
        self.id = config.get("id")
        self.name = config.get("name")
        self.description = config.get("description")
        
        # Initialize nodes
        self.nodes = []
        
        # Add agent nodes
        for agent_config in config.get("agents", []):
            node = AgentNode(**agent_config)
            self.nodes.append(node)
            
        # Add processor nodes
        for processor_config in config.get("processors", []):
            node = ProcessorNode(**processor_config)
            self.nodes.append(node)
            
        # Store connections
        self.connections = config.get("connections", [])
        
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

def main():
    # 示例工作流配置
    workflow_config = {
        "COLLABORATION": {
            "MODE": "DYNAMIC_ROUTING",
            "WORKFLOW": {
                "research_agent": {
                    "dependencies": [],
                    "config_path": "/path/to/research_agent_config.json"
                },
                "writing_agent": {
                    "dependencies": ["research_agent_processed"],
                    "config_path": "/path/to/writing_agent_config.json"
                }
            },
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "BLACKBOARD"
            }
        }
    }
    
    # 初始上下文
    initial_context = {
        "research_topic": "AI Ethics",
        "deadline": "2024-12-31"
    }
    
    # 创建并执行工作流
    workflow = WorkflowEngine(workflow_config)
    result = asyncio.run(workflow.execute(initial_context))
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()