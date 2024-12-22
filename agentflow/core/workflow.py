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

class WorkflowExecutionError(Exception):
    """Error raised during workflow execution"""
    pass

class WorkflowStatus:
    """Workflow execution status constants"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class StepStatus:
    """Individual step execution status constants"""
    PENDING = "PENDING" 
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"

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
    async def _execute_parallel_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow in parallel.
        
        Args:
            context: Context data
            
        Returns:
            Execution results
        """
        tasks = []
        for agent_config in self.workflow:
            tasks.append(self.execute_step(agent_config, context.copy()))
        
        try:
            results = await asyncio.gather(*tasks)
            # Combine results into a dictionary
            final_result = {}
            for i, result in enumerate(results):
                agent_name = self.workflow[i].get('name', f'agent_{i}')
                final_result[agent_name] = result
            return final_result
        except Exception as e:
            logger.error(f"Parallel workflow execution error: {e}")
            raise WorkflowEngineError(f"Parallel execution failed: {e}") from e

    @memory_profiler
    async def _execute_sequential_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow sequentially.
        
        Args:
            context: Context data
            
        Returns:
            Execution results
        """
        results = {}
        for agent_config in self.workflow:
            try:
                result = await self.execute_step(agent_config, context.copy())
                agent_name = agent_config.get('name', 'unnamed_agent')
                results[agent_name] = result
                context.update(result)
            except Exception as e:
                logger.error(f"Sequential workflow execution error: {e}")
                raise WorkflowEngineError(f"Agent execution failed: {agent_config}") from e
        return results

    @memory_profiler
    async def _execute_dynamic_routing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with dynamic routing.
        
        Args:
            context: Context data
            
        Returns:
            Execution results
        """
        results = {}
        workflow_dict = self.workflow if isinstance(self.workflow, dict) else {}
        
        # Track completed agents and their processed outputs
        completed_agents = set()
        processed_outputs = set()
        
        while len(completed_agents) < len(workflow_dict):
            progress_made = False
            
            for agent_name, agent_config in workflow_dict.items():
                if agent_name in completed_agents:
                    continue
                    
                # Check dependencies
                dependencies = agent_config.get('dependencies', [])
                if not dependencies or all(d in processed_outputs for d in dependencies):
                    try:
                        result = await self.execute_step(agent_config, context.copy())
                        results[agent_name] = result
                        completed_agents.add(agent_name)
                        processed_outputs.add(f"{agent_name}_processed")
                        context.update(result)
                        progress_made = True
                    except Exception as e:
                        logger.error(f"Dynamic routing execution error: {e}")
                        raise WorkflowEngineError(f"Agent execution failed: {agent_config}") from e
            
            # If no progress was made in this iteration, we have a deadlock
            if not progress_made:
                raise WorkflowEngineError("Deadlock detected in workflow")
                
        return results

    def _apply_communication_protocol(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply communication protocol to process results.
        
        Args:
            results: Execution results
            
        Returns:
            Processed results according to protocol
        """
        if not results:
            return {}
            
        protocol_type = self.communication_protocol.get('TYPE')
        
        if protocol_type == 'FEDERATED':
            # Average model parameters for federated learning
            global_model = {}
            num_models = 0
            
            for result in results.values():
                if isinstance(result, dict) and 'model_params' in result:
                    num_models += 1
                    for param, value in result['model_params'].items():
                        if param not in global_model:
                            global_model[param] = 0
                        global_model[param] += value
            
            # Calculate averages
            if num_models > 0:
                for param in global_model:
                    global_model[param] /= num_models
                    
            return {'global_model': global_model}
            
        elif protocol_type == 'GOSSIP':
            # Share knowledge between nodes
            shared_knowledge = {}
            
            for result in results.values():
                if isinstance(result, dict) and 'knowledge' in result:
                    for topic, info in result['knowledge'].items():
                        key = f"shared_{topic}"
                        if key not in shared_knowledge:
                            shared_knowledge[key] = []
                        shared_knowledge[key].append(info)
                        
            # If no knowledge was shared, return original results
            if not shared_knowledge:
                return results
                
            return shared_knowledge
            
        elif protocol_type == 'HIERARCHICAL':
            # Merge data hierarchically
            hierarchy_levels = {}
            
            for result in results.values():
                if isinstance(result, dict) and 'hierarchy_level' in result:
                    level = result['hierarchy_level']
                    level_key = f'level_{level}'
                    if level_key not in hierarchy_levels:
                        hierarchy_levels[level_key] = []
                    hierarchy_levels[level_key].append(result.get('data', {}))
                    
            # If no hierarchical data was found, return original results
            if not hierarchy_levels:
                return results
                
            return hierarchy_levels
            
        else:
            # For unknown protocols, return results as is
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
        """Execute a workflow step.
        
        Args:
            agent_config: Agent configuration
            context: Current context
            
        Returns:
            Execution results
        """
        agent = self._create_agent(agent_config)
        
        # Add model_params and knowledge to context if present
        if 'model_params' in agent_config:
            context['model_params'] = agent_config['model_params']
        if 'knowledge' in agent_config:
            context['knowledge'] = agent_config['knowledge']
        if 'data' in agent_config:
            context['data'] = agent_config['data']
        if 'hierarchy_level' in agent_config:
            context['hierarchy_level'] = agent_config['hierarchy_level']
            
        # Execute agent
        try:
            result = await agent.execute(context)
            return result
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            raise WorkflowEngineError(f"Agent execution failed: {agent_config}") from e

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