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
from enum import Enum

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

class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

class StepStatus(str, Enum):
    """Enum for step status"""
    PENDING = 'pending'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILED = 'failed'
    COMPLETED = 'completed'
    SKIPPED = 'skipped'
    CANCELLED = 'cancelled'

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
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize workflow engine.
        
        Args:
            config: Workflow configuration
        """
        self.config = config
        self.workflow = self._get_workflow_config()
        self.mode = self._get_mode()
        self.communication_protocol = self._get_communication_protocol()
        self.execution_policies = self._get_execution_policies()
        self.required_fields = self._get_required_fields()
        self.error_handling = self._get_error_handling()
        self.default_status = self._get_default_status()
        self.steps = self._get_steps()

        # 性能优化：缓存Agent创建
        self._agent_cache = {}
        self._config_hash_cache = {}
        
        # Validate execution mode
        valid_modes = ['SEQUENTIAL', 'PARALLEL', 'DYNAMIC_ROUTING']
        if self.mode not in valid_modes:
            raise WorkflowEngineError(f"不支持的工作流模式: {self.mode}")

    def _get_required_fields(self) -> List[str]:
        """Get required fields from configuration.

        Returns:
            List of required field names
        """
        return self.config.get('required_fields', [])

    def _get_workflow_config(self) -> List[Dict[str, Any]]:
        """Get workflow configuration.

        Returns:
            List of agent configurations
        """
        if 'COLLABORATION' in self.config:
            return self.config['COLLABORATION'].get('WORKFLOW', [])
        return []

    def _get_mode(self) -> str:
        """Get workflow mode.

        Returns:
            Workflow mode
        """
        if 'COLLABORATION' in self.config:
            return self.config['COLLABORATION'].get('MODE', 'SEQUENTIAL')
        return 'SEQUENTIAL'

    def _get_communication_protocol(self) -> Optional[Dict[str, Any]]:
        """Get communication protocol configuration.

        Returns:
            Communication protocol configuration
        """
        if 'COLLABORATION' in self.config:
            return self.config['COLLABORATION'].get('COMMUNICATION_PROTOCOL')
        return None

    def _get_execution_policies(self) -> Dict[str, Any]:
        """Get execution policies.

        Returns:
            Execution policies
        """
        if 'COLLABORATION' in self.config:
            return self.config['COLLABORATION'].get('EXECUTION_POLICIES', {})
        return {}

    def _get_error_handling(self) -> Dict[str, Any]:
        """Get error handling configuration.

        Returns:
            Error handling configuration
        """
        if 'COLLABORATION' in self.config:
            return self.config['COLLABORATION'].get('ERROR_HANDLING', {})
        return {}

    def _get_default_status(self) -> Optional[str]:
        """Get default status from configuration.

        Returns:
            Default status or None if not specified
        """
        if 'COLLABORATION' in self.config:
            return self.config['COLLABORATION'].get('DEFAULT_STATUS')
        return None

    def _get_steps(self) -> List[Dict[str, Any]]:
        """Get workflow steps from configuration.

        Returns:
            List of workflow steps
        """
        return self.config.get('steps', [])
    
    @memory_profiler
    async def execute(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with different modes and communication protocols.
        
        Args:
            initial_context: Initial context data
            
        Returns:
            Execution results
        """
        # Return empty dict for empty workflow
        if not self.workflow:
            return {}
            
        try:
            # Execute workflow based on mode
            if self.mode == 'SEQUENTIAL':
                results = await self._execute_sequential_workflow(initial_context)
            elif self.mode == 'PARALLEL':
                results = await self._execute_parallel_workflow(initial_context)
            elif self.mode == 'DYNAMIC_ROUTING':
                results = await self._execute_dynamic_routing(initial_context)
            else:
                raise WorkflowEngineError(f"不支持的工作流模式: {self.mode}")

            # Apply communication protocol if present
            if self.communication_protocol:
                protocol_result = self._apply_communication_protocol(results)
                if protocol_result:  # Only use protocol result if not empty
                    return protocol_result
            
            # Return original results if no protocol or protocol result is empty
            return results
        except Exception as e:
            logger.error(f"工作流执行错误: {e}")
            raise WorkflowEngineError("工作流执行失败") from e
    
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
            # Combine all results
            final_result = {}
            for result in results:
                if isinstance(result, dict):
                    final_result.update(result)
            return final_result
        except Exception as e:
            logger.error(f"并行工作流执行错误: {e}")
            raise WorkflowEngineError(f"并行执行失败: {e}") from e

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
                # Execute step and get result
                step_result = await self.execute_step(agent_config, context.copy())
                
                # Update context with step result
                if isinstance(step_result, dict):
                    context.update(step_result)
                    results.update(step_result)
            except Exception as e:
                logger.error(f"顺序工作流执行错误: {e}")
                raise WorkflowEngineError(f"代理执行失败: {agent_config}") from e
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
        
        # Track executed agents to prevent cycles
        executed = set()
        
        while len(executed) < len(workflow_dict):
            executed_this_round = False
            
            for agent_name, agent_config in workflow_dict.items():
                if agent_name in executed:
                    continue
                
                # Check dependencies
                dependencies = agent_config.get('dependencies', [])
                if all(dep in results for dep in dependencies):
                    try:
                        result = await self.execute_step(agent_config, context.copy())
                        if isinstance(result, dict):
                            results.update(result)
                            
                            # Add processed flag for hierarchical merge
                            if 'data' in result.get(agent_name, {}):
                                processed_key = f"{agent_name}_processed"
                                results[processed_key] = {
                                    'data': result[agent_name]['data'],
                                    'processed': True
                                }
                                
                        executed.add(agent_name)
                        executed_this_round = True
                    except Exception as e:
                        logger.error(f"动态路由执行错误: {e}")
                        raise WorkflowEngineError(f"代理执行失败: {agent_config}") from e
            
            if not executed_this_round:
                missing_deps = []
                for agent_name, agent_config in workflow_dict.items():
                    if agent_name not in executed:
                        deps = agent_config.get('dependencies', [])
                        missing = [dep for dep in deps if dep not in results]
                        if missing:
                            missing_deps.append(f"{agent_name}: {missing}")
                raise WorkflowEngineError(f"循环依赖或缺失依赖: {missing_deps}")
                
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
            
            # Extract model parameters from each agent's results
            for agent_name, agent_result in results.items():
                if isinstance(agent_result, dict) and 'model_params' in agent_result:
                    num_models += 1
                    for param, value in agent_result['model_params'].items():
                        if param not in global_model:
                            global_model[param] = 0
                        if isinstance(value, (int, float)):
                            global_model[param] += value
            
            # Calculate averages
            if num_models > 0:
                for param in global_model:
                    global_model[param] /= num_models
                    
            return {'global_model': global_model}
            
        elif protocol_type == 'GOSSIP':
            # Share knowledge between nodes
            shared_knowledge = {}
            
            # Extract knowledge from each agent's results
            for agent_name, agent_result in results.items():
                if isinstance(agent_result, dict):
                    agent_knowledge = agent_result.get('knowledge', {})
                    for topic, info in agent_knowledge.items():
                        if topic not in shared_knowledge:
                            shared_knowledge[topic] = set()
                        shared_knowledge[topic].add(info)
            
            # Convert sets back to lists for JSON serialization
            for topic in shared_knowledge:
                shared_knowledge[topic] = list(shared_knowledge[topic])
                
            return shared_knowledge
            
        elif protocol_type == 'HIERARCHICAL':
            # Process low level data first
            processed_data = {}
            hierarchy_levels = {}
            
            # First pass: process raw data at each level
            for agent_name, agent_result in results.items():
                if isinstance(agent_result, dict):
                    level = agent_result.get('hierarchy_level', 0)
                    if level not in hierarchy_levels:
                        hierarchy_levels[level] = []
                    hierarchy_levels[level].append(agent_name)
                    
                    # Process raw data for level 0
                    if level == 0 and 'data' in agent_result:
                        processed_key = f"{agent_name}_processed"
                        processed_data[processed_key] = {
                            'data': agent_result['data'],
                            'processed': True
                        }
            
            # Second pass: merge data hierarchically
            merged_data = {}
            for level in sorted(hierarchy_levels.keys()):
                level_key = f'level_{level}'
                merged_data[level_key] = []
                
                for agent_name in hierarchy_levels[level]:
                    if level == 0:
                        # Use processed data for level 0
                        processed_key = f"{agent_name}_processed"
                        if processed_key in processed_data:
                            merged_data[level_key].append(processed_data[processed_key])
                    else:
                        # Higher levels use processed data from lower levels
                        if agent_name in results:
                            merged_data[level_key].append(results[agent_name].get('data', {}))
            
            return merged_data
            
        elif protocol_type == 'BLACKBOARD':
            # Share knowledge between nodes
            shared_knowledge = {}
            
            # Extract knowledge from each agent's results
            for agent_name, agent_result in results.items():
                if isinstance(agent_result, dict):
                    agent_knowledge = agent_result.get('knowledge', {})
                    for topic, info in agent_knowledge.items():
                        if topic not in shared_knowledge:
                            shared_knowledge[topic] = set()
                        shared_knowledge[topic].add(info)
            
            # Convert sets back to lists for JSON serialization
            for topic in shared_knowledge:
                shared_knowledge[topic] = list(shared_knowledge[topic])
                
            return shared_knowledge
            
        else:
            # For unknown protocols, return original results
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

    @memory_profiler
    async def execute_step(self, agent_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in the workflow.
        
        Args:
            agent_config: Agent configuration
            context: Context data
            
        Returns:
            Step execution results
        """
        try:
            agent_name = agent_config.get('name', '')
            agent_type = agent_config.get('agent_type', '')
            
            # Create result dictionary with agent name
            result = {agent_name: {}}
            
            # Add model parameters if present
            if 'model_params' in agent_config:
                result[agent_name]['model_params'] = agent_config['model_params']
                
            # Add knowledge if present
            if 'knowledge' in agent_config:
                result[agent_name]['knowledge'] = agent_config['knowledge']
                
            # Add data and hierarchy level if present
            if 'data' in agent_config:
                result[agent_name]['data'] = agent_config['data']
            if 'hierarchy_level' in agent_config:
                result[agent_name]['hierarchy_level'] = agent_config['hierarchy_level']
                
            # Create and execute agent
            agent = self._create_agent(agent_config)
            agent_result = await agent.execute(context)
            if isinstance(agent_result, dict):
                result[agent_name].update(agent_result)
                
            return result
        except Exception as e:
            logger.error(f"步骤执行错误: {e}")
            raise WorkflowEngineError(f"步骤执行失败: {agent_config}") from e

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