from typing import Dict, Any, List, Optional
import os
import ell
from agentflow.core.agent import Agent
from agentflow.core.config import AgentConfig, ModelConfig

class BaseTestAgent(Agent):
    """用于测试的基础Agent，支持ell-ai LLM调用"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        
        # 解析配置
        if isinstance(config, dict):
            self.agent_config = AgentConfig(**config)
        else:
            self.agent_config = config
            
        # 从配置中提取名称
        self.name = self.agent_config.name or 'unnamed_agent'
        
        # 从配置中获取模型配置
        self.model_config = self.agent_config.model
        
        # 初始化ell
        if not hasattr(self, '_ell_initialized'):
            ell.init(verbose=True)
            self._ell_initialized = True

    @property
    def _model_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            'model': self.model_config.name,
            'temperature': self.model_config.temperature,
            'max_tokens': self.model_config.max_tokens
        }

    @property
    def _system_prompt(self) -> str:
        """获取系统提示词"""
        return self.agent_config.execution_policies.get(
            'system_prompt',
            self.agent_config.description or ''
        )

    @ell.complex(model='gpt-3.5-turbo')
    def _generate_llm_response(self, messages: List[ell.Message]) -> List[ell.Message]:
        """
        使用ell-ai生成LLM响应
        
        :param messages: 消息历史
        :return: LLM响应消息列表
        """
        system_message = [ell.system(self._system_prompt)] if self._system_prompt else []
        return system_message + messages

    def generate_llm_response(
        self, 
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        生成LLM响应的便捷方法
        
        :param prompt: 输入提示词
        :param context: 可选的上下文信息
        :return: LLM生成的响应或None
        """
        try:
            # 构建消息历史
            messages = []
            if context and 'message_history' in context:
                messages.extend(context['message_history'])
            messages.append(ell.user(prompt))
            
            # 使用配置的模型参数调用LLM
            response = self._generate_llm_response(messages, **self._model_params)
            return response.text if response else None
            
        except Exception as e:
            print(f"Error generating LLM response for agent {self.name}: {e}")
            return None

    @ell.complex(model='gpt-3.5-turbo')
    def __call__(self, task: str) -> str:
        """
        执行Agent的主要任务处理逻辑
        
        :param task: 任务描述
        :return: 任务执行结果
        """
        # 对于测试场景，直接返回一个默认响应
        return f"Agent {self.name} processed task: {task}"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行逻辑，支持LLM处理
        
        :param context: 输入上下文
        :return: 更新后的上下文
        """
        # 对于不同的通信协议，返回不同的结果
        if 'model_params' in self.config:
            context['global_model'] = self.config['model_params']
        
        if 'knowledge' in self.config:
            context.update(self.config['knowledge'])
        
        if 'data' in self.config:
            context[f'{self.name}_data'] = self.config['data']
        
        # LLM处理
        if 'llm_prompt' in context:
            llm_response = self.generate_llm_response(
                context['llm_prompt'],
                context
            )
            if llm_response:
                context[f'{self.name}_llm_response'] = llm_response
                # 保存消息历史
                if 'message_history' not in context:
                    context['message_history'] = []
                context['message_history'].append(ell.user(context['llm_prompt']))
                context['message_history'].append(ell.assistant(llm_response))
        
        context[f'{self.name}_processed'] = True
        return context
