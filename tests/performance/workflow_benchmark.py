import time
import random
import statistics
import json
import logging
from typing import Dict, Any
import pytest
from agentflow.core.workflow import WorkflowEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockPerformanceAgent:
    def __init__(self, name, complexity=1):
        self.name = name
        self.complexity = complexity
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 模拟不同复杂度的任务处理
        time.sleep(0.01 * self.complexity)
        context[f'{self.name}_processed'] = True
        return context

def generate_workflow_config(mode, agent_count, complexity_range=(1, 5)):
    """
    生成不同复杂度的工作流配置
    
    :param mode: 工作流模式
    :param agent_count: Agent数量
    :param complexity_range: Agent复杂度范围
    :return: 工作流配置
    """
    if mode == 'SEQUENTIAL':
        return {
            "COLLABORATION": {
                "MODE": mode,
                "WORKFLOW": [
                    {"name": f"agent_{i}", "complexity": random.uniform(*complexity_range)} 
                    for i in range(agent_count)
                ]
            }
        }
    elif mode == 'PARALLEL':
        return {
            "COLLABORATION": {
                "MODE": mode,
                "WORKFLOW": [
                    {"name": f"agent_{i}", "complexity": random.uniform(*complexity_range)} 
                    for i in range(agent_count)
                ]
            }
        }
    elif mode == 'DYNAMIC_ROUTING':
        return {
            "COLLABORATION": {
                "MODE": mode,
                "WORKFLOW": {
                    f"agent_{i}": {
                        "dependencies": [] if i == 0 else [f"agent_{i-1}_processed"],
                        "complexity": random.uniform(*complexity_range)
                    } 
                    for i in range(agent_count)
                }
            }
        }

def benchmark_workflow(mode, agent_count, iterations=10):
    """
    性能基准测试
    
    :param mode: 工作流模式
    :param agent_count: Agent数量
    :param iterations: 测试迭代次数
    :return: 性能测试结果
    """
    workflow_config = generate_workflow_config(mode, agent_count)
    
    # 性能测试前的准备
    def create_agent(config):
        return MockPerformanceAgent(
            config.get('name', 'unnamed'), 
            config.get('complexity', 1)
        )
    
    # 临时替换Agent创建方法
    original_create_agent = WorkflowEngine._create_agent
    WorkflowEngine._create_agent = lambda self, config: create_agent(config)
    
    try:
        execution_times = []
        memory_usages = []
        
        for _ in range(iterations):
            initial_context = {"test": "performance"}
            
            start_time = time.time()
            workflow = WorkflowEngine(workflow_config)
            result = workflow.execute(initial_context)
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
        
        return {
            "mode": mode,
            "agent_count": agent_count,
            "avg_execution_time": statistics.mean(execution_times),
            "std_execution_time": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times)
        }
    finally:
        # 恢复原始Agent创建方法
        WorkflowEngine._create_agent = original_create_agent

def test_workflow_performance():
    """
    性能测试主函数
    """
    modes = ['SEQUENTIAL', 'PARALLEL', 'DYNAMIC_ROUTING']
    agent_counts = [5, 10, 20, 50]
    
    performance_results = []
    
    for mode in modes:
        for agent_count in agent_counts:
            result = benchmark_workflow(mode, agent_count)
            performance_results.append(result)
            logger.info(json.dumps(result, indent=2))
    
    # 保存性能测试结果
    with open('/Users/xingqiangchen/TASK/APOS/tests/performance/performance_results.json', 'w') as f:
        json.dump(performance_results, f, indent=2)
    
    # 生成性能报告
    generate_performance_report(performance_results)

def generate_performance_report(results):
    """
    生成性能测试报告
    
    :param results: 性能测试结果
    """
    report = "# AgentFlow 工作流性能测试报告\n\n"
    
    for result in results:
        report += f"## {result['mode']} 模式 (Agent数量: {result['agent_count']})\n"
        report += f"- 平均执行时间: {result['avg_execution_time']:.4f}秒\n"
        report += f"- 执行时间标准差: {result['std_execution_time']:.4f}秒\n"
        report += f"- 最小执行时间: {result['min_execution_time']:.4f}秒\n"
        report += f"- 最大执行时间: {result['max_execution_time']:.4f}秒\n\n"
    
    with open('/Users/xingqiangchen/TASK/APOS/tests/performance/performance_report.md', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    test_workflow_performance()
