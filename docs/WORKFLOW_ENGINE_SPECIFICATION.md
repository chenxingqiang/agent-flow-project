# AgentFlow 工作流引擎规范

## 1. 概述

AgentFlow 工作流引擎是一个高度灵活、可配置的 AI Agent 协作框架，支持多种复杂的工作流执行模式和通信协议。

## 2. 工作流模式

### 2.1 顺序执行 (Sequential)

顺序执行模式按照预定义的顺序依次执行 Agent，每个 Agent 的输出作为下一个 Agent 的输入。

#### 配置示例

```json
{
    "COLLABORATION": {
        "MODE": "SEQUENTIAL",
        "WORKFLOW": [
            {"name": "research_agent"},
            {"name": "writing_agent"},
            {"name": "review_agent"}
        ]
    }
}
```

### 2.2 并行执行 (Parallel)

并行执行模式同时启动多个 Agent，提高整体执行效率。

#### 2.2.1 配置示例

```json
{
    "COLLABORATION": {
        "MODE": "PARALLEL",
        "WORKFLOW": [
            {"name": "data_collection_agent"},
            {"name": "analysis_agent"},
            {"name": "visualization_agent"}
        ]
    }
}
```

### 2.3 动态路由 (Dynamic Routing)

动态路由模式根据上下文和依赖关系动态决定 Agent 执行顺序。

#### 2.3.1 配置示例

```json
{
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
        }
    }
}
```

## 3. 通信协议

### 3.1 语义消息 (Semantic Message)

简单的消息合并，直接更新上下文。

### 3.2 RPC 风格 (RPC)

支持复杂的远程过程调用风格消息合并。

### 3.3 事件驱动 (Event-Driven)

收集和管理事件列表。

### 3.4 共识算法 (Consensus)

通过投票或统计方法选择最佳结果。

### 3.5 黑板模式 (Blackboard)

动态更新和选择最优解决方案。

## 4. 性能优化

- 内存性能分析装饰器
- Agent 实例缓存
- 动态线程池管理
- 上下文数据及时清理

## 5. 错误处理

- 自定义异常 `WorkflowEngineError`
- 详细日志记录
- 异常追踪和上下文保留

## 6. 最佳实践

1. 保持 Agent 配置简洁和模块化
2. 合理设计依赖关系
3. 选择适合场景的工作流模式
4. 注意内存和性能开销

## 7. 扩展性

框架支持通过继承和插件机制进行功能扩展。

## 8. 性能基准测试

请参考 `tests/performance/performance_report.md` 获取最新的性能测试报告。

## 9. 许可和贡献

AgentFlow 遵循开源协议，欢迎社区贡献和反馈。
