# AgentFlow DSL 规范 v2.0.0

## 1. 概述

本文档定义了 AgentFlow 系统的领域特定语言（DSL）规范，用于描述智能代理（Agent）的配置、行为和交互方式。

## 2. 基本结构

Agent 配置由以下主要部分组成：

- `AGENT`：基本元数据
- `INPUT_SPECIFICATION`：输入规范
- `OUTPUT_SPECIFICATION`：输出规范
- `DATA_FLOW_CONTROL`：数据流控制
- `INTERFACE_CONTRACTS`：接口契约

## 3. 详细规范

### 3.1 AGENT 元数据

```json
{
  "AGENT": {
    "NAME": "string",
    "VERSION": "semver",
    "TYPE": "string"
  }
}
```

#### 字段说明

- `NAME`：代理名称
- `VERSION`：语义化版本号
- `TYPE`：代理类型（如 research, support, analysis）

### 3.2 INPUT_SPECIFICATION

```json
{
  "INPUT_SPECIFICATION": {
    "MODES": [
      "DIRECT_INPUT",
      "CONTEXT_INJECTION",
      "STREAM_INPUT",
      "REFERENCE_INPUT"
    ],
    "TYPES": {
      "DIRECT": {},
      "CONTEXT": {
        "sources": [
          "PREVIOUS_AGENT_OUTPUT",
          "GLOBAL_MEMORY",
          "EXTERNAL_CONTEXT"
        ]
      },
      "STREAM": {
        "modes": ["REAL_TIME", "BATCH", "INCREMENTAL"]
      },
      "REFERENCE": {
        "types": ["FILE_PATH", "DATABASE_QUERY", "MEMORY_POINTER"]
      }
    },
    "VALIDATION": {
      "STRICT_MODE": "boolean",
      "SCHEMA_VALIDATION": "boolean",
      "TRANSFORM_STRATEGIES": ["TYPE_COERCION", "DEFAULT_VALUE", "NULLABLE"]
    }
  }
}
```

#### 关键特性

- 支持多种输入模式
- 灵活的输入类型
- 可配置的验证策略

### 3.3 OUTPUT_SPECIFICATION

```json
{
  "OUTPUT_SPECIFICATION": {
    "MODES": ["RETURN", "FORWARD", "STORE", "TRIGGER"],
    "STRATEGIES": {
      "RETURN": {
        "options": ["FULL_RESULT", "PARTIAL_RESULT", "SUMMARY"]
      },
      "FORWARD": {
        "routing_options": ["DIRECT_PASS", "TRANSFORM", "SELECTIVE_FORWARD"]
      },
      "STORE": {
        "storage_types": [
          "GLOBAL_MEMORY",
          "TEMPORARY_CACHE",
          "PERSISTENT_STORAGE"
        ]
      },
      "TRIGGER": {
        "trigger_types": [
          "WORKFLOW_CONTINUATION",
          "PARALLEL_EXECUTION",
          "CONDITIONAL_BRANCH"
        ]
      }
    },
    "TRANSFORMATION": {
      "ENABLED": "boolean",
      "METHODS": ["FILTER", "MAP", "REDUCE", "AGGREGATE"]
    }
  }
}
```

#### 关键特性

- 多样的输出模式
- 结果转换能力
- 灵活的路由策略

### 3.4 DATA_FLOW_CONTROL

```json
{
  "DATA_FLOW_CONTROL": {
    "ROUTING_RULES": {
      "DEFAULT_BEHAVIOR": "FORWARD_ALL",
      "CONDITIONAL_ROUTING": {
        "CONDITIONS": [
          {
            "WHEN": "条件表达式",
            "ACTION": "处理动作"
          }
        ]
      }
    },
    "ERROR_HANDLING": {
      "STRATEGIES": ["SKIP", "RETRY", "FALLBACK", "COMPENSATE"],
      "MAX_RETRIES": "integer"
    }
  }
}
```

#### 关键特性

- 条件路由
- 多策略错误处理
- 动态流程控制

### 3.5 INTERFACE_CONTRACTS

```json
{
  "INTERFACE_CONTRACTS": {
    "INPUT_CONTRACT": {
      "REQUIRED_FIELDS": ["string"],
      "OPTIONAL_FIELDS": ["string"]
    },
    "OUTPUT_CONTRACT": {
      "MANDATORY_FIELDS": ["string"],
      "OPTIONAL_FIELDS": ["string"]
    }
  }
}
```

#### 关键特性

- 定义输入输出接口规范
- 强制和可选字段
- 接口一致性保证

## 4. 使用示例

### 4.1 基本配置示例

```json
{
  "AGENT": {
    "NAME": "Research Assistant",
    "VERSION": "1.0.0",
    "TYPE": "research"
  },
  "INPUT_SPECIFICATION": {
    "MODES": ["CONTEXT_INJECTION"],
    "TYPES": {
      "CONTEXT": {
        "sources": ["PREVIOUS_AGENT_OUTPUT"]
      }
    }
  },
  "OUTPUT_SPECIFICATION": {
    "MODES": ["FORWARD"],
    "STRATEGIES": {
      "FORWARD": {
        "routing_options": ["TRANSFORM"]
      }
    }
  }
}
```

## 5. 版本历史

- v1.0.0: 初始版本
- v2.0.0: 增强输入输出灵活性，支持更复杂的工作流

## 6. 附录

### 6.1 术语表

- **Agent**：执行特定任务的智能代理
- **DSL**：领域特定语言
- **输入规范**：定义代理接受输入的方式和约束
- **输出规范**：定义代理产生输出的方式和策略

### 6.2 最佳实践

1. 保持配置的简洁和清晰
2. 合理使用条件路由
3. 充分利用转换和验证机制
4. 考虑错误处理策略

## 7. 许可

本规范采用 MIT 开源许可协议。
