"""Metric type module."""

from enum import Enum

class MetricType(Enum):
    """Metric type enum."""
    
    LATENCY = "latency"
    TOKEN_COUNT = "token_count"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    ACTIVE_CONNECTIONS = "active_connections"
    RETRY_COUNT = "retry_count"
    STEP_DURATION = "step_duration"
    WORKFLOW_DURATION = "workflow_duration"
    AGENT_COUNT = "agent_count"
    STEP_COUNT = "step_count"
    MESSAGE_COUNT = "message_count"
    DATA_SIZE = "data_size" 