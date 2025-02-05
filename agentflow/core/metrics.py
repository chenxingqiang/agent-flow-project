"""Metrics module for AgentFlow."""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import time

class MetricType(str, Enum):
    """Metric type enum."""
    
    LATENCY = "latency"
    TOKEN_COUNT = "token_count"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    PERFORMANCE = "performance"
    RESOURCE_USAGE = "resource_usage"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    NETWORK_USAGE = "network_usage"
    DISK_USAGE = "disk_usage"
    QUEUE_SIZE = "queue_size"
    BACKLOG = "backlog"
    ACTIVE_CONNECTIONS = "active_connections"
    RESPONSE_TIME = "response_time"
    REQUEST_COUNT = "request_count"
    VALIDATION_SCORE = "validation_score"
    CUSTOM = "custom"

@dataclass
class MetricPoint:
    """Metric data point."""
    metric_type: MetricType
    value: float
    timestamp: float
    labels: Dict[str, str]

class MetricsManager:
    """Metrics manager class."""
    
    def __init__(self, persistence: Optional[Any] = None):
        """Initialize metrics manager.
        
        Args:
            persistence: Optional persistence layer for storing metrics
        """
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.persistence = persistence
        
    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric."""
        labels = labels or {}
        metric_key = metric_type.value
        
        if metric_key not in self.metrics:
            self.metrics[metric_key] = []
            
        metric_point = MetricPoint(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now().timestamp(),
            labels=labels
        )
        
        self.metrics[metric_key].append(metric_point)
        
        # Persist metric if persistence layer is available
        if self.persistence:
            self.persistence.store_metric(metric_point)
        
    def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[MetricPoint]]:
        """Get metrics with optional filtering."""
        filtered_metrics = {}
        
        # If metric_type is specified, only look at that type
        metric_keys = [metric_type.value] if metric_type else self.metrics.keys()
        
        for key in metric_keys:
            if key not in self.metrics:
                continue
                
            filtered_points = []
            for point in self.metrics[key]:
                # Apply time filters
                if start_time and point.timestamp < start_time:
                    continue
                if end_time and point.timestamp > end_time:
                    continue
                    
                # Apply label filters
                if labels:
                    matches_labels = True
                    for k, v in labels.items():
                        if k not in point.labels or point.labels[k] != v:
                            matches_labels = False
                            break
                    if not matches_labels:
                        continue
                        
                filtered_points.append(point)
                
            if filtered_points:
                filtered_metrics[key] = filtered_points
                
        return filtered_metrics
        
    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self.metrics = {}

    async def cleanup(self) -> None:
        """Clean up metrics manager resources."""
        # Clear metrics
        self.clear_metrics()
        
        # Clean up persistence if available
        if self.persistence and hasattr(self.persistence, 'cleanup'):
            await self.persistence.cleanup()

# Alias for backward compatibility with existing tests
MetricsCollector = MetricsManager
