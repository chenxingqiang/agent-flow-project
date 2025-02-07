"""Metrics module for AgentFlow."""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)

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
        self._initialized = False
        
    async def initialize(self):
        """Initialize metrics manager."""
        if not self._initialized:
            # Initialize metrics storage
            self.metrics = {}
            self._initialized = True
            
    async def cleanup(self) -> None:
        """Clean up metrics manager resources."""
        try:
            # Reset metrics
            self.metrics = {}
            self._initialized = False
        except Exception as e:
            logger.error(f"Error during metrics cleanup: {str(e)}")
            raise
        
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

    def get_metric(self, name: str, metric_type: MetricType) -> Optional[Any]:
        """Get a metric value.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            
        Returns:
            Metric value if found, None otherwise
            
        Raises:
            ValueError: If metric type is invalid
        """
        if not self._initialized:
            raise RuntimeError("Metrics manager not initialized")
            
        if metric_type == MetricType.COUNTER:
            return self.metrics["counters"].get(name)
        elif metric_type == MetricType.GAUGE:
            return self.metrics["gauges"].get(name)
        elif metric_type == MetricType.HISTOGRAM:
            return self.metrics["histograms"].get(name)
        elif metric_type == MetricType.SUMMARY:
            return self.metrics["summaries"].get(name)
        else:
            raise ValueError(f"Invalid metric type: {metric_type}")
            
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics.
        
        Returns:
            Dict of all metrics
            
        Raises:
            RuntimeError: If metrics manager is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Metrics manager not initialized")
            
        return self.metrics

# Alias for backward compatibility with existing tests
MetricsCollector = MetricsManager
