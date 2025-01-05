"""Monitoring module for AgentFlow."""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MetricPoint:
    """Data point for a metric."""
    
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Collector for various metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.logger = logging.getLogger(__name__)
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            metadata=metadata or {}
        )
        self.metrics[name].append(point)
    
    def get_metric_history(self, name: str) -> List[MetricPoint]:
        """Get history of a metric."""
        return self.metrics.get(name, [])
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get latest value of a metric."""
        history = self.get_metric_history(name)
        return history[-1].value if history else None
    
    def clear_metrics(self):
        """Clear all metrics."""
        self.metrics.clear()

class Monitor:
    """System and performance monitor."""
    
    def __init__(self):
        """Initialize monitor."""
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """Start monitoring system metrics."""
        self._record_system_metrics()
        
    def _record_system_metrics(self):
        """Record system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.record_metric(
            "cpu_usage",
            cpu_percent,
            {"unit": "percent"}
        )
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics_collector.record_metric(
            "memory_usage",
            memory.percent,
            {"unit": "percent", "total": memory.total}
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics_collector.record_metric(
            "disk_usage",
            disk.percent,
            {"unit": "percent", "total": disk.total}
        )
        
    def record_execution_time(self, name: str, start_time: float):
        """Record execution time for an operation."""
        duration = time.time() - start_time
        self.metrics_collector.record_metric(
            f"{name}_execution_time",
            duration,
            {"unit": "seconds"}
        )
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            "cpu_usage": self.metrics_collector.get_latest_value("cpu_usage"),
            "memory_usage": self.metrics_collector.get_latest_value("memory_usage"),
            "disk_usage": self.metrics_collector.get_latest_value("disk_usage")
        }
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = {}
        for name, points in self.metrics_collector.metrics.items():
            if name.endswith("_execution_time"):
                metrics[name] = points[-1].value if points else None
        return metrics 