"""Monitoring module for AgentFlow."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict

class MetricsCollector:
    """Collector for performance and operational metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.tags: Dict[str, Dict[str, str]] = {}
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        self.metrics[name].append(value)
        if tags:
            self.tags[name] = tags
        self.logger.debug(f"Recorded metric: {name}={value} with tags {tags}")
        
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = self.metrics.get(name, [])
        if not values:
            return {}
            
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values)
        }
        
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics with their stats and tags."""
        return {
            name: {
                "stats": self.get_metric_stats(name),
                "tags": self.tags.get(name, {})
            }
            for name in self.metrics
        }
        
    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self.tags.clear()
        self.logger.debug("Cleared all metrics")

class Monitor:
    """Monitor class for tracking workflow execution."""
    
    def __init__(self):
        """Initialize monitor."""
        self.logger = logging.getLogger(__name__)
        self.start_time: Optional[datetime] = None
        self.metrics: Dict[str, Any] = {}
        self.metrics_collector = MetricsCollector()
        
    def start_monitoring(self) -> None:
        """Start monitoring."""
        self.start_time = datetime.now()
        self.logger.info("Started monitoring")
        
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        if self.start_time:
            duration = datetime.now() - self.start_time
            self.metrics["duration"] = duration.total_seconds()
            self.logger.info(f"Stopped monitoring. Duration: {duration}")
            
    def record_metric(self, name: str, value: Any) -> None:
        """Record a metric."""
        self.metrics[name] = value
        self.logger.debug(f"Recorded metric: {name}={value}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return self.metrics.copy()
        
    def clear_metrics(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()
        self.logger.debug("Cleared all metrics")
        
    def record_performance_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a performance metric."""
        self.metrics_collector.record_metric(name, value, tags)
        
    def get_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all performance metrics."""
        return self.metrics_collector.get_all_metrics() 