"""Metrics collection and monitoring system for AgentFlow."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import time
from dataclasses import dataclass
from enum import Enum

class MetricType(Enum):
    """Types of metrics that can be collected."""
    LATENCY = "latency"
    TOKEN_COUNT = "token_count"
    MEMORY_USAGE = "memory_usage"
    API_CALLS = "api_calls"
    VALIDATION_SCORE = "validation_score"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"

@dataclass
class MetricPoint:
    """Single point of metric data."""
    metric_type: MetricType
    value: float
    timestamp: float
    labels: Dict[str, str]
    
class MetricsCollector:
    """Collects and manages metrics for AgentFlow components."""
    
    def __init__(self, persistence):
        """Initialize metrics collector.
        
        Args:
            persistence: Persistence instance for storing metrics
        """
        self.persistence = persistence
        self.current_metrics: Dict[str, List[MetricPoint]] = {}
        
    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a new metric point.
        
        Args:
            metric_type: Type of metric
            value: Metric value
            labels: Optional key-value pairs for metric labeling
        """
        if labels is None:
            labels = {}
            
        metric_point = MetricPoint(
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            labels=labels
        )
        
        metric_key = f"{metric_type.value}:{'.'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        
        if metric_key not in self.current_metrics:
            self.current_metrics[metric_key] = []
        self.current_metrics[metric_key].append(metric_point)
        
        # Persist metric
        self.persistence.save_result(
            objective_id=labels.get("objective_id", "system"),
            validation_type=f"metric_{metric_type.value}",
            result={
                "value": value,
                "timestamp": metric_point.timestamp,
                "labels": labels
            }
        )
        
    def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[MetricPoint]]:
        """Get metrics matching the specified criteria.
        
        Args:
            metric_type: Optional metric type filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            labels: Optional label filters
            
        Returns:
            Dictionary of metric series
        """
        filtered_metrics = {}
        
        for key, points in self.current_metrics.items():
            if metric_type and not key.startswith(metric_type.value):
                continue
                
            matching_points = []
            for point in points:
                if start_time and point.timestamp < start_time:
                    continue
                if end_time and point.timestamp > end_time:
                    continue
                if labels:
                    matches_labels = True
                    for k, v in labels.items():
                        if k not in point.labels or point.labels[k] != v:
                            matches_labels = False
                            break
                    if not matches_labels:
                        continue
                matching_points.append(point)
                
            if matching_points:
                filtered_metrics[key] = matching_points
                
        return filtered_metrics
        
    def clear_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        before_time: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Clear metrics matching the specified criteria.
        
        Args:
            metric_type: Optional metric type to clear
            before_time: Optional timestamp to clear metrics before
            labels: Optional labels to match for clearing
        """
        keys_to_remove = []
        
        for key, points in self.current_metrics.items():
            if metric_type and not key.startswith(metric_type.value):
                continue
                
            if before_time:
                # Keep points after before_time
                self.current_metrics[key] = [
                    p for p in points if p.timestamp >= before_time
                ]
                if not self.current_metrics[key]:
                    keys_to_remove.append(key)
            elif labels:
                # Keep points that don't match labels
                self.current_metrics[key] = [
                    p for p in points
                    if not all(
                        k in p.labels and p.labels[k] == v
                        for k, v in labels.items()
                    )
                ]
                if not self.current_metrics[key]:
                    keys_to_remove.append(key)
            else:
                # Clear all points for this metric type
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.current_metrics[key]
