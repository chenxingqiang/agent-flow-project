"""Monitoring system for AgentFlow framework."""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
import asyncio
from dataclasses import dataclass
from enum import Enum

from .metrics import MetricsCollector, MetricType
from .persistence import PersistenceFactory

class MonitoringLevel(Enum):
    """Monitoring detail levels."""
    BASIC = "basic"  # Basic metrics and validations
    DETAILED = "detailed"  # Detailed metrics and all validations
    DEBUG = "debug"  # All metrics, validations, and debug info

@dataclass
class HealthStatus:
    """System health status."""
    healthy: bool
    message: str
    metrics: Dict[str, float]
    validations: List[Dict[str, Any]]
    timestamp: float

class SystemMonitor:
    """Monitors system health and performance."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize system monitor.
        
        Args:
            config: Monitor configuration
        """
        self.config = config
        self.level = MonitoringLevel(config.get("level", "basic"))
        
        # Initialize persistence
        persistence_config = config.get("persistence", {})
        persistence_type = persistence_config.get("type", "file")
        self.persistence = PersistenceFactory.create_persistence(
            persistence_type,
            **persistence_config.get("config", {})
        )
        
        # Initialize metrics collector
        self.metrics = MetricsCollector(self.persistence)
        
        # Health check thresholds
        self.thresholds = config.get("thresholds", {
            "error_rate": 0.1,  # 10% error rate threshold
            "latency_ms": 1000,  # 1 second latency threshold
            "memory_mb": 1024,  # 1GB memory threshold
            "validation_score": 0.7  # 70% validation score threshold
        })
        
        # Background monitoring task
        self.monitoring_task = None
        self.monitoring_interval = config.get("monitoring_interval", 60)  # seconds
        
    async def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitor_loop())
            
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            
    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await self.check_health()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
                
    async def check_health(self) -> HealthStatus:
        """Check system health status.
        
        Returns:
            HealthStatus object with current health information
        """
        current_metrics = {}
        issues = []
        
        # Check error rate
        error_metrics = self.metrics.get_metrics(
            metric_type=MetricType.ERROR_RATE,
            start_time=time.time() - 3600  # Last hour
        )
        if error_metrics:
            error_rate = sum(p.value for points in error_metrics.values() for p in points) / len(error_metrics)
            current_metrics["error_rate"] = error_rate
            if error_rate > self.thresholds["error_rate"]:
                issues.append(f"High error rate: {error_rate:.2%}")
                
        # Check latency
        latency_metrics = self.metrics.get_metrics(
            metric_type=MetricType.LATENCY,
            start_time=time.time() - 3600
        )
        if latency_metrics:
            avg_latency = sum(p.value for points in latency_metrics.values() for p in points) / len(latency_metrics)
            current_metrics["latency_ms"] = avg_latency
            if avg_latency > self.thresholds["latency_ms"]:
                issues.append(f"High latency: {avg_latency:.0f}ms")
                
        # Check validation scores
        validation_metrics = self.metrics.get_metrics(
            metric_type=MetricType.VALIDATION_SCORE,
            start_time=time.time() - 3600
        )
        if validation_metrics:
            avg_score = sum(p.value for points in validation_metrics.values() for p in points) / len(validation_metrics)
            current_metrics["validation_score"] = avg_score
            if avg_score < self.thresholds["validation_score"]:
                issues.append(f"Low validation score: {avg_score:.2%}")
                
        # Get recent validations
        recent_validations = self.persistence.get_results(
            objective_id="system",
            start_time=str(time.time() - 3600)
        )
        
        # Determine overall health
        healthy = len(issues) == 0
        message = "System healthy" if healthy else "; ".join(issues)
        
        status = HealthStatus(
            healthy=healthy,
            message=message,
            metrics=current_metrics,
            validations=recent_validations,
            timestamp=time.time()
        )
        
        # Record health check
        self.metrics.record_metric(
            metric_type=MetricType.SUCCESS_RATE,
            value=1.0 if healthy else 0.0,
            labels={"check_type": "health"}
        )
        
        return status
        
    def get_performance_metrics(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Get system performance metrics.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary of metric series with timestamps
        """
        metrics = {}
        
        for metric_type in MetricType:
            metric_data = self.metrics.get_metrics(
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time
            )
            
            if metric_data:
                metrics[metric_type.value] = [
                    (point.timestamp, point.value)
                    for points in metric_data.values()
                    for point in points
                ]
                
        return metrics
        
    def get_validation_summary(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of validation results.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Summary of validation results
        """
        validations = self.persistence.get_results(
            objective_id="system",
            start_time=start_time,
            end_time=end_time
        )
        
        total = len(validations)
        if total == 0:
            return {
                "total": 0,
                "success_rate": 0.0,
                "validations": []
            }
            
        successful = sum(1 for v in validations if v.get("result", {}).get("is_valid", False))
        
        return {
            "total": total,
            "success_rate": successful / total,
            "validations": validations
        }
