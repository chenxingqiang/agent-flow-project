"""ELL integration module for AgentFlow."""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

from ..core.metrics import MetricsCollector, MetricType
from ..core.monitoring import SystemMonitor

logger = logging.getLogger(__name__)

@dataclass
class ELLMetricAdapter:
    """Adapter to convert ELL metrics to AgentFlow metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize the adapter.
        
        Args:
            metrics_collector: AgentFlow metrics collector
        """
        self.metrics_collector = metrics_collector
        
    def record_ell_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None
    ):
        """Record an ELL metric in AgentFlow format.
        
        Args:
            metric_name: Name of the ELL metric
            value: Metric value
            labels: Optional metric labels
            timestamp: Optional metric timestamp
        """
        # Map ELL metric names to AgentFlow metric types
        metric_type_mapping = {
            "latency": MetricType.LATENCY,
            "memory": MetricType.MEMORY_USAGE,
            "tokens": MetricType.TOKEN_COUNT,
            "api_calls": MetricType.API_CALLS,
            "validation": MetricType.VALIDATION_SCORE,
            "error_rate": MetricType.ERROR_RATE,
            "success_rate": MetricType.SUCCESS_RATE
        }
        
        # Default to API_CALLS if no mapping exists
        metric_type = metric_type_mapping.get(
            metric_name.lower(),
            MetricType.API_CALLS
        )
        
        if labels is None:
            labels = {}
        labels["source"] = "ell"
        labels["original_metric"] = metric_name
        
        self.metrics_collector.record_metric(
            metric_type=metric_type,
            value=value,
            labels=labels
        )

class ELLMonitoringIntegration:
    """Integration of ELL monitoring with AgentFlow monitoring."""
    
    def __init__(self, system_monitor: SystemMonitor):
        """Initialize the integration.
        
        Args:
            system_monitor: AgentFlow system monitor
        """
        self.system_monitor = system_monitor
        self.metric_adapter = ELLMetricAdapter(system_monitor.metrics)
        
    def register_ell_metrics(self, ell_metrics: Dict[str, Any]):
        """Register ELL metrics with AgentFlow monitoring.
        
        Args:
            ell_metrics: Dictionary of ELL metrics
        """
        for metric_name, metric_data in ell_metrics.items():
            if isinstance(metric_data, (int, float)):
                value = float(metric_data)
            elif isinstance(metric_data, dict) and "value" in metric_data:
                value = float(metric_data["value"])
            else:
                logger.warning(f"Skipping invalid metric data for {metric_name}")
                continue
                
            self.metric_adapter.record_ell_metric(
                metric_name=metric_name,
                value=value,
                labels=metric_data.get("labels") if isinstance(metric_data, dict) else None
            )
            
    async def sync_health_status(self) -> Dict[str, Any]:
        """Synchronize health status between ELL and AgentFlow.
        
        Returns:
            Combined health status
        """
        # Get AgentFlow health status
        agentflow_status = await self.system_monitor.check_health()
        
        # Combine with ELL status (placeholder for actual ELL status)
        combined_status = {
            "healthy": agentflow_status.healthy,
            "message": agentflow_status.message,
            "metrics": {
                "agentflow": agentflow_status.metrics,
                "ell": {}  # To be populated with actual ELL metrics
            },
            "validations": agentflow_status.validations,
            "timestamp": agentflow_status.timestamp
        }
        
        return combined_status
