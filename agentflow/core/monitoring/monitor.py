"""Real-time performance monitoring and anomaly detection."""
import time
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass
from ..utils.metrics import MetricsCollector

@dataclass
class Alert:
    """Alert data structure."""
    severity: str
    message: str
    timestamp: float
    metrics: Dict[str, float]
    context: Dict[str, Any]

class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = MetricsCollector()
        self.anomaly_detector = AnomalyDetector(config)
        self.alert_manager = AlertManager(config)
        
        # Performance thresholds
        self.thresholds = {
            "latency": config.get("latency_threshold", 1000),  # ms
            "error_rate": config.get("error_threshold", 0.05),
            "memory_usage": config.get("memory_threshold", 0.9),
            "cpu_usage": config.get("cpu_threshold", 0.8)
        }
    
    def collect_metrics(self, metrics: Dict[str, Any]):
        """Collect and analyze new metrics."""
        self.metrics.add_metrics(metrics)
        
        # Check for anomalies
        if self.anomaly_detector.check_anomalies(metrics):
            self.alert_manager.create_alert(
                "WARNING",
                "Anomaly detected in system metrics",
                metrics
            )
        
        # Check thresholds
        self._check_thresholds(metrics)
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check if metrics exceed defined thresholds."""
        for metric, threshold in self.thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                self.alert_manager.create_alert(
                    "CRITICAL",
                    f"{metric} exceeded threshold: {metrics[metric]:.2f} > {threshold}",
                    metrics
                )

class AnomalyDetector:
    """Anomaly detection using statistical methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history = []
        self.window_size = config.get("window_size", 100)
        self.threshold = config.get("anomaly_threshold", 3.0)
    
    def check_anomalies(self, metrics: Dict[str, Any]) -> bool:
        """Check for anomalies in metrics."""
        self.history.append(metrics)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Calculate z-scores for each metric
        anomalies = []
        for key in metrics:
            if len(self.history) > 10:  # Need minimum history
                values = [h[key] for h in self.history if key in h]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    if std > 0:
                        z_score = abs((metrics[key] - mean) / std)
                        anomalies.append(z_score > self.threshold)
        
        return any(anomalies)

class AlertManager:
    """Alert management system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alerts: List[Alert] = []
        self.alert_handlers = []
    
    def create_alert(self, severity: str, message: str, context: Dict[str, Any]):
        """Create and handle a new alert."""
        alert = Alert(
            severity=severity,
            message=message,
            timestamp=time.time(),
            metrics=context,
            context={"environment": self.config.get("environment", "prod")}
        )
        self.alerts.append(alert)
        self._handle_alert(alert)
    
    def add_handler(self, handler):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def _handle_alert(self, alert: Alert):
        """Process alert through all handlers."""
        for handler in self.alert_handlers:
            handler(alert)
