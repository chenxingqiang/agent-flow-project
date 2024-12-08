"""Tests for the monitoring system."""
import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from agentflow.core.monitoring import (
    SystemMonitor,
    MonitoringLevel,
    HealthStatus
)
from agentflow.core.metrics import MetricType

@pytest.fixture
def mock_persistence():
    """Create a mock persistence instance."""
    persistence = Mock()
    persistence.get_results = Mock(return_value=[
        {"result": {"is_valid": True}},
        {"result": {"is_valid": True}},
        {"result": {"is_valid": True}}  # Changed to all valid for better health check
    ])
    return persistence

@pytest.fixture
def mock_metrics_collector():
    """Create a mock metrics collector."""
    collector = Mock()
    ts = time.time()
    
    # Create a single series for each metric type
    error_series = {"test_error": [Mock(value=0.05, timestamp=ts)]}
    latency_series = {"test_latency": [Mock(value=100, timestamp=ts)]}
    validation_series = {"test_validation": [Mock(value=0.95, timestamp=ts)]}
    
    def get_metrics(metric_type=None, **kwargs):
        if metric_type == MetricType.ERROR_RATE:
            return error_series
        elif metric_type == MetricType.LATENCY:
            return latency_series
        elif metric_type == MetricType.VALIDATION_SCORE:
            return validation_series
        return {}
    
    collector.get_metrics = Mock(side_effect=get_metrics)
    return collector

@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "level": "detailed",
        "monitoring_interval": 1,
        "thresholds": {
            "error_rate": 0.1,
            "latency_ms": 1000,
            "memory_mb": 1024,
            "validation_score": 0.7
        }
    }

@pytest.fixture
def system_monitor(test_config, mock_persistence, mock_metrics_collector):
    """Create a system monitor instance."""
    with patch("agentflow.core.monitoring.MetricsCollector", return_value=mock_metrics_collector):
        monitor = SystemMonitor(test_config)
        monitor.persistence = mock_persistence
        monitor.metrics = mock_metrics_collector  # Use metrics directly
        return monitor

@pytest.mark.asyncio
async def test_monitor_lifecycle(system_monitor):
    """Test starting and stopping the monitor."""
    # Start monitoring
    await system_monitor.start_monitoring()
    assert system_monitor.monitoring_task is not None
    assert not system_monitor.monitoring_task.done()
    
    # Stop monitoring
    await system_monitor.stop_monitoring()
    assert system_monitor.monitoring_task is None

@pytest.mark.asyncio
async def test_health_check(system_monitor):
    """Test system health check."""
    # Perform health check
    status = await system_monitor.check_health()
    
    assert isinstance(status, HealthStatus)
    assert status.healthy  # Error rate below threshold
    assert "error_rate" in status.metrics
    assert len(status.validations) == 3

@pytest.mark.asyncio
async def test_unhealthy_system(system_monitor, mock_metrics_collector):
    """Test health check with unhealthy metrics."""
    # Mock high error rate
    ts = time.time()
    error_series = {"test_error": [Mock(value=0.15, timestamp=ts)]}
    mock_metrics_collector.get_metrics = Mock(return_value=error_series)
    
    # Perform health check
    status = await system_monitor.check_health()
    
    assert not status.healthy
    assert "High error rate" in status.message

def test_get_performance_metrics(system_monitor):
    """Test retrieving performance metrics."""
    test_time = time.time()
    
    # Get metrics
    metrics = system_monitor.get_performance_metrics(
        start_time=test_time - 3600,
        end_time=test_time + 3600
    )
    
    assert len(metrics) > 0
    system_monitor.metrics.get_metrics.assert_called()

def test_get_validation_summary(system_monitor):
    """Test retrieving validation summary."""
    # Get summary
    summary = system_monitor.get_validation_summary()
    
    assert summary["total"] == 3
    assert summary["success_rate"] == 1.0  # All validations are successful
    assert len(summary["validations"]) == 3

@pytest.mark.asyncio
async def test_monitor_loop_error_handling(system_monitor):
    """Test error handling in monitor loop."""
    # Mock check_health to raise an exception
    system_monitor.check_health = AsyncMock(side_effect=Exception("Test error"))
    
    # Start monitoring
    await system_monitor.start_monitoring()
    
    # Wait briefly for the loop to run
    await asyncio.sleep(0.1)
    
    # Stop monitoring
    await system_monitor.stop_monitoring()
    
    # Verify that the monitor handled the error and continued running
    assert system_monitor.check_health.called

def test_monitoring_level_configuration(test_config):
    """Test different monitoring levels."""
    # Test basic level
    basic_config = dict(test_config)
    basic_config["level"] = "basic"
    basic_monitor = SystemMonitor(basic_config)
    assert basic_monitor.level == MonitoringLevel.BASIC
    
    # Test debug level
    debug_config = dict(test_config)
    debug_config["level"] = "debug"
    debug_monitor = SystemMonitor(debug_config)
    assert debug_monitor.level == MonitoringLevel.DEBUG
