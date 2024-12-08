"""Tests for the metrics collection system."""
import pytest
import time
from unittest.mock import Mock, patch
from agentflow.core.metrics import (
    MetricsCollector,
    MetricType,
    MetricPoint
)

@pytest.fixture
def mock_persistence():
    """Create a mock persistence instance."""
    return Mock()

@pytest.fixture
def metrics_collector(mock_persistence):
    """Create a metrics collector instance."""
    return MetricsCollector(mock_persistence)

def test_record_metric(metrics_collector, mock_persistence):
    """Test recording a single metric."""
    # Record a test metric
    metrics_collector.record_metric(
        metric_type=MetricType.LATENCY,
        value=100.0,
        labels={"service": "test"}
    )
    
    # Check that metric was recorded
    metric_key = f"{MetricType.LATENCY.value}:service=test"
    assert metric_key in metrics_collector.current_metrics
    assert len(metrics_collector.current_metrics[metric_key]) == 1
    
    # Verify metric point
    point = metrics_collector.current_metrics[metric_key][0]
    assert point.metric_type == MetricType.LATENCY
    assert point.value == 100.0
    assert point.labels == {"service": "test"}
    assert isinstance(point.timestamp, float)
    
    # Verify persistence was called
    mock_persistence.save_result.assert_called_once()
    call_args = mock_persistence.save_result.call_args[1]
    assert call_args["objective_id"] == "system"
    assert call_args["validation_type"] == "metric_latency"
    
def test_get_metrics_with_filters(metrics_collector):
    """Test retrieving metrics with various filters."""
    # Record test metrics
    now = time.time()
    test_metrics = [
        (MetricType.LATENCY, 100.0, {"service": "api"}),
        (MetricType.LATENCY, 200.0, {"service": "db"}),
        (MetricType.TOKEN_COUNT, 1000, {"service": "api"}),
    ]
    
    for metric_type, value, labels in test_metrics:
        metrics_collector.record_metric(
            metric_type=metric_type,
            value=value,
            labels=labels
        )
        
    # Test filtering by metric type
    latency_metrics = metrics_collector.get_metrics(
        metric_type=MetricType.LATENCY
    )
    assert len(latency_metrics) == 2
    
    # Test filtering by time range
    recent_metrics = metrics_collector.get_metrics(
        start_time=now - 1
    )
    assert len(recent_metrics) == 3
    
    # Test filtering by labels
    api_metrics = metrics_collector.get_metrics(
        labels={"service": "api"}
    )
    assert len(api_metrics) == 2
    
def test_clear_metrics(metrics_collector):
    """Test clearing metrics."""
    # Record test metrics
    metrics_collector.record_metric(
        metric_type=MetricType.LATENCY,
        value=100.0,
        labels={"service": "test1"}
    )
    metrics_collector.record_metric(
        metric_type=MetricType.TOKEN_COUNT,
        value=1000,
        labels={"service": "test2"}
    )
    
    # Clear metrics by type
    metrics_collector.clear_metrics(metric_type=MetricType.LATENCY)
    remaining_metrics = metrics_collector.get_metrics()
    assert len(remaining_metrics) == 1
    assert list(remaining_metrics.keys())[0].startswith(MetricType.TOKEN_COUNT.value)
    
    # Clear all metrics
    metrics_collector.clear_metrics()
    assert len(metrics_collector.get_metrics()) == 0
    
def test_metric_point_creation():
    """Test creation of metric points."""
    point = MetricPoint(
        metric_type=MetricType.LATENCY,
        value=100.0,
        timestamp=time.time(),
        labels={"test": "label"}
    )
    
    assert point.metric_type == MetricType.LATENCY
    assert point.value == 100.0
    assert isinstance(point.timestamp, float)
    assert point.labels == {"test": "label"}
