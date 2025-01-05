"""Tests for the metrics collection system."""
import pytest
import time
from agentflow.core.metrics import MetricsCollector, MetricType

@pytest.fixture
def metrics_collector():
    """Create a metrics collector instance."""
    return MetricsCollector()

def test_record_metric(metrics_collector):
    """Test recording a single metric."""
    metrics_collector.record_metric(
        metric_type=MetricType.LATENCY,
        value=100.0,
        labels={"service": "api"}
    )
    
    metrics = metrics_collector.get_metrics(metric_type=MetricType.LATENCY)
    assert len(metrics) == 1
    assert len(metrics[MetricType.LATENCY.value]) == 1
    assert metrics[MetricType.LATENCY.value][0].value == 100.0
    assert metrics[MetricType.LATENCY.value][0].labels == {"service": "api"}

def test_get_metrics_with_filters(metrics_collector):
    """Test getting metrics with filters."""
    # Record some metrics
    metrics_collector.record_metric(
        metric_type=MetricType.LATENCY,
        value=100,
        labels={"service": "api"}
    )
    metrics_collector.record_metric(
        metric_type=MetricType.THROUGHPUT,
        value=50,
        labels={"service": "api"}
    )
    metrics_collector.record_metric(
        metric_type=MetricType.ERROR_RATE,
        value=0.1,
        labels={"service": "api"}
    )
    metrics_collector.record_metric(
        metric_type=MetricType.SUCCESS_RATE,
        value=0.9,
        labels={"service": "api"}
    )

    # Test filtering by type
    latency_metrics = metrics_collector.get_metrics(metric_type=MetricType.LATENCY)
    assert len(latency_metrics[MetricType.LATENCY.value]) == 1
    assert latency_metrics[MetricType.LATENCY.value][0].value == 100

    # Test filtering by time range
    now = time.time()
    time_metrics = metrics_collector.get_metrics(start_time=now - 3600, end_time=now + 3600)
    assert len(time_metrics) == 4

def test_clear_metrics(metrics_collector):
    """Test clearing metrics."""
    # Record some metrics
    metrics_collector.record_metric(
        metric_type=MetricType.LATENCY,
        value=100,
        labels={"service": "api"}
    )
    metrics_collector.record_metric(
        metric_type=MetricType.THROUGHPUT,
        value=50,
        labels={"service": "api"}
    )
    metrics_collector.record_metric(
        metric_type=MetricType.ERROR_RATE,
        value=0.1,
        labels={"service": "api"}
    )
    metrics_collector.record_metric(
        metric_type=MetricType.SUCCESS_RATE,
        value=0.9,
        labels={"service": "api"}
    )

    assert len(metrics_collector.get_metrics()) == 4

    # Clear metrics
    metrics_collector.clear_metrics()
    assert len(metrics_collector.get_metrics()) == 0
