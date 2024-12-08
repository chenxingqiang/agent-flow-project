"""Tests for ELL controller integration."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from agentflow.integrations.ell_controller import ELLController

# Test configurations
TEST_CONFIG = {
    "persistence": {
        "storage_path": "/tmp/test_storage",
        "backup_interval": 300
    },
    "ell_integration": {
        "api_endpoint": "http://localhost:8000",
        "sync_interval": 60,
        "metrics_mapping": {
            "cpu_usage": "system.cpu",
            "memory_usage": "system.memory"
        }
    },
    "visualization": {
        "refresh_rate": 5,
        "default_layout": "grid"
    }
}

@pytest.fixture
def mock_ell_adapter():
    """Create a mock ELL adapter."""
    adapter = AsyncMock()
    adapter.monitoring = AsyncMock()
    adapter.monitoring.sync_health_status = AsyncMock(return_value={"status": "healthy"})
    adapter.get_ell_config = AsyncMock(return_value=TEST_CONFIG["ell_integration"])
    adapter.get_ell_validation_results = AsyncMock(return_value=[])
    adapter.send_validation_results = AsyncMock()
    adapter.start = AsyncMock()
    return adapter

@pytest.fixture
def mock_dashboard_manager():
    """Create a mock dashboard manager."""
    manager = AsyncMock()
    manager.list_dashboards = Mock(return_value=["system_overview", "validation_dashboard"])
    manager.create_dashboard = Mock(return_value="new_dashboard_id")
    manager.start = AsyncMock()
    return manager

@pytest.fixture
async def ell_controller(mock_ell_adapter, mock_dashboard_manager):
    """Create a test ELL controller instance."""
    controller = ELLController(TEST_CONFIG)
    controller.ell_adapter = mock_ell_adapter
    controller.dashboard_manager = mock_dashboard_manager
    await controller.initialize()
    return controller

@pytest.mark.asyncio
async def test_initialization(ell_controller):
    """Test controller initialization."""
    controller = await ell_controller
    assert controller.initialized
    assert not controller._shutdown

@pytest.mark.asyncio
async def test_shutdown(ell_controller):
    """Test controller shutdown."""
    controller = await ell_controller
    await controller.shutdown()
    assert controller._shutdown

@pytest.mark.asyncio
async def test_get_system_status(ell_controller):
    """Test getting system status."""
    controller = await ell_controller
    status = await controller.get_system_status()
    assert "health" in status
    assert "dashboards" in status
    assert "timestamp" in status

@pytest.mark.asyncio
async def test_update_configuration(ell_controller):
    """Test configuration updates."""
    controller = await ell_controller
    new_config = {
        "ell_integration": {
            "sync_interval": 120,
            "metrics_mapping": {"new_metric": "new.path"}
        }
    }
    
    result = await controller.update_configuration(new_config)
    assert "ell_config" in result
    assert "dashboards" in result

@pytest.mark.asyncio
async def test_create_custom_dashboard(ell_controller):
    """Test custom dashboard creation."""
    controller = await ell_controller
    dashboard_type = "custom_view"
    config = {
        "layout": "vertical",
        "components": ["metrics", "health"]
    }
    
    dashboard_id = controller.create_custom_dashboard(dashboard_type, config)
    assert dashboard_id == "new_dashboard_id"

@pytest.mark.asyncio
async def test_get_validation_results(ell_controller):
    """Test retrieving validation results."""
    controller = await ell_controller
    start_time = (datetime.now() - timedelta(hours=1)).isoformat()
    end_time = datetime.now().isoformat()
    
    results = await controller.get_validation_results(start_time, end_time)
    assert "ell_results" in results
    assert "agentflow_results" in results
    assert "timestamp" in results

@pytest.mark.asyncio
async def test_sync_validation_results(ell_controller):
    """Test synchronizing validation results."""
    controller = await ell_controller
    test_results = [
        {
            "id": "test1",
            "status": "passed",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    await controller.sync_validation_results(test_results)
    controller.ell_adapter.send_validation_results.assert_called_once_with(test_results)

@pytest.mark.asyncio
async def test_error_handling(ell_controller):
    """Test error handling in controller methods."""
    controller = await ell_controller
    controller.ell_adapter.monitoring.sync_health_status.side_effect = Exception("Test error")
    
    status = await controller.get_system_status()
    assert "error" in status
    assert "timestamp" in status

@pytest.mark.asyncio
async def test_concurrent_operations(ell_controller):
    """Test concurrent operations handling."""
    controller = await ell_controller
    operations = [
        controller.get_system_status(),
        controller.get_validation_results(
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ),
        controller.update_configuration({"test": "config"})
    ]
    
    results = await asyncio.gather(*operations, return_exceptions=True)
    assert len(results) == 3
    assert not any(isinstance(r, Exception) for r in results)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
