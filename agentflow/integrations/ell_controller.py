"""ELL integration controller for AgentFlow."""

from typing import Dict, Any, List, Optional
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class ELLController:
    """Controller for ELL integration with AgentFlow."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the controller.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize mock components for testing
        self.metrics_collector = MockMetricsCollector()
        self.ell_adapter = MockELLAdapter()
        self.dashboard_manager = MockDashboardManager()
        
        # State tracking
        self.initialized = False
        self._shutdown = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
    
    async def initialize(self):
        """Initialize the ELL integration."""
        if self.initialized:
            return
            
        try:
            # Start components
            await self.ell_adapter.start()
            await self.dashboard_manager.start()
            
            # Create default dashboards
            self.dashboard_manager.create_dashboard("system_overview")
            self.dashboard_manager.create_dashboard("validation_dashboard")
            
            self.initialized = True
            logger.info("ELL integration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ELL integration: {e}")
            raise
            
    async def shutdown(self):
        """Shutdown the ELL integration."""
        self._shutdown = True
        
        try:
            # Stop components
            await self.ell_adapter.stop()
            await self.dashboard_manager.stop()
            
            logger.info("ELL integration shut down")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status.
        
        Returns:
            Dictionary containing system status
        """
        try:
            # Get health status
            health_status = await self.ell_adapter.monitoring.sync_health_status()
            
            # Get active dashboards
            dashboards = self.dashboard_manager.list_dashboards()
            
            # Get ELL config
            ell_config = self.ell_adapter.get_ell_config()
            
            return {
                "health": health_status,
                "dashboards": dashboards,
                "config": ell_config,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    async def update_configuration(
        self,
        new_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update ELL integration configuration.
        
        Args:
            new_config: New configuration values
            
        Returns:
            Updated configuration
        """
        try:
            # Update ELL config
            ell_config = await self.ell_adapter.update_ell_config(
                new_config.get("ell_integration", {})
            )
            
            # Update dashboard config if present
            if "visualization" in new_config:
                for dashboard_id in self.dashboard_manager.active_dashboards:
                    self.dashboard_manager.update_dashboard(
                        dashboard_id,
                        new_config["visualization"]
                    )
                    
            return {
                "ell_config": ell_config,
                "dashboards": self.dashboard_manager.list_dashboards()
            }
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            raise
            
    def get_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get a dashboard by ID.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            Dashboard data if found
        """
        return self.dashboard_manager.get_dashboard(dashboard_id)
        
    def create_custom_dashboard(
        self,
        dashboard_type: str,
        config: Dict[str, Any]
    ) -> str:
        """Create a custom dashboard.
        
        Args:
            dashboard_type: Type of dashboard to create
            config: Dashboard configuration
            
        Returns:
            Dashboard ID
        """
        return self.dashboard_manager.create_dashboard(
            dashboard_type,
            config
        )
        
    async def get_validation_results(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get validation results.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary containing validation results
        """
        try:
            # Get results from both systems
            ell_results = await self.ell_adapter.get_ell_validation_results(
                start_time=start_time,
                end_time=end_time
            )
            
            agentflow_results = self.ell_adapter.monitoring.system_monitor.get_validation_summary(
                start_time=start_time,
                end_time=end_time
            )
            
            return {
                "ell_results": ell_results,
                "agentflow_results": agentflow_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting validation results: {e}")
            raise
            
    async def sync_validation_results(
        self,
        results: List[Dict[str, Any]]
    ):
        """Synchronize validation results with ELL.
        
        Args:
            results: List of validation results
        """
        try:
            await self.ell_adapter.send_validation_results(results)
        except Exception as e:
            logger.error(f"Error syncing validation results: {e}")
            raise

# Mock classes for testing
class MockMetricsCollector:
    """Mock metrics collector for testing."""
    pass

class MockSystemMonitor:
    """Mock system monitor for testing."""
    def get_validation_summary(self, start_time=None, end_time=None):
        return []

class MockMonitoring:
    """Mock monitoring for testing."""
    def __init__(self):
        self.system_monitor = MockSystemMonitor()
        
    async def sync_health_status(self):
        return {"status": "healthy"}

class MockELLAdapter:
    """Mock ELL adapter for testing."""
    def __init__(self):
        self.monitoring = MockMonitoring()
        
    async def start(self):
        pass
        
    async def stop(self):
        pass
        
    def get_ell_config(self):
        return {}
        
    async def update_ell_config(self, config):
        return config
        
    async def get_ell_validation_results(self, start_time=None, end_time=None):
        return []
        
    async def send_validation_results(self, results):
        pass

class MockDashboardManager:
    """Mock dashboard manager for testing."""
    def __init__(self):
        self.active_dashboards = set()
        
    async def start(self):
        pass
        
    async def stop(self):
        pass
        
    def list_dashboards(self):
        return list(self.active_dashboards)
        
    def create_dashboard(self, dashboard_type, config=None):
        dashboard_id = f"{dashboard_type}_{len(self.active_dashboards)}"
        self.active_dashboards.add(dashboard_id)
        return dashboard_id
        
    def get_dashboard(self, dashboard_id):
        if dashboard_id in self.active_dashboards:
            return {"id": dashboard_id}
        return None
        
    def update_dashboard(self, dashboard_id, config):
        pass
