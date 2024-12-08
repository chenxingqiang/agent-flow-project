"""Dashboard manager for AgentFlow visualization."""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
import json
import asyncio
from datetime import datetime, timedelta

from ..core.metrics import MetricsCollector
from ..integrations.ell_adapter import ELLAdapter
from .ell_visualizer import ELLVisualizer

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for dashboard management."""
    enabled: bool = True
    refresh_interval: int = 30
    default_timespan: str = "1h"
    max_datapoints: int = 1000

class DashboardManager:
    """Manager for AgentFlow dashboards."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        metrics_collector: MetricsCollector,
        ell_adapter: ELLAdapter
    ):
        """Initialize the dashboard manager.
        
        Args:
            config: Configuration dictionary
            metrics_collector: Metrics collector instance
            ell_adapter: ELL adapter instance
        """
        self.config = DashboardConfig(**config.get("visualization", {}))
        self.metrics_collector = metrics_collector
        self.ell_adapter = ell_adapter
        self.ell_visualizer = ELLVisualizer(
            ell_adapter.monitoring,
            metrics_collector
        )
        
        # Load dashboard templates
        self.templates = self._load_dashboard_templates()
        
        # Active dashboards
        self.active_dashboards: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.refresh_task = None
        self._shutdown = False
        
    def _load_dashboard_templates(self) -> Dict[str, Any]:
        """Load dashboard templates from configuration."""
        try:
            with open("templates/dashboard_templates.json") as f:
                return json.load(f).get("dashboards", {})
        except Exception as e:
            logger.error(f"Failed to load dashboard templates: {e}")
            return {}
            
    async def start(self):
        """Start the dashboard manager."""
        if self.config.enabled:
            self._shutdown = False
            self.refresh_task = asyncio.create_task(self._refresh_loop())
            logger.info("Dashboard manager started")
            
    async def stop(self):
        """Stop the dashboard manager."""
        self._shutdown = True
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("Dashboard manager stopped")
        
    async def _refresh_loop(self):
        """Background dashboard refresh loop."""
        while not self._shutdown:
            try:
                await self._refresh_dashboards()
                await asyncio.sleep(self.config.refresh_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}")
                await asyncio.sleep(self.config.refresh_interval)
                
    async def _refresh_dashboards(self):
        """Refresh all active dashboards."""
        for dashboard_id, dashboard in self.active_dashboards.items():
            try:
                await self._refresh_dashboard(dashboard_id)
            except Exception as e:
                logger.error(f"Error refreshing dashboard {dashboard_id}: {e}")
                
    async def _refresh_dashboard(self, dashboard_id: str):
        """Refresh a specific dashboard.
        
        Args:
            dashboard_id: ID of dashboard to refresh
        """
        dashboard = self.active_dashboards.get(dashboard_id)
        if not dashboard:
            return
            
        # Get timespan
        timespan = dashboard.get("timespan", self.config.default_timespan)
        end_time = datetime.now()
        if timespan.endswith("h"):
            start_time = end_time - timedelta(hours=int(timespan[:-1]))
        elif timespan.endswith("d"):
            start_time = end_time - timedelta(days=int(timespan[:-1]))
        else:
            start_time = end_time - timedelta(hours=1)
            
        # Update dashboard components
        if dashboard.get("type") == "system_overview":
            dashboard["data"] = await self._create_system_overview(
                start_time.timestamp(),
                end_time.timestamp()
            )
        elif dashboard.get("type") == "validation_dashboard":
            dashboard["data"] = await self._create_validation_dashboard(
                start_time.isoformat(),
                end_time.isoformat()
            )
            
    async def _create_system_overview(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Create system overview dashboard data.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Dashboard data dictionary
        """
        return {
            "metrics": self.ell_visualizer.create_combined_metrics_dashboard(
                start_time=start_time,
                end_time=end_time
            ),
            "health": self.ell_visualizer.create_health_status_view()
        }
        
    async def _create_validation_dashboard(
        self,
        start_time: str,
        end_time: str
    ) -> Dict[str, Any]:
        """Create validation dashboard data.
        
        Args:
            start_time: Start time ISO string
            end_time: End time ISO string
            
        Returns:
            Dashboard data dictionary
        """
        return {
            "summary": self.ell_visualizer.create_validation_summary(
                start_time=start_time,
                end_time=end_time
            ),
            "ell_results": await self.ell_adapter.get_ell_validation_results(
                start_time=start_time,
                end_time=end_time
            )
        }
        
    def create_dashboard(
        self,
        dashboard_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new dashboard.
        
        Args:
            dashboard_type: Type of dashboard to create
            config: Optional dashboard configuration
            
        Returns:
            Dashboard ID
        """
        if dashboard_type not in self.templates:
            raise ValueError(f"Unknown dashboard type: {dashboard_type}")
            
        # Create dashboard
        dashboard_id = f"{dashboard_type}_{len(self.active_dashboards)}"
        self.active_dashboards[dashboard_id] = {
            "type": dashboard_type,
            "config": config or {},
            "template": self.templates[dashboard_type],
            "data": {},
            "created_at": datetime.now().isoformat()
        }
        
        return dashboard_id
        
    def get_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get a dashboard by ID.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            Dashboard dictionary if found, None otherwise
        """
        return self.active_dashboards.get(dashboard_id)
        
    def update_dashboard(
        self,
        dashboard_id: str,
        config: Dict[str, Any]
    ) -> bool:
        """Update dashboard configuration.
        
        Args:
            dashboard_id: Dashboard ID
            config: New configuration
            
        Returns:
            True if updated, False otherwise
        """
        if dashboard_id not in self.active_dashboards:
            return False
            
        self.active_dashboards[dashboard_id]["config"].update(config)
        return True
        
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete a dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            True if deleted, False otherwise
        """
        return bool(self.active_dashboards.pop(dashboard_id, None))
        
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all active dashboards.
        
        Returns:
            List of dashboard summaries
        """
        return [
            {
                "id": dashboard_id,
                "type": dashboard["type"],
                "created_at": dashboard["created_at"]
            }
            for dashboard_id, dashboard in self.active_dashboards.items()
        ]
