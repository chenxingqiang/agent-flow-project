"""ELL adapter for AgentFlow integration."""

from typing import Dict, Any, List, Optional, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json

from ..core.metrics import MetricsCollector, MetricType
from ..core.monitoring import SystemMonitor
from .ell_integration import ELLMonitoringIntegration

logger = logging.getLogger(__name__)

@dataclass
class ELLConfig:
    """Configuration for ELL integration."""
    enabled: bool = True
    sync_interval: int = 60
    metrics_enabled: bool = True
    collection_interval: int = 30
    retention_days: int = 30
    monitoring_enabled: bool = True
    health_check_interval: int = 300

class ELLAdapter:
    """Adapter for ELL integration with AgentFlow."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the adapter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = ELLConfig(**config.get("ell_integration", {}))
        self.metrics_collector = MetricsCollector(config.get("persistence", {}))
        self.system_monitor = SystemMonitor(config)
        self.monitoring = ELLMonitoringIntegration(self.system_monitor)
        
        # Load metric mappings
        self.metric_mappings = self._load_metric_mappings()
        
        # Background tasks
        self.sync_task = None
        self._shutdown = False
        
    def _load_metric_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load metric mappings from configuration."""
        try:
            with open("templates/metric_mappings.json") as f:
                mappings = json.load(f)
            return mappings.get("metric_mappings", {})
        except Exception as e:
            logger.error(f"Failed to load metric mappings: {e}")
            return {}
            
    async def start(self):
        """Start the ELL adapter."""
        if self.config.enabled:
            self._shutdown = False
            self.sync_task = asyncio.create_task(self._sync_loop())
            logger.info("ELL adapter started")
            
    async def stop(self):
        """Stop the ELL adapter."""
        self._shutdown = True
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
        logger.info("ELL adapter stopped")
        
    async def _sync_loop(self):
        """Background synchronization loop."""
        while not self._shutdown:
            try:
                await self._sync_metrics()
                await self._sync_health_status()
                await asyncio.sleep(self.config.sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(self.config.sync_interval)
                
    async def _sync_metrics(self):
        """Synchronize metrics with ELL."""
        if not self.config.metrics_enabled:
            return
            
        try:
            # Get ELL metrics (placeholder for actual ELL API call)
            ell_metrics = self._get_ell_metrics()
            
            # Convert and store metrics
            for metric_name, metric_data in ell_metrics.items():
                if metric_name in self.metric_mappings.get("ell_to_agentflow", {}):
                    mapping = self.metric_mappings["ell_to_agentflow"][metric_name]
                    self.monitoring.metric_adapter.record_ell_metric(
                        metric_name=metric_name,
                        value=metric_data.get("value", 0) * mapping.get("unit_conversion", 1.0),
                        labels=metric_data.get("labels", {})
                    )
        except Exception as e:
            logger.error(f"Error syncing metrics: {e}")
            
    async def _sync_health_status(self):
        """Synchronize health status with ELL."""
        if not self.config.monitoring_enabled:
            return
            
        try:
            # Get combined health status
            status = await self.monitoring.sync_health_status()
            
            # Update ELL health status (placeholder for actual ELL API call)
            self._update_ell_health_status(status)
        except Exception as e:
            logger.error(f"Error syncing health status: {e}")
            
    def _get_ell_metrics(self) -> Dict[str, Any]:
        """Get metrics from ELL (placeholder).
        
        Returns:
            Dictionary of ELL metrics
        """
        # TODO: Implement actual ELL API call
        return {
            "response_time": {
                "value": 100,
                "labels": {"service": "api"}
            },
            "memory_usage": {
                "value": 512,
                "labels": {"service": "worker"}
            }
        }
        
    def _update_ell_health_status(self, status: Dict[str, Any]):
        """Update health status in ELL (placeholder).
        
        Args:
            status: Health status data
        """
        # TODO: Implement actual ELL API call
        logger.info(f"Would update ELL health status: {status}")
        
    async def get_ell_validation_results(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get validation results from ELL.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of validation results
        """
        # TODO: Implement actual ELL API call
        return []
        
    async def send_validation_results(
        self,
        results: List[Dict[str, Any]]
    ):
        """Send validation results to ELL.
        
        Args:
            results: List of validation results to send
        """
        # TODO: Implement actual ELL API call
        logger.info(f"Would send validation results to ELL: {len(results)} results")
        
    def get_ell_config(self) -> Dict[str, Any]:
        """Get current ELL configuration.
        
        Returns:
            Dictionary of ELL configuration
        """
        return {
            "enabled": self.config.enabled,
            "sync_interval": self.config.sync_interval,
            "metrics": {
                "enabled": self.config.metrics_enabled,
                "collection_interval": self.config.collection_interval,
                "retention_days": self.config.retention_days
            },
            "monitoring": {
                "enabled": self.config.monitoring_enabled,
                "health_check_interval": self.config.health_check_interval
            }
        }
        
    async def update_ell_config(
        self,
        new_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update ELL configuration.
        
        Args:
            new_config: New configuration values
            
        Returns:
            Updated configuration
        """
        # Update configuration
        self.config = ELLConfig(**{
            **self.config.__dict__,
            **new_config
        })
        
        # Restart if running
        if self.sync_task:
            await self.stop()
            await self.start()
            
        return self.get_ell_config()
