"""ELL visualization integration for AgentFlow."""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from ..core.metrics import MetricsCollector
from ..integrations.ell_integration import ELLMonitoringIntegration

logger = logging.getLogger(__name__)

class ELLVisualizer:
    """Visualization component for ELL metrics and monitoring data."""
    
    def __init__(
        self,
        ell_integration: ELLMonitoringIntegration,
        metrics_collector: MetricsCollector
    ):
        """Initialize the visualizer.
        
        Args:
            ell_integration: ELL monitoring integration
            metrics_collector: AgentFlow metrics collector
        """
        self.ell_integration = ell_integration
        self.metrics_collector = metrics_collector
        
    def create_combined_metrics_dashboard(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create a dashboard combining ELL and AgentFlow metrics.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary containing dashboard components
        """
        # Get metrics data
        metrics_data = self.metrics_collector.get_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Convert to DataFrame for easier plotting
        df_rows = []
        for metric_key, points in metrics_data.items():
            for point in points:
                row = {
                    "metric_type": point.metric_type.value,
                    "value": point.value,
                    "timestamp": datetime.fromtimestamp(point.timestamp),
                    "source": point.labels.get("source", "agentflow"),
                    **point.labels
                }
                df_rows.append(row)
                
        if not df_rows:
            return {"error": "No metrics data available"}
            
        df = pd.DataFrame(df_rows)
        
        # Create time series plot
        fig_timeseries = px.line(
            df,
            x="timestamp",
            y="value",
            color="metric_type",
            facet_col="source",
            title="Metrics Over Time by Source"
        )
        
        # Create distribution plot
        fig_dist = px.box(
            df,
            x="metric_type",
            y="value",
            color="source",
            title="Metric Distributions by Source"
        )
        
        # Create heatmap of correlations
        pivot_df = df.pivot_table(
            index="timestamp",
            columns=["source", "metric_type"],
            values="value",
            aggfunc="mean"
        )
        fig_heatmap = px.imshow(
            pivot_df.corr(),
            title="Metric Correlations"
        )
        
        return {
            "timeseries": fig_timeseries,
            "distributions": fig_dist,
            "correlations": fig_heatmap,
            "raw_data": df.to_dict(orient="records")
        }
        
    def create_health_status_view(self) -> Dict[str, Any]:
        """Create a view of the current health status.
        
        Returns:
            Dictionary containing health status visualization
        """
        # Get current health metrics
        health_metrics = self.metrics_collector.get_metrics(
            metric_type=MetricType.SUCCESS_RATE
        )
        
        # Create gauge chart for overall health
        if health_metrics:
            latest_health = max(
                (p.value for points in health_metrics.values() for p in points),
                default=0
            )
        else:
            latest_health = 0
            
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_health * 100,
            title={"text": "System Health"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkgreen"},
                "steps": [
                    {"range": [0, 50], "color": "red"},
                    {"range": [50, 80], "color": "yellow"},
                    {"range": [80, 100], "color": "green"}
                ]
            }
        ))
        
        return {
            "gauge": fig_gauge,
            "health_score": latest_health
        }
        
    def create_validation_summary(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a summary of validation results.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary containing validation summary visualization
        """
        # Get validation summary
        summary = self.ell_integration.system_monitor.get_validation_summary(
            start_time=start_time,
            end_time=end_time
        )
        
        # Create success rate trend
        validation_df = pd.DataFrame(summary["validations"])
        validation_df["timestamp"] = pd.to_datetime(validation_df["timestamp"])
        validation_df["is_valid"] = validation_df["result"].apply(
            lambda x: x.get("is_valid", False)
        )
        
        fig_trend = px.line(
            validation_df.groupby("timestamp")["is_valid"].mean().reset_index(),
            x="timestamp",
            y="is_valid",
            title="Validation Success Rate Over Time"
        )
        
        # Create validation type distribution
        validation_types = validation_df["result"].apply(
            lambda x: x.get("validation_type", "unknown")
        ).value_counts()
        
        fig_types = px.pie(
            values=validation_types.values,
            names=validation_types.index,
            title="Validation Types Distribution"
        )
        
        return {
            "trend": fig_trend,
            "type_distribution": fig_types,
            "summary_stats": {
                "total": summary["total"],
                "success_rate": summary["success_rate"]
            }
        }
