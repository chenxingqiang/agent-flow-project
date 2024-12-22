"""Federated learning optimization strategies."""
from typing import Dict, Any, List, Optional
import numpy as np
from ..base import BaseOptimizer
from ..utils.metrics import MetricsCollector

class FederatedOptimizer(BaseOptimizer):
    """Federated learning optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metrics = MetricsCollector()
        self.clients = []
        self.aggregation_strategy = config.get("aggregation", "fedavg")
        
    def add_client(self, client_id: str, config: Dict[str, Any]):
        """Add a federated learning client."""
        self.clients.append({
            "id": client_id,
            "config": config,
            "model": None,
            "metrics": MetricsCollector()
        })
    
    def train_round(self, global_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one round of federated training."""
        # Distribute global model to clients
        client_results = []
        for client in self.clients:
            local_result = self._train_client(client, global_params)
            client_results.append(local_result)
        
        # Aggregate results using selected strategy
        if self.aggregation_strategy == "fedavg":
            return self._fedavg_aggregate(client_results)
        else:
            return self._weighted_aggregate(client_results)
    
    def _train_client(self, client: Dict[str, Any], 
                     global_params: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single client with local data."""
        # Simulate local training
        local_updates = {}
        for key, value in global_params.items():
            noise = np.random.normal(0, 0.1, value.shape)
            local_updates[key] = value + noise
        return local_updates
    
    def _fedavg_aggregate(self, 
                         client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Federated averaging aggregation."""
        aggregated = {}
        for key in client_results[0].keys():
            aggregated[key] = np.mean(
                [r[key] for r in client_results], axis=0
            )
        return aggregated
    
    def _weighted_aggregate(self, 
                          client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted aggregation based on client metrics."""
        weights = self._compute_client_weights()
        aggregated = {}
        for key in client_results[0].keys():
            weighted_sum = np.sum(
                [w * r[key] for w, r in zip(weights, client_results)], 
                axis=0
            )
            aggregated[key] = weighted_sum / np.sum(weights)
        return aggregated
    
    def _compute_client_weights(self) -> np.ndarray:
        """Compute weights for client aggregation."""
        metrics = [c["metrics"].get_metrics() for c in self.clients]
        weights = np.array([m.get("performance", 1.0) for m in metrics])
        return weights / np.sum(weights)
