"""Advanced optimization strategies for instruction execution"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
import optuna
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

logger = logging.getLogger(__name__)

@dataclass
class OptimizationObjective:
    """Definition of an optimization objective"""
    name: str
    weight: float
    minimize: bool
    target: Optional[float] = None
    threshold: Optional[float] = None

class MultiObjectiveOptimizer:
    """Multi-objective optimization for instruction execution"""
    
    def __init__(self, objectives: List[OptimizationObjective]):
        self.objectives = objectives
        self.history = []
        self.pareto_front = []
    
    def _normalize_objective(self, value: float, objective: OptimizationObjective) -> float:
        """Normalize objective value to [0, 1] range"""
        if not self.history:
            return value
        
        values = [h[objective.name] for h in self.history]
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return 0.5
        
        return (value - min_val) / (max_val - min_val)
    
    def _calculate_weighted_sum(self, values: Dict[str, float]) -> float:
        """Calculate weighted sum of objectives"""
        total = 0.0
        for obj in self.objectives:
            norm_value = self._normalize_objective(values[obj.name], obj)
            if obj.minimize:
                norm_value = 1 - norm_value
            total += obj.weight * norm_value
        return total
    
    def _is_pareto_efficient(self, values: Dict[str, float]) -> bool:
        """Check if solution is Pareto efficient"""
        for front_values in self.pareto_front:
            dominates = True
            for obj in self.objectives:
                curr_val = values[obj.name]
                front_val = front_values[obj.name]
                if obj.minimize:
                    if curr_val > front_val:
                        dominates = False
                        break
                else:
                    if curr_val < front_val:
                        dominates = False
                        break
            if dominates:
                return False
        return True
    
    def update(self, values: Dict[str, float]):
        """Update optimization history and Pareto front"""
        self.history.append(values)
        
        # Update Pareto front
        if self._is_pareto_efficient(values):
            self.pareto_front.append(values)
            
            # Remove dominated solutions
            self.pareto_front = [
                front_values for front_values in self.pareto_front
                if self._is_pareto_efficient(front_values)
            ]
    
    def get_best_solution(self) -> Optional[Dict[str, float]]:
        """Get best solution based on weighted sum"""
        if not self.pareto_front:
            return None
        
        return max(
            self.pareto_front,
            key=self._calculate_weighted_sum
        )

class HyperparameterOptimizer:
    """Automatic hyperparameter optimization"""
    
    def __init__(self, param_space: Dict[str, Any],
                 objective_fn: Callable[[Dict[str, Any]], float],
                 max_trials: int = 50):
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.max_trials = max_trials
        self.study = optuna.create_study(direction="maximize")
        self.best_params = None
        
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function"""
        params = {}
        for name, space in self.param_space.items():
            if isinstance(space, tuple):
                if isinstance(space[0], int):
                    params[name] = trial.suggest_int(name, space[0], space[1])
                else:
                    params[name] = trial.suggest_float(name, space[0], space[1])
            elif isinstance(space, list):
                params[name] = trial.suggest_categorical(name, space)
        
        return self.objective_fn(params)
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        self.study.optimize(self._objective, n_trials=self.max_trials)
        self.best_params = self.study.best_params
        return self.best_params
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        return [
            {
                "trial": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state
            }
            for trial in self.study.trials
        ]

@ray.remote
class ResourceOptimizer:
    """Dynamic resource allocation optimization"""
    
    def __init__(self, resources: Dict[str, float],
                 min_allocation: Dict[str, float]):
        self.total_resources = resources
        self.min_allocation = min_allocation
        self.current_allocation = dict(min_allocation)
        self.usage_history = []
    
    def _calculate_efficiency(self, usage: Dict[str, float],
                            allocation: Dict[str, float]) -> float:
        """Calculate resource usage efficiency"""
        efficiencies = []
        for resource, used in usage.items():
            allocated = allocation[resource]
            if allocated > 0:
                efficiency = used / allocated
                efficiencies.append(min(1.0, efficiency))
        return np.mean(efficiencies) if efficiencies else 0.0
    
    def update_usage(self, usage: Dict[str, float]):
        """Update resource usage history"""
        self.usage_history.append(usage)
        if len(self.usage_history) > 100:
            self.usage_history.pop(0)
    
    def optimize_allocation(self) -> Dict[str, float]:
        """Optimize resource allocation based on usage history"""
        if not self.usage_history:
            return self.current_allocation
        
        # Calculate average usage for each resource
        avg_usage = {}
        for resource in self.total_resources:
            usage_values = [h.get(resource, 0.0) for h in self.usage_history]
            avg_usage[resource] = np.mean(usage_values)
        
        # Calculate new allocation
        new_allocation = {}
        remaining_resources = dict(self.total_resources)
        
        # First, ensure minimum allocation
        for resource, min_value in self.min_allocation.items():
            new_allocation[resource] = min_value
            remaining_resources[resource] -= min_value
        
        # Then, allocate based on usage patterns
        total_usage = sum(avg_usage.values())
        if total_usage > 0:
            for resource, usage in avg_usage.items():
                usage_ratio = usage / total_usage
                additional = remaining_resources[resource] * usage_ratio
                new_allocation[resource] += additional
        
        self.current_allocation = new_allocation
        return new_allocation
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get resource allocation statistics"""
        if not self.usage_history:
            return {}
        
        recent_history = self.usage_history[-20:]
        stats = {
            "current_allocation": self.current_allocation,
            "average_usage": {},
            "efficiency": {},
            "waste": {}
        }
        
        for resource in self.total_resources:
            usage_values = [h.get(resource, 0.0) for h in recent_history]
            avg_usage = np.mean(usage_values)
            stats["average_usage"][resource] = avg_usage
            
            allocation = self.current_allocation[resource]
            efficiency = avg_usage / allocation if allocation > 0 else 0.0
            stats["efficiency"][resource] = efficiency
            
            waste = max(0.0, allocation - avg_usage)
            stats["waste"][resource] = waste
        
        return stats
