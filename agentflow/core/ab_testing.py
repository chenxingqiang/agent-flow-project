"""Advanced A/B testing with contextual bandits"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
import numpy as np
from dataclasses import dataclass
from scipy.stats import beta
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

@dataclass
class TestVariant:
    """A/B test variant definition"""
    name: str
    config: Dict[str, Any]
    context_features: List[str]
    min_samples: int = 100
    confidence_threshold: float = 0.95

class ContextualBandit:
    """Contextual multi-armed bandit for A/B testing"""
    
    def __init__(self):
        self.variants: Dict[str, TestVariant] = {}
        self.history: List[Dict[str, Any]] = []
        self.models: Dict[str, RandomForestRegressor] = {}
        self.scaler = StandardScaler()
        self.exploration_rate = 0.1  # Initial exploration rate
    
    def add_variant(self, variant: TestVariant):
        """Add a new variant for testing"""
        self.variants[variant.name] = variant
        self.models[variant.name] = RandomForestRegressor()
    
    def _extract_features(self, context: Dict[str, Any],
                         variant: TestVariant) -> np.ndarray:
        """Extract features from context"""
        features = []
        for feature in variant.context_features:
            value = context.get(feature, 0.0)
            if isinstance(value, (int, float)):
                features.append(value)
            elif isinstance(value, str):
                # Simple one-hot encoding
                features.append(hash(value) % 2)
            else:
                features.append(0.0)
        return np.array(features).reshape(1, -1)
    
    def _update_model(self, variant_name: str):
        """Update prediction model for variant"""
        variant = self.variants[variant_name]
        relevant_history = [
            h for h in self.history
            if h["variant"] == variant_name
        ]
        
        if len(relevant_history) < variant.min_samples:
            return
        
        # Prepare training data
        X = []
        y = []
        for h in relevant_history:
            features = self._extract_features(h["context"], variant)
            X.append(features[0])
            y.append(h["reward"])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) > 0:
            # Scale features and train model
            X_scaled = self.scaler.fit_transform(X)
            self.models[variant_name].fit(X_scaled, y)
    
    def _predict_reward(self, variant_name: str,
                       context: Dict[str, Any]) -> Tuple[float, float]:
        """Predict reward and uncertainty for variant"""
        variant = self.variants[variant_name]
        model = self.models[variant_name]
        
        features = self._extract_features(context, variant)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from all trees
        predictions = []
        for tree in model.estimators_:
            pred = tree.predict(features_scaled)
            predictions.append(pred[0])
        
        # Calculate mean and uncertainty
        mean_reward = np.mean(predictions)
        uncertainty = np.std(predictions)
        
        return mean_reward, uncertainty
    
    def select_variant(self, context: Dict[str, Any]) -> str:
        """Select best variant using Thompson sampling"""
        if not self.variants:
            raise ValueError("No variants registered")
        
        # Calculate scores for each variant
        scores = {}
        for name, variant in self.variants.items():
            # Get relevant history
            relevant_history = [
                h for h in self.history
                if h["variant"] == name
            ]
            
            if len(relevant_history) < variant.min_samples:
                # Not enough samples, use Thompson sampling
                successes = sum(1 for h in relevant_history if h["reward"] > 0.5)
                trials = len(relevant_history)
                score = beta.rvs(successes + 1, trials - successes + 1)
            else:
                # Use contextual model
                predicted_reward, uncertainty = self._predict_reward(name, context)
                exploration_bonus = self.exploration_rate * uncertainty
                score = predicted_reward + exploration_bonus
            
            scores[name] = score
        
        # Select variant with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def update(self, variant_name: str, context: Dict[str, Any],
               reward: float):
        """Update history and models"""
        self.history.append({
            "variant": variant_name,
            "context": context,
            "reward": reward,
            "timestamp": time.time()
        })
        
        # Update model for the variant
        self._update_model(variant_name)
        
        # Update exploration rate
        self.exploration_rate *= 0.995  # Gradually reduce exploration
    
    def get_test_results(self) -> Dict[str, Any]:
        """Get comprehensive test results"""
        results = {}
        
        for name, variant in self.variants.items():
            relevant_history = [
                h for h in self.history
                if h["variant"] == name
            ]
            
            if not relevant_history:
                continue
            
            # Calculate statistics
            rewards = [h["reward"] for h in relevant_history]
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            
            # Calculate confidence interval
            confidence_interval = None
            if len(rewards) >= variant.min_samples:
                confidence_interval = (
                    mean_reward - 1.96 * std_reward / np.sqrt(len(rewards)),
                    mean_reward + 1.96 * std_reward / np.sqrt(len(rewards))
                )
            
            results[name] = {
                "samples": len(relevant_history),
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "confidence_interval": confidence_interval,
                "is_significant": (
                    confidence_interval is not None and
                    confidence_interval[0] > 0.5  # Assuming 0.5 is baseline
                )
            }
        
        return results
    
    def should_stop_test(self) -> Tuple[bool, Optional[str]]:
        """Check if test should be stopped"""
        results = self.get_test_results()
        
        # Check if any variant is significantly better
        significant_variants = [
            name for name, stats in results.items()
            if stats["is_significant"]
        ]
        
        if significant_variants:
            # Find best among significant variants
            best_variant = max(
                significant_variants,
                key=lambda x: results[x]["mean_reward"]
            )
            return True, best_variant
        
        # Check if we have enough samples for all variants
        all_sufficient_samples = all(
            len([h for h in self.history if h["variant"] == name]) >= variant.min_samples
            for name, variant in self.variants.items()
        )
        
        if all_sufficient_samples:
            # If no significant difference found after sufficient sampling
            best_variant = max(
                results.items(),
                key=lambda x: x[1]["mean_reward"]
            )[0]
            return True, best_variant
        
        return False, None
