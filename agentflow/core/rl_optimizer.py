"""Reinforcement Learning based optimization using Ray RLlib"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace, Discrete
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

class InstructionOptEnv(gym.Env):
    """Environment for instruction optimization using RL"""
    
    def __init__(self, config: EnvContext):
        self.max_steps = config.get("max_steps", 100)
        self.current_step = 0
        
        # Define action space (instruction parameters)
        self.action_space = Box(
            low=np.array([1, 1, 100, 1.0]),
            high=np.array([64, 8, 10000, 30.0]),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0] + [0]*4),
            high=np.array([1, 100, 1, 1] + [1]*4),
            dtype=np.float32
        )
        
        self.state = self._get_initial_state()
        self.history = []
    
    def _get_initial_state(self) -> np.ndarray:
        """Get initial environment state"""
        return np.array([
            0.5,    # load
            50.0,   # latency
            0.9,    # success_rate
            0.5,    # cache_hit_rate
            0.5, 0.5, 0.5, 0.5  # resource_usage
        ], dtype=np.float32)
    
    def _calculate_reward(self, action: np.ndarray,
                         new_state: np.ndarray) -> float:
        """Calculate reward based on action and resulting state"""
        # Performance metrics
        latency_score = 1.0 / (1.0 + new_state[1])  # latency
        success_score = new_state[2]  # success_rate
        cache_score = new_state[3]  # cache_hit_rate
        
        # Resource efficiency
        resource_usage = new_state[4:]  # resource_usage
        resource_efficiency = np.mean(resource_usage)
        resource_balance = 1.0 - np.std(resource_usage)
        
        # Combine metrics with weights
        reward = (
            0.3 * latency_score +
            0.3 * success_score +
            0.2 * cache_score +
            0.1 * resource_efficiency +
            0.1 * resource_balance
        )
        
        return float(reward)
    
    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """Apply action and get new state"""
        # Extract action components
        batch_size = action[0]
        num_parallel = action[1] 
        cache_size = action[2]
        timeout = action[3]
        
        # Simulate system response to action
        new_state = np.zeros(8, dtype=np.float32)
        
        # Update each state component
        new_state[0] = min(1.0, self.state[0] * (1.0 + np.random.normal(0, 0.1)))  # load
        new_state[1] = max(1.0, self.state[1] * (  # latency
            1.0 - 0.1 * batch_size / 64.0 +
            0.05 * self.state[0]
        ))
        new_state[2] = max(0.5, min(1.0, self.state[2] * (  # success_rate
            1.0 + 0.05 * timeout / 30.0 -
            0.1 * self.state[0]
        )))
        new_state[3] = max(0.0, min(1.0, self.state[3] * (  # cache_hit_rate
            1.0 + 0.1 * cache_size / 10000.0
        )))
        
        # Update resource usage
        for i in range(4):
            new_state[4+i] = max(0.0, min(1.0, self.state[4+i] * (
                1.0 + np.random.normal(0, 0.1)
            )))
        
        return new_state
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.state = self._get_initial_state()
        return self.state, {}
    
    def step(self, action: np.ndarray) -> Tuple[
        np.ndarray, float, bool, bool, Dict[str, Any]
    ]:
        """Execute one environment step"""
        self.current_step += 1
        
        # Apply action and get new state
        new_state = self._apply_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(action, new_state)
        
        # Update state
        self.state = new_state
        
        # Record history
        self.history.append({
            "step": self.current_step,
            "action": action.tolist(),
            "state": new_state.tolist(),
            "reward": reward
        })
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        return new_state, reward, done, False, {}
    
    def render(self) -> None:
        """Render environment"""
        pass
    
    def close(self) -> None:
        """Close environment"""
        pass

class DictInputNetwork(TorchModelV2, nn.Module):
    """Custom model for dictionary observation spaces"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Get input dimensions from observation space
        if isinstance(obs_space, Box):
            self.input_dim = int(np.prod(obs_space.shape))
        else:
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

        # Create shared layers with appropriate dimensions
        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Policy branch
        self.policy_branch = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )

        # Value branch
        self._value_branch = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._features = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        if isinstance(obs, dict):
            # This shouldn't happen since we expect a flattened Box space
            raise ValueError("Unexpected dictionary observation")
        
        # Pass through shared layers
        self._features = self.shared_layers(obs.float())
        
        # Get policy output
        logits = self.policy_branch(self._features)
        
        return logits, state

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)

class RLOptimizer:
    """Base class for reinforcement learning optimizers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RL optimizer.
        
        Args:
            config: Configuration dictionary containing optimizer settings
        """
        self.config = config
        self._initialized = False
        
        # Register the environment with Ray
        tune.register_env(
            "InstructionOpt-v0",
            lambda config: InstructionOptEnv(config)
        )
        
        # Create PPO config and disable new API stack
        ppo_config = (
            PPOConfig()
            .environment("InstructionOpt-v0")
            .framework("torch")
            .training(
                model={
                    "custom_model": DictInputNetwork,
                    "vf_share_layers": True,
                    "fcnet_activation": "relu",
                    "fcnet_hiddens": [512, 256, 128]
                }
            )
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .resources(num_gpus=0)
            .env_runners(num_env_runners=4)
            .training(train_batch_size=4000)
        )
        
        # Initialize PPO trainer
        self.trainer = ppo_config.build()
    
    async def initialize(self):
        """Initialize the optimizer asynchronously."""
        if self._initialized:
            return self
        
        if not ray.is_initialized():
            ray.init()
        
        self._initialized = True
        return self
    
    def train(self, num_iterations: int = 100):
        """Train the RL agent"""
        for i in range(num_iterations):
            result = self.trainer.train()
            if i % 10 == 0:
                logger.info(
                    f"Iteration {i}: "
                    f"episode_reward_mean={result['episode_reward_mean']:.2f}, "
                    f"episode_reward_min={result['episode_reward_min']:.2f}, "
                    f"episode_reward_max={result['episode_reward_max']:.2f}"
                )
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get optimized action for given state"""
        action = self.trainer.compute_single_action(state)
        
        # Convert action to appropriate format
        action = np.array([
            int(action[0]),  # batch_size
            int(action[1]),  # num_parallel
            int(action[2]),  # cache_size
            float(action[3])  # timeout
        ])
        
        return action
    
    def save_model(self, path: str):
        """Save trained model"""
        self.trainer.save(path)
    
    def load_model(self, path: str):
        """Load trained model"""
        self.trainer.restore(path)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            "total_steps": self.trainer.get_policy().global_timestep,
            "episodes": len(self.trainer.workers.local_worker().env.history)
        }
        
    async def cleanup(self):
        """Clean up optimizer state"""
        if self.trainer:
            self.trainer.stop()
            self.trainer = None
        self._initialized = False
        return self

class SAC_Optimizer(RLOptimizer):
    """Soft Actor-Critic based optimizer"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SAC optimizer.
        
        Args:
            config: Configuration dictionary containing optimizer settings
        """
        self.config = config
        self._initialized = False
        
        # Register the environment with Ray
        tune.register_env(
            "InstructionOpt-v0",
            lambda config: InstructionOptEnv(config)
        )
        
        # Create SAC config and disable new API stack
        sac_config = (
            SACConfig()
            .environment("InstructionOpt-v0")
            .framework("torch")
            .training(
                model={
                    "custom_model": DictInputNetwork,
                    "vf_share_layers": True,
                    "fcnet_activation": "relu",
                    "fcnet_hiddens": [512, 256, 128]
                }
            )
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .resources(num_gpus=0)
            .env_runners(num_env_runners=4)
            .training(
                train_batch_size=4000,
                learning_starts=10000,
                buffer_size=500000,
                twin_q=True,
                prioritized_replay=True
            )
        )
        
        # Initialize SAC trainer
        self.trainer = sac_config.build()
    
    async def initialize(self):
        """Initialize the optimizer asynchronously."""
        if self._initialized:
            return self
            
        # Call parent initialize
        await super().initialize()
        
        if not ray.is_initialized():
            ray.init()
            
        self._initialized = True
        return self
    
    def train(self, num_iterations: int = 100):
        """Train the RL agent"""
        for i in range(num_iterations):
            result = self.trainer.train()
            
            # Log training progress
            logger.info(
                f"Iteration {i}: "
                f"episode_reward_mean={result['episode_reward_mean']:.2f}, "
                f"episode_reward_max={result['episode_reward_max']:.2f}"
            )
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get optimized action for given state"""
        action = self.trainer.compute_single_action(state)
        
        # Convert action to proper format
        action = np.array([
            int(action[0]),  # batch_size
            int(action[1]),  # num_parallel
            int(action[2]),  # cache_size
            float(action[3])  # timeout
        ])
        
        return action
    
    def save_model(self, path: str):
        """Save trained model"""
        self.trainer.save(path)
    
    def load_model(self, path: str):
        """Load trained model"""
        self.trainer.restore(path)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            "total_steps": self.trainer.get_policy().global_timestep,
            "episodes": len(self.trainer.workers.local_worker().env.history)
        }
        
    async def cleanup(self):
        """Clean up optimizer state"""
        if self.trainer:
            self.trainer.stop()
            self.trainer = None
        self._initialized = False
        return self

def create_optimizer(config: Optional[Union[Dict[str, Any], 'AgentConfig']] = None) -> Optional[RLOptimizer]:
    """Create an RL optimizer based on configuration.
    
    Args:
        config: Configuration dictionary or AgentConfig object
        
    Returns:
        RLOptimizer or None if no algorithm specified
    """
    if config is None:
        return None
        
    # Get algorithm type
    if isinstance(config, dict):
        algorithm = config.get('algorithm')
    else:
        algorithm = getattr(config, 'algorithm', None)
        
    if algorithm is None:
        return None
        
    if algorithm.upper() == 'PPO':
        return RLOptimizer(config)
    elif algorithm.upper() == 'SAC':
        return SAC_Optimizer(config)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
