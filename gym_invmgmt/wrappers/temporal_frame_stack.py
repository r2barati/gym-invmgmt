"""
temporal_frame_stack.py — Temporal Frame Stack Wrapper for ST-PPO.

Maintains a sliding window of the last `n_history` augmented observations,
enabling the Transformer extractor to perform true spatio-temporal
self-attention (across both nodes AND time steps).

Must be applied AFTER DomainFeatureWrapper and BEFORE RescaleAction:
  CoreEnv → DomainFeatureWrapper → TemporalFrameStack → RescaleAction
"""

from collections import deque

import gymnasium as gym
import numpy as np


class TemporalFrameStack(gym.Wrapper):
    """
    Stacks the last `n_history` observations into a single flat vector.

    On reset(), the buffer is filled with copies of the initial observation.
    On step(), the new observation is appended and the oldest is dropped.

    The output observation is: [obs_t-(n-1) | obs_t-(n-2) | ... | obs_t]
    where each obs_t-k has the same dimension as the wrapped env's obs.

    Parameters
    ----------
    env : gym.Env
        The wrapped environment (should already have augmented observations).
    n_history : int
        Number of frames to stack (including the current one).
    """

    def __init__(self, env, n_history: int = 4):
        super().__init__(env)
        self.n_history = n_history
        self._buffer = deque(maxlen=n_history)

        # Update observation space to reflect the stacked dimension
        obs_dim = env.observation_space.shape[0]
        self.single_obs_dim = obs_dim
        stacked_dim = obs_dim * n_history

        self.observation_space = gym.spaces.Box(
            low=np.full(stacked_dim, -np.inf, dtype=np.float64),
            high=np.full(stacked_dim, np.inf, dtype=np.float64),
            dtype=np.float64,
        )

    def _get_stacked_obs(self) -> np.ndarray:
        """Concatenate buffer frames into a single flat vector."""
        return np.concatenate(list(self._buffer), axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill buffer with copies of the initial observation
        self._buffer.clear()
        for _ in range(self.n_history):
            self._buffer.append(obs.copy())
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(obs.copy())
        return self._get_stacked_obs(), reward, terminated, truncated, info
