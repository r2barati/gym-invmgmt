"""
Lightweight episode logger for trajectory capture and post-hoc analysis.
"""

import os
import time

import gymnasium as gym
import numpy as np


class EpisodeLoggerWrapper(gym.Wrapper):
    """
    Lightweight episode logger that captures full trajectory matrices.

    At the end of each episode, saves the complete environment state history
    (inventory, pipeline, demand, orders, sales, profit, backlog) as a
    compressed .npz file for post-hoc analysis.

    This is useful for:
    - Analyzing bullwhip effects across the network
    - Visualizing inventory dynamics
    - Computing service levels and fill rates
    - Inspecting agent behavior

    Example::

        # Evaluation: save every episode (default)
        env = EpisodeLoggerWrapper(env, log_dir="./logs", run_name="eval")

        # Training: save every 100th episode to avoid I/O bottleneck
        env = EpisodeLoggerWrapper(env, log_dir="./logs", run_name="train", save_freq=100)

    Loading saved trajectories::

        data = np.load("logs/experiment_1_ep1_1234567890.npz")
        inventory = data['inventory_X']   # shape: (T+1, n_nodes)
        demand = data['demand_D']         # shape: (T, n_retail_links)
    """

    def __init__(self, env, log_dir="./data/logs", run_name="run", save_freq=1):
        super().__init__(env)
        self.log_dir = log_dir
        self.run_name = run_name
        self.save_freq = save_freq
        self.current_episode = 0
        self.episode_rewards = []

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def reset(self, **kwargs):
        self.current_episode += 1
        self.episode_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_rewards.append(reward)

        if (terminated or truncated) and self.current_episode % self.save_freq == 0:
            self._save_trajectory()

        return obs, reward, terminated, truncated, info

    def _save_trajectory(self):
        """Save full trajectory matrices from the core environment."""
        base_env = self.unwrapped

        trajectory = {
            "inventory_X": base_env.X,
            "pipeline_Y": base_env.Y,
            "orders_R": base_env.R,
            "sales_S": base_env.S,
            "demand_D": base_env.D,
            "unfulfilled_U": base_env.U,
            "profit_P": base_env.P,
            "actions": base_env.action_log,
            "goodwill_GW": base_env.GW,
            "episode_rewards": np.array(self.episode_rewards),
        }

        timestamp = int(time.time())
        filename = os.path.join(self.log_dir, f"{self.run_name}_ep{self.current_episode}_{timestamp}.npz")
        np.savez_compressed(filename, **trajectory)

    def close(self):
        super().close()
