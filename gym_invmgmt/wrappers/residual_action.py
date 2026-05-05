"""
ResidualActionWrapper — Hybrid heuristic + RL wrapper.

The RL agent learns a RESIDUAL correction on top of a base heuristic:
    final_action = clip(heuristic_action + rl_delta × max_residual, 0, cap)

The RL policy outputs ∈ [-1, 1], which are scaled by max_residual.
This starts training from heuristic-level performance (warm start).

Pitfall guards:
  - P1: Wrapper order = CoreEnv → DomainFeatureWrapper → ResidualAction → RescaleAction → IntegerAction
  - P10: Handles heuristic returning dict vs array
"""

import gymnasium as gym
import numpy as np


class ResidualActionWrapper(gym.Wrapper):
    """Combines a heuristic's base action with RL residual corrections.

    The RL agent sees the same observation space, but its actions are
    interpreted as adjustments to the heuristic's suggestion.

    Args:
        env: The base environment (after DomainFeatureWrapper).
        heuristic_fn: Callable(obs, period, env) → np.ndarray of base actions.
        max_residual: Maximum absolute correction per action dimension.
    """

    def __init__(self, env, heuristic_fn, max_residual=50.0):
        super().__init__(env)
        self.heuristic_fn = heuristic_fn
        self.max_residual = max_residual

        # The RL agent's action space is [-1, 1] for each dimension
        n_actions = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(
            low=-np.ones(n_actions, dtype=np.float64),
            high=np.ones(n_actions, dtype=np.float64),
            dtype=np.float64,
        )

        # Store the underlying action-space bounds for clipping
        self._orig_high = env.action_space.high.copy()
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, rl_action):
        """Combine heuristic + RL residual, then step the environment."""
        base = self.unwrapped
        period = base.period

        # P9: Terminal state guard
        if period >= base.num_periods:
            obs, reward, terminated, truncated, info = self.env.step(np.zeros(self.env.action_space.shape[0]))
            self._last_obs = obs
            return obs, reward, terminated, truncated, info

        # Get heuristic base action
        base_action = self.heuristic_fn(self._last_obs, period, base)

        # P10: Handle heuristic returning dict vs array
        if isinstance(base_action, dict):
            net = base.network
            arr = np.zeros(len(net.reorder_links))
            for key, val in base_action.items():
                if key in net.reorder_map:
                    arr[net.reorder_map[key]] = val
            base_action = arr

        base_action = np.asarray(base_action, dtype=np.float64)

        # Scale residual and combine
        rl_action_clipped = np.clip(rl_action, -1.0, 1.0)
        residual = rl_action_clipped * self.max_residual
        final_action = np.clip(base_action + residual, 0.0, self._orig_high)

        obs, reward, terminated, truncated, info = self.env.step(final_action)
        self._last_obs = obs
        info["base_action"] = base_action
        info["residual"] = residual
        return obs, reward, terminated, truncated, info


class ProportionalResidualWrapper(gym.Wrapper):
    """Apply RL actions as percentage deviations from a heuristic action.

    The policy outputs δ in [-max_pct, +max_pct] for each reorder link, and the
    wrapped environment receives ``base_action * (1 + δ)``. This is the residual
    interface used by the shipped SharedMLP checkpoints.
    """

    def __init__(self, env, heuristic_agent, max_pct=0.5, reward_lambda=0.001):
        super().__init__(env)
        self.heuristic_agent = heuristic_agent
        self.max_pct = max_pct
        self.reward_lambda = reward_lambda
        self._last_residual = None

        n_actions = len(self.unwrapped.network.reorder_links)
        self.action_space = gym.spaces.Box(
            low=-self.max_pct * np.ones(n_actions, dtype=np.float64),
            high=self.max_pct * np.ones(n_actions, dtype=np.float64),
            dtype=np.float64,
        )

    def step(self, action):
        core = self.unwrapped
        period = core.period

        if period >= core.num_periods:
            obs, reward, terminated, truncated, info = self.env.step(np.zeros(self.env.action_space.shape[0]))
            self._last_residual = np.zeros(len(core.network.reorder_links))
            return obs, reward, terminated, truncated, info

        core._update_state()
        base_actions = self.heuristic_agent.get_action(core.state, period)
        if isinstance(base_actions, dict):
            base_arr = np.zeros(len(core.network.reorder_links), dtype=np.float64)
            for edge, val in base_actions.items():
                if edge in core.network.reorder_map:
                    base_arr[core.network.reorder_map[edge]] = val
            base_actions = base_arr
        else:
            base_actions = np.asarray(base_actions, dtype=np.float64)

        delta = np.clip(np.asarray(action, dtype=np.float64), -self.max_pct, self.max_pct)
        final_action = np.clip(
            base_actions * (1.0 + delta),
            core.action_space.low,
            core.action_space.high,
        )
        self._last_residual = delta

        obs, reward, terminated, truncated, info = self.env.step(final_action)
        if self.reward_lambda > 0:
            reward = reward - self.reward_lambda * float(np.sum(delta**2))
        info["base_action"] = base_actions
        info["residual"] = delta
        return obs, reward, terminated, truncated, info
