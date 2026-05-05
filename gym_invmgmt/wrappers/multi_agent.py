"""
MultiAgentWrapper — Converts single-agent env to per-node multi-agent setup.

Each managed node in the supply chain acts as an independent RL agent
with its own local observation and action. This enables testing
decentralized decision-making (CTDE — Centralized Training,
Decentralized Execution).

For CPU feasibility, uses parameter sharing — all agents share the
same policy network (Independent PPO with shared weights).

Inspired by: Kotecha & del Rio Chanona (2025), MAPPO for supply chains.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiAgentWrapper(gym.Wrapper):
    """Per-node multi-agent wrapper for supply chain environments.

    Converts the centralized (flat obs, flat action) environment into
    a simple multi-agent loop where each node sees its local state and
    produces its own action (order quantities for incoming supply links).

    For SB3 compatibility, this wrapper runs a round-robin loop internally:
    each node takes its sub-action, and the joint action is assembled
    before calling env.step(). The observation seen by the RL agent is the
    local observation of each node, shuffled through one at a time.

    However, for simplicity with SB3 (which requires a single-agent Gym API),
    this wrapper operates in "shared parameter" mode: it concatenates all
    per-node local observations into one observation vector, and expects
    the policy to output actions for all nodes simultaneously. The key
    difference from the flat centralized approach is that the observation
    is structured per-node (enabling shared weight architectures).

    Args:
        env: The base Gymnasium environment (CoreEnv or wrapped).
    """

    def __init__(self, env):
        super().__init__(env)
        base = self.unwrapped
        net = base.network

        # Identify agent nodes (managed nodes that place orders)
        self.agent_nodes = sorted(n for n in net.graph.nodes() if n not in net.market and n not in net.rawmat)
        self.n_agents = len(self.agent_nodes)

        # Per-agent info: which reorder links it controls
        self.agent_links = {}
        self.agent_local_dim = {}

        for node in self.agent_nodes:
            incoming = [(i, e) for i, e in enumerate(net.reorder_links) if e[1] == node]
            self.agent_links[node] = incoming
            # Local obs: on_hand(1) + in_transit(1) + backlog(1)
            #          + demand_vel(1) + norm_time(1) = 5 per agent
            self.agent_local_dim[node] = 5

        # Action: one per reorder link
        # Observation: per-agent local obs concatenated
        total_local_dim = sum(self.agent_local_dim.values())

        self.observation_space = spaces.Box(
            low=np.full(total_local_dim, -1e6, dtype=np.float64),
            high=np.full(total_local_dim, 1e6, dtype=np.float64),
            dtype=np.float64,
        )

        # Action space unchanged
        self.action_space = env.action_space

    def _build_local_obs(self):
        """Build per-node local observation vector."""
        base = self.unwrapped
        t = base.period
        net = base.network

        local_obs = []

        mu = base.demand_engine.get_current_mu(t)

        for node in self.agent_nodes:
            node_idx = net.node_map[node]
            on_hand = base.X[t, node_idx]

            # In-transit
            in_transit = 0.0
            for i, edge in self.agent_links[node]:
                in_transit += base.Y[t, i] if t <= base.num_periods else 0.0

            # Backlog
            backlog = 0.0
            if getattr(base, "backlog", True) and node in net.retail:
                for k in base.graph.successors(node):
                    if (node, k) in net.retail_map:
                        r_idx = net.retail_map[(node, k)]
                        if t > 0:
                            backlog += base.U[t - 1, r_idx]

            # Demand velocity
            if t > 0:
                lookback = min(3, t)
                demand_vel = float(np.mean(base.D[t - lookback : t]))
            else:
                demand_vel = mu

            norm_time = t / max(base.num_periods, 1)

            local_obs.extend([on_hand, in_transit, backlog, demand_vel, norm_time])

        return np.array(local_obs, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        local_obs = self._build_local_obs()
        return local_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        local_obs = self._build_local_obs()
        return local_obs, reward, terminated, truncated, info
