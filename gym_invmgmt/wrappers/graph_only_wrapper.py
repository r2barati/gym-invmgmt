"""
graph_only_wrapper.py — Topology-Invariant Graph Feature Wrapper

Produces an observation containing ONLY graph-structured features:
  Per-node (×n_main): inv_pos, lt_target, gap   (3 features each)
  Global:             demand_vel, norm_time      (2 features)

Total obs dim = 3 * n_main + 2

This makes the observation structure topology-invariant: the per-node
features have the same semantic meaning regardless of network size,
enabling zero-shot transfer between topologies.
"""

import gymnasium as gym
import numpy as np


class GraphOnlyWrapper(gym.Wrapper):
    """
    Wrapper that produces graph-structured observations for topology-invariant
    GNN processing.  Drops the flat base_obs entirely.

    Observation layout:
        [inv_pos_0, ..., inv_pos_n,
         lt_target_0, ..., lt_target_n,
         gap_0, ..., gap_n,
         demand_vel, norm_time]

    where n = n_main (number of non-market, non-rawmat nodes).
    """

    def __init__(self, env, is_blind=False):
        super().__init__(env)
        self.is_blind = is_blind

        # --- Pre-compute static topology info ---
        core = self.unwrapped
        net = core.network
        self.main_nodes = sorted([n for n in net.graph.nodes() if n not in net.market and n not in net.rawmat])
        self.n_main = len(self.main_nodes)

        # Per-node: incoming edges + max lead time
        self.node_edges = {}
        for node in self.main_nodes:
            incoming = [(i, e) for i, e in enumerate(net.reorder_links) if e[1] == node]
            max_L = max([net.graph.edges[e]["L"] for _, e in incoming], default=0)
            self.node_edges[node] = {"incoming": incoming, "max_L": max_L}

        # Observation space: 3 features per node + 2 global
        obs_dim = 3 * self.n_main + 2
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float64),
            high=np.full(obs_dim, np.inf, dtype=np.float64),
            dtype=np.float64,
        )

    def _estimate_mu(self, t):
        core = self.unwrapped
        if not self.is_blind:
            return core.demand_engine.get_current_mu(t)
        else:
            if t == 0:
                return 20.0
            lookback = min(5, t)
            recent = core.D[t - lookback : t]
            return float(np.mean(recent)) if recent.size > 0 else 20.0

    def _compute_graph_obs(self):
        core = self.unwrapped
        t = core.period
        net = core.network

        inv_positions = []
        lt_targets = []
        gaps = []

        mu = self._estimate_mu(t)

        for node in self.main_nodes:
            node_idx = net.node_map[node]
            on_hand = core.X[t, node_idx]

            in_transit = 0.0
            info = self.node_edges[node]
            for i, edge in info["incoming"]:
                in_transit += core.Y[t, i] if t < core.num_periods else 0

            backlog = 0.0
            if getattr(core, "backlog", True) and node in net.retail:
                for k in core.graph.successors(node):
                    if (node, k) in net.retail_map:
                        r_idx = net.retail_map[(node, k)]
                        if t > 0:
                            backlog += core.U[t - 1, r_idx]

            inv_pos = on_hand + in_transit - backlog
            inv_positions.append(inv_pos)

            lt_target = mu * (info["max_L"] + 1)
            lt_targets.append(lt_target)

            gaps.append(lt_target - inv_pos)

        # Global features
        if t > 0:
            lookback = min(3, t)
            demand_vel = np.mean(core.D[t - lookback : t])
        else:
            demand_vel = mu

        norm_time = t / max(core.num_periods, 1)

        return np.array(
            inv_positions + lt_targets + gaps + [demand_vel, norm_time],
            dtype=np.float64,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._compute_graph_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["raw_reward"] = reward
        return self._compute_graph_obs(), reward, terminated, truncated, info
