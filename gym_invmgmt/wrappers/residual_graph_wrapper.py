"""
residual_graph_wrapper.py — Enriched Graph-Only Observation for Transfer Architectures

Produces the same V2-enriched node features as DomainFeatureWrapper(enhanced=True)
but in graph-only format (no base_obs), suitable for topology-invariant architectures.

Observation layout (grouped):
  Per-node (×n_main, 8 each): inv_pos, lt_target, gap, on_hand,
                                holding_cost, capacity, is_factory, is_retail
  Global (10): demand_vel, norm_time, sin_time, cos_time,
               demand_hist[5], goodwill_sentiment

Total: 8 * n_main + 10

This wrapper is used by both BA-MPNN-Pool and Residual GCN-Pool.
For the Residual variant, heuristic actions are computed and stored
as an observation tail so the policy receives the current decision's
heuristic anchor before choosing a residual action.
"""

import gymnasium as gym
import numpy as np


class ResidualGraphWrapper(gym.Wrapper):
    """
    Graph-structured observation wrapper with V2-enriched features.

    If heuristic_agent is provided:
      - Computes heuristic base actions before each decision
      - Appends them to the graph observation
      - Applies residual action mapping: final = max(0, base + δ × max_residual)
    """

    NODE_FEATS = 8
    GLOBAL_FEATS = 10
    DEMAND_WINDOW = 5

    def __init__(self, env, heuristic_agent=None, max_residual=50.0, is_blind=False):
        super().__init__(env)
        self.heuristic_agent = heuristic_agent
        self.max_residual = max_residual
        self.is_blind = is_blind

        net = self.env.unwrapped.network
        self.main_nodes = sorted([n for n in net.graph.nodes() if n not in net.market and n not in net.rawmat])
        self.n_main = len(self.main_nodes)

        # Per-node topology info
        self.node_edges = {}
        for node in self.main_nodes:
            incoming = [(i, e) for i, e in enumerate(net.reorder_links) if e[1] == node]
            max_L = max([net.graph.edges[e]["L"] for _, e in incoming], default=0)
            self.node_edges[node] = {"incoming": incoming, "max_L": max_L}

        # Static node properties (normalized)
        cap_max = max([net.graph.nodes[n].get("C", 0) for n in self.main_nodes], default=1)
        h_max = max([net.graph.nodes[n].get("h", 0) for n in self.main_nodes], default=1)
        self._node_holding = []
        self._node_capacity = []
        self._node_is_factory = []
        self._node_is_retail = []
        for node in self.main_nodes:
            props = net.graph.nodes[node]
            self._node_holding.append(props.get("h", 0.0) / max(h_max, 1e-8))
            self._node_capacity.append(props.get("C", 0.0) / max(cap_max, 1e-8))
            self._node_is_factory.append(1.0 if node in net.factory else 0.0)
            self._node_is_retail.append(1.0 if node in net.retail else 0.0)

        # Observation space
        obs_dim = self.NODE_FEATS * self.n_main + self.GLOBAL_FEATS
        num_reorder = len(net.reorder_links)
        if heuristic_agent is not None:
            obs_dim += num_reorder

        self.observation_space = gym.spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float64),
            high=np.full(obs_dim, np.inf, dtype=np.float64),
            dtype=np.float64,
        )

        # Action space: if residual, δ ∈ [-max_residual, +max_residual]
        if heuristic_agent is not None:
            self.action_space = gym.spaces.Box(
                low=-self.max_residual * np.ones(num_reorder),
                high=self.max_residual * np.ones(num_reorder),
                dtype=np.float64,
            )

        # Last heuristic actions (for extractor side-channel)
        self._last_heuristic_actions = None

    @property
    def last_heuristic_actions(self):
        """Access the heuristic base actions from the most recent step."""
        return self._last_heuristic_actions

    def _estimate_mu(self, t):
        core = self.env.unwrapped
        if not self.is_blind:
            return core.demand_engine.get_current_mu(t)
        else:
            if t == 0:
                return 20.0
            lookback = min(5, t)
            recent = core.D[t - lookback : t]
            return float(np.mean(recent)) if recent.size > 0 else 20.0

    def _compute_obs(self):
        core = self.env.unwrapped
        t = core.period
        net = core.network

        inv_positions = []
        lt_targets = []
        gaps = []
        on_hands = []

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
            on_hands.append(on_hand)

            lt_target = mu * (info["max_L"] + 1)
            lt_targets.append(lt_target)
            gaps.append(lt_target - inv_pos)

        # Grouped layout: [all_feat0, all_feat1, ...]
        node_feats = (
            inv_positions
            + lt_targets
            + gaps
            + on_hands
            + self._node_holding
            + self._node_capacity
            + self._node_is_factory
            + self._node_is_retail
        )

        # Global features
        if t > 0:
            lookback = min(3, t)
            demand_vel = np.mean(core.D[t - lookback : t])
        else:
            demand_vel = mu

        T = max(core.num_periods, 1)
        norm_time = t / T
        sin_time = np.sin(2 * np.pi * t / T)
        cos_time = np.cos(2 * np.pi * t / T)

        demand_hist = np.zeros(self.DEMAND_WINDOW, dtype=np.float64)
        for k in range(self.DEMAND_WINDOW):
            idx = t - k - 1
            if idx >= 0:
                demand_hist[k] = float(np.mean(core.D[idx]))

        goodwill = getattr(core.demand_engine, "sentiment", 1.0)

        global_feats = [demand_vel, norm_time, sin_time, cos_time] + demand_hist.tolist() + [goodwill]

        obs_arr = np.array(node_feats + global_feats, dtype=np.float64)

        if self.heuristic_agent is not None:
            t_curr = core.period
            if t_curr < core.num_periods:
                core._update_state()
                base_actions = self.heuristic_agent.get_action(core.state, t_curr)
                if isinstance(base_actions, dict):
                    h_act = np.array([base_actions.get(e, 0.0) for e in net.reorder_links])
                else:
                    h_act = np.array(base_actions)
            else:
                h_act = np.zeros(len(net.reorder_links))

            self._last_heuristic_actions = h_act
            obs_arr = np.concatenate([obs_arr, h_act], dtype=np.float64)

        return obs_arr

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_heuristic_actions = None
        graph_obs = self._compute_obs()
        return graph_obs, info

    def step(self, action):
        core = self.env.unwrapped

        if self.heuristic_agent is not None:
            if self._last_heuristic_actions is None:
                self._last_heuristic_actions = np.zeros(len(core.network.reorder_links))

            action_arr = np.atleast_1d(np.asarray(action, dtype=np.float64))
            final_action = np.zeros(len(core.network.reorder_links))
            for i in range(len(core.network.reorder_links)):
                base_val = self._last_heuristic_actions[i]
                delta = action_arr[i] if i < len(action_arr) else 0.0
                final_action[i] = max(0.0, base_val + delta)

            final_action = np.clip(final_action, core.action_space.low, core.action_space.high)
            obs, reward, terminated, truncated, info = self.env.step(final_action)
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)

        graph_obs = self._compute_obs()
        info["raw_reward"] = reward
        if self._last_heuristic_actions is not None:
            info["heuristic_actions"] = self._last_heuristic_actions.copy()

        return graph_obs, reward, terminated, truncated, info
