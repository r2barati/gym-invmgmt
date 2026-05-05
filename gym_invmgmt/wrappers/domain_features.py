import gymnasium as gym
import numpy as np


class DomainFeatureWrapper(gym.Wrapper):
    """
    Feature-augmented wrapper.

    Appends domain-aware features to the observation vector.

    V1 (enhanced=False, default — backward compatible):
      Per-node (×n_main): inv_pos, lt_target, gap  (3 features)
      Global: demand_vel, norm_time  (2 features)
      Total augmented = 3 * n_main + 2

    V2 (enhanced=True):
      Per-node (×n_main): inv_pos, lt_target, gap, on_hand,
                           holding_cost, capacity, is_factory, is_retail  (8 features)
      Global: demand_vel, norm_time, sin_time, cos_time,
              demand_hist[5], goodwill_sentiment  (10 features)
      Total augmented = 8 * n_main + 10

    V4 (enhanced=True, pipeline_feats=True):
      Per-node: V2 features + in_transit, backlog  (10 features)
      Global: same as V2  (10 features)
      Total augmented = 10 * n_main + 10
    """

    # Class constant so external code (GNN extractor) can query the schema
    V1_NODE_FEATS = 3
    V1_GLOBAL_FEATS = 2
    V2_NODE_FEATS = 8
    V4_NODE_FEATS = 10  # V2 + in_transit + backlog
    V2_GLOBAL_FEATS = 10
    V2_DEMAND_WINDOW = 5

    def __init__(self, env, is_blind=False, enhanced=False, grouped=False, pipeline_feats=False):
        super().__init__(env)
        self.is_blind = is_blind
        self.enhanced = enhanced
        self.grouped = grouped  # V3: grouped layout [all_feat0, all_feat1, ...] vs interleaved
        self.pipeline_feats = pipeline_feats  # V4: expose in_transit & backlog separately

        # --- Pre-compute static topology info ---
        core = self.unwrapped
        net = core.network
        self.main_nodes = sorted([n for n in net.graph.nodes() if n not in net.market and n not in net.rawmat])
        n_main = len(self.main_nodes)

        # Per-node: incoming edges + max lead time
        self.node_edges = {}
        for node in self.main_nodes:
            incoming = [(i, e) for i, e in enumerate(net.reorder_links) if e[1] == node]
            max_L = max([net.graph.edges[e]["L"] for _, e in incoming], default=0)
            self.node_edges[node] = {"incoming": incoming, "max_L": max_L}

        # Pre-compute static per-node properties for V2
        if self.enhanced:
            self._node_holding = []
            self._node_capacity = []
            self._node_is_factory = []
            self._node_is_retail = []
            cap_max = max([net.graph.nodes[n].get("C", 0) for n in self.main_nodes], default=1)
            h_max = max([net.graph.nodes[n].get("h", 0) for n in self.main_nodes], default=1)
            for node in self.main_nodes:
                props = net.graph.nodes[node]
                self._node_holding.append(props.get("h", 0.0) / max(h_max, 1e-8))
                self._node_capacity.append(props.get("C", 0.0) / max(cap_max, 1e-8))
                self._node_is_factory.append(1.0 if node in net.factory else 0.0)
                self._node_is_retail.append(1.0 if node in net.retail else 0.0)

        if self.enhanced and self.pipeline_feats:
            n_node_feats = self.V4_NODE_FEATS
            n_global_feats = self.V2_GLOBAL_FEATS
        elif self.enhanced:
            n_node_feats = self.V2_NODE_FEATS
            n_global_feats = self.V2_GLOBAL_FEATS
        else:
            n_node_feats = self.V1_NODE_FEATS
            n_global_feats = self.V1_GLOBAL_FEATS

        self.n_node_feats = n_node_feats
        self.n_global_feats = n_global_feats
        self.n_augmented = n_node_feats * n_main + n_global_feats
        total_dim = self.env.observation_space.shape[0] + self.n_augmented

        self.observation_space = gym.spaces.Box(
            low=np.full(total_dim, -np.inf, dtype=np.float64),
            high=np.full(total_dim, np.inf, dtype=np.float64),
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

    def _compute_augmented_features(self):
        env = self.unwrapped
        t = env.period
        net = env.network

        inv_positions = []
        lt_targets = []
        gaps = []
        on_hands = []
        in_transits = []
        backlogs = []

        mu = self._estimate_mu(t)

        for node in self.main_nodes:
            node_idx = net.node_map[node]
            on_hand = env.X[t, node_idx]

            in_transit = 0.0
            info = self.node_edges[node]
            for i, edge in info["incoming"]:
                in_transit += env.Y[t, i] if t < env.num_periods else 0

            backlog = 0.0
            if getattr(env, "backlog", True) and node in net.retail:
                for k in env.graph.successors(node):
                    if (node, k) in net.retail_map:
                        r_idx = net.retail_map[(node, k)]
                        if t > 0:
                            backlog += env.U[t - 1, r_idx]

            inv_pos = on_hand + in_transit - backlog
            inv_positions.append(inv_pos)
            on_hands.append(on_hand)
            in_transits.append(in_transit)
            backlogs.append(backlog)

            lt_target = mu * (info["max_L"] + 1)
            lt_targets.append(lt_target)

            gaps.append(lt_target - inv_pos)

        if t > 0:
            lookback = min(3, t)
            demand_vel = np.mean(env.D[t - lookback : t])
        else:
            demand_vel = mu

        norm_time = t / max(env.num_periods, 1)

        if not self.enhanced:
            # V1 layout: [inv_pos * n, lt_target * n, gap * n, demand_vel, norm_time]
            return np.array(
                inv_positions + lt_targets + gaps + [demand_vel, norm_time],
                dtype=np.float64,
            )

        # --- V2/V3/V4 Enhanced Features ---
        base_per_node = [
            inv_positions,
            lt_targets,
            gaps,
            on_hands,
            self._node_holding,
            self._node_capacity,
            self._node_is_factory,
            self._node_is_retail,
        ]
        if self.pipeline_feats:
            base_per_node += [in_transits, backlogs]

        if self.grouped:
            # Grouped layout: [all_feat0, all_feat1, ...]
            node_feats = []
            for feat_list in base_per_node:
                node_feats.extend(feat_list)
        else:
            # Interleaved layout: [feat0_n0, feat1_n0, ... feat0_n1, feat1_n1, ...]
            node_feats = []
            for idx in range(len(self.main_nodes)):
                for feat_list in base_per_node:
                    node_feats.append(feat_list[idx])

        # Global: [demand_vel, norm_time, sin_time, cos_time, demand_hist[5], goodwill]
        T = max(env.num_periods, 1)
        sin_time = np.sin(2 * np.pi * t / T)
        cos_time = np.cos(2 * np.pi * t / T)

        # Demand history window (last 5 periods, zero-padded)
        demand_hist = np.zeros(self.V2_DEMAND_WINDOW, dtype=np.float64)
        for k in range(self.V2_DEMAND_WINDOW):
            idx = t - k - 1
            if idx >= 0:
                demand_hist[k] = np.mean(env.D[idx]) if env.D[idx].ndim > 0 else float(env.D[idx])

        # Goodwill sentiment (1.0 if not using goodwill)
        goodwill = getattr(env.demand_engine, "sentiment", 1.0)

        global_feats = [demand_vel, norm_time, sin_time, cos_time] + demand_hist.tolist() + [goodwill]

        return np.array(node_feats + global_feats, dtype=np.float64)

    def _augment_obs(self, obs):
        features = self._compute_augmented_features()
        return np.hstack([obs, features])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        augmented_obs = self._augment_obs(obs)
        return augmented_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        augmented_obs = self._augment_obs(obs)

        info["raw_reward"] = reward
        return augmented_obs, reward, terminated, truncated, info
