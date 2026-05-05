from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from gym_invmgmt.demand_engine import DemandEngine
from gym_invmgmt.network_topology import SupplyChainNetwork
from gym_invmgmt.utils import assign_env_config


class CoreEnv(gym.Env):
    """Multi-Echelon Inventory Management Digital Twin.

    A Gymnasium-compatible environment simulating inventory management
    across configurable network topologies with composable demand dynamics.

    Architecture
    ------------
    - **Dynamics**: Handled here (Step, Reset, Reward).
    - **Topology**: Handled by ``self.network`` (SupplyChainNetwork class).
    - **Demand**:   Handled by ``self.demand_engine`` (DemandEngine class).

    State Matrices
    --------------
    All matrices are indexed as ``[period, link_or_node]``.

    **Stocks** (cumulative balances — read a single ``[t]`` row, never sum across time):

    ==================  =======================  ====================  ====================================
    Variable            Shape                    Property Alias        Description
    ==================  =======================  ====================  ====================================
    ``X[t, j]``         ``(T+1, n_main_nodes)``  ``.inventory``        On-hand inventory at node *j*,
                                                                       start of period *t*.
    ``Y[t, i]``         ``(T+1, n_reorder)``     ``.pipeline``         Total pipeline (in-transit) on
                                                                       reorder link *i*, start of period
                                                                       *t*.  ``Y[t+1] = Y[t] − arrived
                                                                       + R[t]``.
    ``U[t, i]``         ``(T, n_retail)``         ``.standing_backlog`` Standing backlog on retail link *i*
                                                                       at *end* of period *t*.
    ==================  =======================  ====================  ====================================

    **Flows** (per-period events — can be summed across time for totals):

    ==================  =======================  ======================  ==================================
    Variable            Shape                    Property Alias          Description
    ==================  =======================  ======================  ==================================
    ``D[t, i]``         ``(T, n_retail)``         ``.demand``             Customer demand on retail link *i*
                                                                         during period *t*.
    ``R[t, i]``         ``(T, n_reorder)``        ``.orders_filled``      **Filled** replenishment on
                                                                         reorder link *i*.  May be less
                                                                         than agent's request.
    ``S[t, k]``         ``(T, n_network)``        ``.shipments``          Material shipped on network edge
                                                                         *k*.  Reorder: ``= R``; Retail:
                                                                         ``= min(demand, inv)``.
    ``action_log[t,i]`` ``(T, n_reorder)``        ``.orders_requested``   Agent's **raw unconstrained**
                                                                         request.  ``action_log − R`` =
                                                                         rejected qty.
    ``P[t, j]``         ``(T, n_main_nodes)``     ``.profit``             Profit earned at node *j* during
                                                                         period *t*.
    ==================  =======================  ======================  ==================================

    Step Sequence (order-first)
    ---------------------------
    Each ``env.step(action)`` executes:

    1. **Place orders** — agent's action is constrained and recorded in ``R``; pipeline ``Y`` updated.
    2. **Receive deliveries** — goods arriving (``R[t−L]``) added to on-hand ``X``.
    3. **Realize demand** — ``D[t]`` sampled; retail sales ``S`` computed; backlog ``U`` updated.
    4. **Compute profit** — revenue, costs, penalties → ``P[t]``.
    5. **Build observation** — ``[D[t−1], X[t+1], arrival_schedule_from_R]`` returned to agent.
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        scenario: str = "network",
        demand_config: Optional[dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Store kwargs before popping so agents can reconstruct the environment identically
        import copy

        self.env_kwargs = {"scenario": scenario, "demand_config": demand_config, "render_mode": render_mode}
        self.env_kwargs.update(copy.deepcopy(kwargs))

        self.num_periods = kwargs.pop("num_periods", 30)

        self.backlog = kwargs.get("backlog", True)
        self.alpha = kwargs.pop("alpha", 1.00)
        self.render_mode = render_mode

        # --- 1. Initialize Demand Engine ---
        if demand_config is None:
            demand_config = {"type": "stationary", "base_mu": 20, "use_goodwill": False}
        self.demand_engine = DemandEngine(demand_config)

        # --- 2. Initialize Network Context ---
        network_kwargs = {k: v for k, v in kwargs.items() if k in ["user_D", "sample_path"]}
        config_path = kwargs.pop("config_path", None)
        self.network = SupplyChainNetwork(
            scenario=scenario, num_periods=self.num_periods, config_path=config_path, **network_kwargs
        )

        self.graph = self.network.graph
        self.levels = self.network.levels

        assign_env_config(self, kwargs)

        # --- 3. Setup Spaces ---
        num_reorder = len(self.network.reorder_links)

        self.extra_features_dim = 2
        self.obs_dim = (
            self.network.pipeline_length
            + len(self.network.main_nodes)
            + len(self.network.retail_links) * 2
            + self.extra_features_dim
        )

        self.action_space = gym.spaces.Box(
            low=np.zeros(num_reorder),
            high=np.ones(num_reorder) * (self.network.init_inv_max + self.network.capacity_max * self.num_periods),
            dtype=np.float64,
        )

        self.observation_space = gym.spaces.Box(
            low=np.full(self.obs_dim, -1e6, dtype=np.float64),
            high=np.full(self.obs_dim, 1e6, dtype=np.float64),
            dtype=np.float64,
        )

        # Initialize PRNG for constructor-time reset
        self.np_random = np.random.default_rng(0)
        self.reset()

    # ── Descriptive Property Aliases ──────────────────────────────────
    # Allow external consumers (dashboards, heuristics, analysis scripts)
    # to use readable names while keeping concise notation in core math.

    @property
    def inventory(self) -> np.ndarray:
        """On-hand inventory at each node. Stock variable ``X[t, j]``."""
        return self.X

    @property
    def pipeline(self) -> np.ndarray:
        """Pipeline (in-transit) inventory on each reorder link. Stock variable ``Y[t, i]``."""
        return self.Y

    @property
    def orders_filled(self) -> np.ndarray:
        """Filled replenishment orders on each reorder link. Flow variable ``R[t, i]``."""
        return self.R

    @property
    def orders_requested(self) -> np.ndarray:
        """Agent's raw unconstrained order requests. ``action_log[t, i]``."""
        return self.action_log

    @property
    def shipments(self) -> np.ndarray:
        """Material shipped on all network edges (reorder + retail). Flow variable ``S[t, k]``."""
        return self.S

    @property
    def demand(self) -> np.ndarray:
        """Customer demand realized on each retail link. Flow variable ``D[t, i]``."""
        return self.D

    @property
    def standing_backlog(self) -> np.ndarray:
        """Standing backlog on each retail link. Stock variable ``U[t, i]``."""
        return self.U

    @property
    def profit(self) -> np.ndarray:
        """Profit earned at each node per period. Flow variable ``P[t, j]``."""
        return self.P

    def _RESET(self):
        T = self.num_periods
        n_all_nodes = len(self.network.all_nodes)
        n_reorder = len(self.network.reorder_links)
        n_network = len(self.network.network_links)
        n_retail = len(self.network.retail_links)

        self.X = np.zeros((T + 1, n_all_nodes))
        self.Y = np.zeros((T + 1, n_reorder))
        self.R = np.zeros((T, n_reorder))
        self.S = np.zeros((T, n_network))
        self.D = np.zeros((T, n_retail))
        self.U = np.zeros((T, n_retail))
        self.P = np.zeros((T, n_all_nodes))

        self.GW = np.zeros(T)

        self.demand_engine.reset(np_random=self.np_random)

        self.period = 0
        for j in self.network.main_nodes:
            idx = self.network.node_map[j]
            self.X[0, idx] = self.network.graph.nodes[j].get("I0", 0)

        self.action_log = np.zeros((T, n_reorder))
        self._update_state()
        return self.state

    def _update_state(self):
        # ── Demand: last REALIZED demand (lag-1) ──
        # After step t completes, self.period = t+1 and D[t] holds realized
        # demand. Reading D[self.period] would read an unsampled future slot (=0).
        if self.period > 0:
            demand = self.D[self.period - 1, :]
            backlog = self.U[self.period - 1, :]
        else:
            demand = np.zeros(len(self.network.retail_links))
            backlog = np.zeros(len(self.network.retail_links))

        # ── Inventory: current on-hand (already correct) ──
        main_indices = [self.network.node_map[n] for n in self.network.main_nodes]
        inventory = self.X[self.period, main_indices]

        # ── Pipeline: arrival-indexed from R matrix ──
        # For each edge with lead time L, build an L-element vector:
        #   arrivals[0] = units arriving THIS step    (oldest in-transit order)
        #   arrivals[k] = units arriving in k steps
        #   arrivals[L-1] = units arriving in L-1 steps (most recent order)
        pipeline = []
        for edge_tuple in self.network.reorder_links:
            idx = self.network.reorder_map[edge_tuple]
            L = self.network.lead_times[edge_tuple]

            if L == 0:
                continue

            arrivals = np.zeros(L)
            for k in range(L):
                # Order placed at (self.period - L + k) arrives at step
                # (self.period + k). Only read if the order was already placed.
                order_time = self.period - L + k
                if 0 <= order_time < self.period:
                    arrivals[k] = self.R[order_time, idx]
            pipeline.append(arrivals)

        if len(pipeline) > 0:
            pipeline = np.hstack(pipeline)
        else:
            pipeline = np.array([])

        extra_features = self.demand_engine.get_observation(self.period)

        self.state = np.hstack([demand, backlog, inventory, pipeline, extra_features])

    def _STEP(self, action):
        t = self.period

        # Guard: prevent crash if step() is called after episode ended
        if t >= self.num_periods:
            return self.state, 0.0, False, True, {}

        action_arr = np.zeros(len(self.network.reorder_links))
        if isinstance(action, dict):
            for key, val in action.items():
                if key in self.network.reorder_map:
                    action_arr[self.network.reorder_map[key]] = val
        else:
            # Flat array: index i maps to self.network.reorder_links[i] = (supplier, buyer)
            action_arr = np.array(action)

        self.action_log[t, :] = action_arr

        # 1. Place Orders
        allocated_inv = {}
        allocated_cap = {}

        for i, (supplier, purchaser) in enumerate(self.network.reorder_links):
            request = max(action_arr[i], 0)
            supp_idx = self.network.node_map[supplier]

            if supplier in self.network.rawmat:
                self.R[t, i] = request
                net_idx = self.network.network_map[(supplier, purchaser)]
                self.S[t, net_idx] = request

            elif supplier in self.network.distrib:
                available = self.X[t, supp_idx] - allocated_inv.get(supplier, 0.0)
                amt = min(request, available)
                self.R[t, i] = amt
                net_idx = self.network.network_map[(supplier, purchaser)]
                self.S[t, net_idx] = amt
                allocated_inv[supplier] = allocated_inv.get(supplier, 0.0) + amt

            elif supplier in self.network.factory:
                C = self.network.graph.nodes[supplier]["C"]
                v = self.network.graph.nodes[supplier]["v"]

                available_inv = self.X[t, supp_idx] - allocated_inv.get(supplier, 0.0)
                available_cap = C - allocated_cap.get(supplier, 0.0)

                amt = min(request, available_cap, v * available_inv)
                self.R[t, i] = amt
                net_idx = self.network.network_map[(supplier, purchaser)]
                self.S[t, net_idx] = amt

                allocated_inv[supplier] = allocated_inv.get(supplier, 0.0) + (1.0 / v) * amt
                allocated_cap[supplier] = allocated_cap.get(supplier, 0.0) + amt

        # 2. Deliveries & Inventory Update
        for j in self.network.main_nodes:
            j_idx = self.network.node_map[j]

            incoming_total = 0.0
            for pred_node_idx, reorder_idx, L in self.network.pred_reorder_indices[j]:
                if t - L >= 0:
                    delivery = self.R[t - L, reorder_idx]
                else:
                    delivery = 0.0
                incoming_total += delivery

                self.Y[t + 1, reorder_idx] = self.Y[t, reorder_idx] - delivery + self.R[t, reorder_idx]

            v = self.network.graph.nodes[j].get("v", 1.0)

            outgoing_sum = 0.0
            for succ_node_idx, net_idx in self.network.succ_network_indices[j]:
                outgoing_sum += self.S[t, net_idx]

            outgoing = (1.0 / v) * outgoing_sum

            self.X[t + 1, j_idx] = self.X[t, j_idx] + incoming_total - outgoing

        # Update Goodwill
        if t > 0:
            total_unfulfilled_prev = np.sum(self.U[t - 1, :])
            self.demand_engine.update_goodwill(total_unfulfilled_prev)

        self.GW[t] = self.demand_engine.sentiment

        # 3. Demand Realization
        for i, (j, k) in enumerate(self.network.retail_links):
            edge_props = self.network.graph.edges[(j, k)]
            user_D = edge_props.get("user_D", 0)

            demand_val = 0.0
            if isinstance(user_D, (np.ndarray, list)):
                demand_val = user_D[t]
                if self.demand_engine.use_goodwill:
                    demand_val *= self.demand_engine.sentiment
            else:
                dist = edge_props["demand_dist"]
                dist_params = edge_props.get("dist_param", None)
                demand_val = self.demand_engine.sample(t, dist, dist_params=dist_params)

            self.D[t, i] = demand_val

            if self.backlog and t >= 1:
                eff_D = self.D[t, i] + self.U[t - 1, i]
            else:
                eff_D = self.D[t, i]

            j_idx = self.network.node_map[j]
            X_retail = self.X[t + 1, j_idx]

            sold = min(eff_D, X_retail)

            net_idx = self.network.network_map[(j, k)]
            self.S[t, net_idx] = sold

            self.X[t + 1, j_idx] -= sold
            self.U[t, i] = eff_D - sold

        # 4. Calculate Profit
        for j in self.network.main_nodes:
            j_idx = self.network.node_map[j]

            SR = 0.0
            for succ_node, net_idx in self.network.succ_network_indices[j]:
                succ_id = self.network.idx_to_node[succ_node]
                p = self.network.graph.edges[(j, succ_id)]["p"]
                SR += p * self.S[t, net_idx]

            PC = 0.0
            for pred_node, reorder_idx, L in self.network.pred_reorder_indices[j]:
                pred_id = self.network.idx_to_node[pred_node]
                p = self.network.graph.edges[(pred_id, j)]["p"]
                PC += p * self.R[t, reorder_idx]

            HC = 0.0
            if j not in self.network.rawmat:
                h = self.network.graph.nodes[j]["h"]
                HC += h * self.X[t + 1, j_idx]

                for pred_node, reorder_idx, L in self.network.pred_reorder_indices[j]:
                    pred_id = self.network.idx_to_node[pred_node]
                    g = self.network.graph.edges[(pred_id, j)]["g"]
                    HC += g * self.Y[t + 1, reorder_idx]

            OC = 0.0
            if j in self.network.factory:
                node_props = self.network.graph.nodes[j]
                sales_vol = 0.0
                for _, net_idx in self.network.succ_network_indices[j]:
                    sales_vol += self.S[t, net_idx]
                OC = (node_props["o"] / node_props["v"]) * sales_vol

            UP = 0.0
            if j in self.network.retail:
                for k in self.network.graph.successors(j):
                    if (j, k) in self.network.retail_map:
                        retail_idx = self.network.retail_map[(j, k)]
                        b = self.network.graph.edges[(j, k)]["b"]
                        UP += b * self.U[t, retail_idx]

            # Fixed Ordering Cost: flat fee K per reorder link with non-zero order.
            # When K > 0, this penalizes frequent small orders and incentivizes
            # (s,S)-style batch ordering behavior. K defaults to 0 for backward
            # compatibility with all existing scenarios and YAML configs.
            FK = 0.0
            for pred_node, reorder_idx, L in self.network.pred_reorder_indices[j]:
                pred_id = self.network.idx_to_node[pred_node]
                K_cost = self.network.graph.edges[(pred_id, j)].get("K", 0.0)
                if K_cost > 0.0 and self.R[t, reorder_idx] > 0.0:
                    FK += K_cost

            self.P[t, j_idx] = (self.alpha**t) * (SR - PC - OC - HC - UP - FK)

        self.period += 1
        reward = float(np.sum(self.P[t, :]))

        if self.period >= self.num_periods:
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False

        self._update_state()

        info = {
            "period": t,
            "step_reward": reward,
            "total_inventory": np.sum(self.X[t + 1, :]),
            "total_backlog": np.sum(self.U[t, :]),
            "sentiment": self.demand_engine.sentiment,
        }

        return self.state, reward, terminated, truncated, info

    def sample_action(self):
        return self.action_space.sample()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self._STEP(action)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Support dynamic reconfiguration via options
        if options is not None:
            if "demand_config" in options:
                self.demand_engine = DemandEngine(options["demand_config"])

        return self._RESET(), {}

    def render(self):
        """Render the current environment state."""
        if self.render_mode is None:
            return None

        t = max(0, self.period - 1)
        main_indices = [self.network.node_map[n] for n in self.network.main_nodes]
        inv = self.X[t, main_indices] if t < len(self.X) else np.zeros(len(main_indices))

        lines = [
            f"═══ Period {t + 1}/{self.num_periods} ═══",
            f"  Inventory : {np.round(inv, 1)}",
            f"  Demand    : {np.round(self.D[t, :], 1) if t < len(self.D) else 'N/A'}",
            f"  Backlog   : {np.round(self.U[t, :], 1) if t < len(self.U) else 'N/A'}",
            f"  Profit    : {np.sum(self.P[t, :]):.2f}" if t < len(self.P) else "",
            f"  Sentiment : {self.demand_engine.sentiment:.3f}",
        ]
        output = "\n".join(lines)

        if self.render_mode == "human":
            print(output)
            return None
        elif self.render_mode == "rgb_array":
            from gym_invmgmt.visualization import render_rgb_array

            return render_rgb_array(self)
        return output

    # ── Visualization Methods ─────────────────────────────────────────

    def plot_network(self, detailed: bool = False, save_path: Optional[str] = None):
        """
        Visualize the supply chain network topology.

        Delegates to :func:`gym_invmgmt.visualization.plot_network`.

        Args:
            detailed: If True, show node/edge parameters (costs, lead times, capacities).
            save_path: If provided, save the figure to this path instead of showing.
        """
        from gym_invmgmt.visualization import plot_network

        return plot_network(
            self.network,
            detailed=detailed,
            save_path=save_path,
            num_periods=self.num_periods,
            backlog=self.backlog,
            n_actions=self.action_space.shape[0],
            n_obs=self.observation_space.shape[0],
        )
