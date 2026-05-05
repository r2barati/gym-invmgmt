"""
run_heuristic.py — Base-stock heuristic using the Newsvendor critical ratio.

Implements a simple order-up-to policy: each period, compute a base-stock
target S from the Newsvendor critical ratio (CR = b / (b + h)) and the
Poisson inverse CDF, then order max(0, S − inventory_position).

This is the most common textbook inventory policy and serves as a strong
baseline for RL comparison.

Usage:
    python examples/run_heuristic.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

import gymnasium as gym
import gym_invmgmt
from gym_invmgmt import CoreEnv
from gym_invmgmt.utils import compute_kpis


# ── Newsvendor Base-Stock Policy ──────────────────────────────────────

class NewsvendorPolicy:
    """
    Newsvendor critical-ratio base-stock policy for multi-echelon networks.

    For each managed node, computes:
        CR = b / (b + h)            (critical ratio)
        S  = Poisson.ppf(CR, μ_L)   (order-up-to level)

    where μ_L = mean_demand × (lead_time + 1) is the lead-time demand,
    h is the holding cost, and b is the backlog penalty (or margin proxy).
    """

    def __init__(self, env: CoreEnv, service_level: float = 0.95):
        self.env = env
        network = env.network

        # Identify managed nodes (exclude raw materials and markets)
        self.main_nodes = sorted(
            n for n in env.graph.nodes()
            if n not in network.market and n not in network.rawmat
        )

        # Pre-compute per-node parameters
        self.node_info = {}
        for node in self.main_nodes:
            h = env.graph.nodes[node].get('h', 0.1)  # holding cost

            # Backlog penalty: explicit for retailers, margin-derived upstream
            if node in network.retail:
                b = max(
                    (env.graph.edges[(node, k)].get('b', 0)
                     for k in env.graph.successors(node)),
                    default=0.5,
                )
            else:
                # Approximate echelon cost from downstream selling price
                sell_prices = [
                    env.graph.edges.get((node, k), {}).get('p', 0)
                    for k in env.graph.successors(node)
                ]
                buy_prices = [
                    env.graph.edges.get(e, {}).get('p', 0)
                    for e in network.reorder_links if e[1] == node
                ]
                b = max(max(sell_prices, default=0) - min(buy_prices, default=0), 0.5)

            cr = b / (b + h) if (b + h) > 0 else service_level

            # Incoming supplier edges and max lead time
            incoming = [(i, e) for i, e in enumerate(network.reorder_links) if e[1] == node]
            max_L = max(
                (env.graph.edges[e]['L'] for _, e in incoming),
                default=0,
            )

            self.node_info[node] = {
                'cr': cr,
                'max_L': max_L,
                'incoming': incoming,
            }

    def get_action(self, obs: np.ndarray, period: int) -> np.ndarray:
        """Return order quantities for all reorder links."""
        env = self.env
        mu = env.demand_engine.get_current_mu(period)
        actions = np.zeros(len(env.network.reorder_links))

        for node in self.main_nodes:
            info = self.node_info[node]
            node_idx = env.network.node_map[node]

            # Lead-time demand
            mu_L = mu * (info['max_L'] + 1)
            target = poisson.ppf(info['cr'], mu_L) if mu_L > 0 else 0

            # Inventory position = on-hand + in-transit − backlog
            on_hand = env.X[period, node_idx]
            in_transit = sum(
                env.Y[period, i] for i, _ in info['incoming']
            )
            backlog = 0
            if node in env.network.retail and period > 0:
                for k in env.graph.successors(node):
                    if (node, k) in env.network.retail_map:
                        backlog = env.U[period - 1, env.network.retail_map[(node, k)]]

            inv_position = on_hand + in_transit - backlog
            order_needed = max(0, target - inv_position)

            # Split evenly among suppliers
            n_suppliers = len(info['incoming'])
            if n_suppliers > 0:
                per_supplier = order_needed / n_suppliers
                for idx, _ in info['incoming']:
                    actions[idx] = per_supplier

        return actions


# ── Run ───────────────────────────────────────────────────────────────

def main():
    scenarios = [
        ("network", "stationary", {}),
        ("serial",  "stationary", {}),
        ("network", "shock",      {"shock_time": 15, "shock_mag": 2.0}),
    ]

    for scenario, demand_type, extra_demand in scenarios:
        demand_config = {"type": demand_type, "base_mu": 20, **extra_demand}
        env = CoreEnv(scenario=scenario, demand_config=demand_config, num_periods=30)
        policy = NewsvendorPolicy(env)

        obs, _ = env.reset(seed=42)
        done = False
        t = 0
        while not done:
            action = policy.get_action(obs, t)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            t += 1

        kpis = compute_kpis(env)
        print(f"[{scenario:8s} | {demand_type:10s}]  "
              f"Profit: {kpis['total_profit']:8,.0f}  "
              f"Fill: {kpis['fill_rate']:.1%}  "
              f"Turns: {kpis['inventory_turns']:.2f}")

    print("\nDone. See the companion paper repository for the full benchmark-agent roster.")


if __name__ == "__main__":
    main()
