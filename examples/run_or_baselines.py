"""
run_or_baselines.py — Classical Operations Research inventory policies.

Implements three OR baselines that don't require any machine learning:
  1. Constant Order Policy — fixed order quantity each period
  2. Newsvendor Base-Stock  — critical ratio + Poisson PPF
  3. (s, S) Reorder-Point   — order only when position < s, up to S (with EOQ)

These represent the spectrum of classical approaches from naive to
enterprise-grade and provide meaningful baselines for RL comparison.

Usage:
    python examples/run_or_baselines.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson, norm

from gym_invmgmt import CoreEnv
from gym_invmgmt.utils import compute_kpis


# ── Policy 1: Constant Order ─────────────────────────────────────────

class ConstantOrderPolicy:
    """Order a fixed quantity q on every link every period."""

    def __init__(self, env: CoreEnv, q: float = 20.0):
        self.n_actions = len(env.network.reorder_links)
        self.q = q

    def get_action(self, obs, period):
        return np.full(self.n_actions, self.q)


# ── Policy 2: Newsvendor Base-Stock ───────────────────────────────────

class NewsvendorPolicy:
    """
    Order up to base-stock target S each period.
    S = Poisson.ppf(CR, μ_L) where CR = b / (b + h).
    """

    def __init__(self, env: CoreEnv):
        self.env = env
        net = env.network
        self.main_nodes = sorted(
            n for n in env.graph.nodes()
            if n not in net.market and n not in net.rawmat
        )
        self.node_info = {}
        for node in self.main_nodes:
            h = env.graph.nodes[node].get('h', 0.1)
            if node in net.retail:
                b = max((env.graph.edges[(node, k)].get('b', 0)
                         for k in env.graph.successors(node)), default=0.5)
            else:
                sell = [env.graph.edges.get((node, k), {}).get('p', 0)
                        for k in env.graph.successors(node)]
                buy = [env.graph.edges.get(e, {}).get('p', 0)
                       for e in net.reorder_links if e[1] == node]
                b = max(max(sell, default=0) - min(buy, default=0), 0.5)
            cr = b / (b + h) if (b + h) > 0 else 0.95
            incoming = [(i, e) for i, e in enumerate(net.reorder_links) if e[1] == node]
            max_L = max((env.graph.edges[e]['L'] for _, e in incoming), default=0)
            self.node_info[node] = {'cr': cr, 'max_L': max_L, 'incoming': incoming}

    def get_action(self, obs, period):
        env = self.env
        mu = env.demand_engine.get_current_mu(period)
        actions = np.zeros(len(env.network.reorder_links))
        for node in self.main_nodes:
            info = self.node_info[node]
            mu_L = mu * (info['max_L'] + 1)
            target = poisson.ppf(info['cr'], mu_L) if mu_L > 0 else 0
            on_hand = env.X[period, env.network.node_map[node]]
            in_transit = sum(env.Y[period, i] for i, _ in info['incoming'])
            inv_pos = on_hand + in_transit
            order = max(0, target - inv_pos)
            n_sup = len(info['incoming'])
            if n_sup > 0:
                for idx, _ in info['incoming']:
                    actions[idx] = order / n_sup
        return actions


# ── Policy 3: (s, S) Reorder-Point / Order-Up-To ─────────────────────

class SSPolicy:
    """
    (s, S) policy: order only when inventory position drops below s.
    When triggered, order enough to reach S.

    s = μ_L + z_α × σ_L           (safety stock reorder point)
    S = s + EOQ                    (order-up-to level)
    EOQ = sqrt(2 × μ × K / h)     (Economic Order Quantity)
    """

    def __init__(self, env: CoreEnv, service_level: float = 0.95):
        self.env = env
        self.z_alpha = norm.ppf(service_level)
        net = env.network
        self.main_nodes = sorted(
            n for n in env.graph.nodes()
            if n not in net.market and n not in net.rawmat
        )
        self.node_info = {}
        for node in self.main_nodes:
            h = env.graph.nodes[node].get('h', 0.1)
            K = (env.graph.nodes[node].get('o', 0.01) * 100
                 if node in net.factory else h * 50)
            incoming = [(i, e) for i, e in enumerate(net.reorder_links) if e[1] == node]
            max_L = max((env.graph.edges[e]['L'] for _, e in incoming), default=0)
            self.node_info[node] = {'h': h, 'K': K, 'max_L': max_L, 'incoming': incoming}

    def get_action(self, obs, period):
        env = self.env
        mu = env.demand_engine.get_current_mu(period)
        actions = np.zeros(len(env.network.reorder_links))
        for node in self.main_nodes:
            info = self.node_info[node]
            L, h, K = info['max_L'], info['h'], info['K']
            mu_L = mu * (L + 1)
            sigma_L = np.sqrt(mu_L)
            s = mu_L + self.z_alpha * sigma_L
            eoq = np.sqrt(2 * mu * K / h) if h > 0 and mu > 0 else mu_L
            S = s + eoq

            on_hand = env.X[period, env.network.node_map[node]]
            in_transit = sum(env.Y[period, i] for i, _ in info['incoming'])
            inv_pos = on_hand + in_transit

            if inv_pos < s:
                order = max(0, S - inv_pos)
            else:
                order = 0

            n_sup = len(info['incoming'])
            if n_sup > 0:
                for idx, _ in info['incoming']:
                    actions[idx] = order / n_sup
        return actions


# ── Runner ────────────────────────────────────────────────────────────

def run_policy(env: CoreEnv, policy, seed: int = 42) -> dict:
    """Run an episode and return KPIs."""
    obs, _ = env.reset(seed=seed)
    done, t = False, 0
    while not done:
        action = policy.get_action(obs, t)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        t += 1
    return compute_kpis(env)


def main():
    print("=" * 70)
    print("OR Baseline Comparison — Multi-Echelon Network (30 periods)")
    print("=" * 70)

    env_kwargs = dict(scenario='network', num_periods=30,
                      demand_config={'type': 'stationary', 'base_mu': 20})

    policies = {
        "Constant (q=10)":  lambda env: ConstantOrderPolicy(env, q=10),
        "Constant (q=20)":  lambda env: ConstantOrderPolicy(env, q=20),
        "Constant (q=30)":  lambda env: ConstantOrderPolicy(env, q=30),
        "Newsvendor":       lambda env: NewsvendorPolicy(env),
        "(s, S) Policy":    lambda env: SSPolicy(env),
    }

    print(f"\n{'Policy':<20}  {'Profit':>10}  {'Fill Rate':>10}  {'Avg Inv':>10}  {'Turns':>8}")
    print("-" * 70)

    for name, make_policy in policies.items():
        env = CoreEnv(**env_kwargs)
        policy = make_policy(env)
        kpis = run_policy(env, policy)
        print(f"{name:<20}  {kpis['total_profit']:>10,.0f}  "
              f"{kpis['fill_rate']:>10.1%}  "
              f"{kpis['avg_inventory']:>10,.0f}  "
              f"{kpis['inventory_turns']:>8.2f}")

    # ── Compare across demand scenarios ───────────────────────────────
    print(f"\n{'=' * 70}")
    print("(s, S) Policy Across Demand Scenarios")
    print(f"{'=' * 70}")

    demand_scenarios = [
        ("Stationary",  {'type': 'stationary', 'base_mu': 20}),
        ("Trend",       {'type': 'trend',      'base_mu': 20, 'trend_slope': 0.5}),
        ("Seasonal",    {'type': 'seasonal',   'base_mu': 20, 'seasonal_amp': 0.5}),
        ("Shock",       {'type': 'shock',      'base_mu': 20, 'shock_time': 15, 'shock_mag': 2.0}),
    ]

    print(f"\n{'Demand':<15}  {'Profit':>10}  {'Fill Rate':>10}  {'Avg Inv':>10}")
    print("-" * 55)

    for name, dcfg in demand_scenarios:
        env = CoreEnv(scenario='network', num_periods=30, demand_config=dcfg)
        policy = SSPolicy(env)
        kpis = run_policy(env, policy)
        print(f"{name:<15}  {kpis['total_profit']:>10,.0f}  "
              f"{kpis['fill_rate']:>10.1%}  "
              f"{kpis['avg_inventory']:>10,.0f}")


if __name__ == "__main__":
    main()
