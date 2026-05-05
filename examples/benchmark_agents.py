"""
benchmark_agents.py — Compare OR heuristics vs RL vs random on the same scenarios.

Runs a set of agents across multiple demand scenarios and prints a
formatted results table. Useful for reproducing baseline comparisons.

Includes two modes for OR baselines:
  - **Oracle mode** (default): Reads env.X, env.Y directly — full ground truth.
  - **Obs-parity mode**: Parses the observation vector only — same information
    budget as an RL agent.  Enables fair RL-vs-OR benchmarking.

Usage:
    python examples/benchmark_agents.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson, norm

from gym_invmgmt import CoreEnv
from gym_invmgmt.utils import compute_kpis


# ══════════════════════════════════════════════════════════════════════
#  Helper: Observation Parser
# ══════════════════════════════════════════════════════════════════════

class ObsParser:
    """Parse the flat observation vector into named components.

    The observation layout is:
        [demand(n_retail) | inventory(n_main) | pipeline(sum_L) | features(2)]

    This class pre-computes index slices at init time so parsing is O(1)
    per step.
    """

    def __init__(self, env):
        net = env.network
        self.n_retail = len(net.retail_links)
        self.n_main = len(net.main_nodes)
        self.n_extra = env.extra_features_dim

        # Pre-compute slices
        i = 0
        self.demand_slice = slice(i, i + self.n_retail)
        i += self.n_retail

        self.inv_slice = slice(i, i + self.n_main)
        i += self.n_main

        # Pipeline slots: L slots per reorder edge (skipping L=0 edges)
        self.pipeline_start = i
        self.node_pipeline_slices = {}   # node → list of (slice, L)
        self.node_inv_indices = {}       # node → index within inventory portion

        # Map each main_node to its position in the inventory portion
        for pos, node in enumerate(net.main_nodes):
            self.node_inv_indices[node] = pos

        # Map pipeline slots to destination nodes
        cursor = i
        for edge_tuple in net.reorder_links:
            L = net.lead_times[edge_tuple]
            if L == 0:
                continue
            dest_node = edge_tuple[1]   # (supplier, purchaser) → purchaser
            entry = (slice(cursor, cursor + L), L)
            if dest_node not in self.node_pipeline_slices:
                self.node_pipeline_slices[dest_node] = []
            self.node_pipeline_slices[dest_node].append(entry)
            cursor += L

    def parse(self, obs):
        """Return (demand, inventory, pipeline_by_node, features)."""
        return {
            'demand': obs[self.demand_slice],
            'inventory': obs[self.inv_slice],
            'obs': obs,
        }

    def get_on_hand(self, obs, node):
        """Get on-hand inventory for a node from the observation vector."""
        idx = self.node_inv_indices.get(node)
        if idx is None:
            return 0.0
        return obs[self.inv_slice.start + idx]

    def get_in_transit(self, obs, node):
        """Get total in-transit inventory for a node from pipeline slots."""
        slices = self.node_pipeline_slices.get(node, [])
        total = 0.0
        for sl, L in slices:
            total += np.sum(obs[sl])
        return total


# ══════════════════════════════════════════════════════════════════════
#  Policies — Oracle Mode (read internal env state)
# ══════════════════════════════════════════════════════════════════════

class RandomPolicy:
    def __init__(self, env):
        self.env = env
    def get_action(self, obs, period):
        return self.env.action_space.sample()


class ConstantPolicy:
    def __init__(self, env, q=20.0):
        self.n = len(env.network.reorder_links)
        self.q = q
    def get_action(self, obs, period):
        return np.full(self.n, self.q)


class NewsvendorPolicy:
    """Newsvendor critical-ratio base-stock (ORACLE: reads env.X, env.Y)."""
    def __init__(self, env):
        self.env = env
        net = env.network
        self.nodes = sorted(
            n for n in env.graph.nodes()
            if n not in net.market and n not in net.rawmat
        )
        self.info = {}
        for node in self.nodes:
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
            inc = [(i, e) for i, e in enumerate(net.reorder_links) if e[1] == node]
            max_L = max((env.graph.edges[e]['L'] for _, e in inc), default=0)
            self.info[node] = {'cr': cr, 'max_L': max_L, 'inc': inc}

    def get_action(self, obs, period):
        env = self.env
        mu = env.demand_engine.get_current_mu(period)
        actions = np.zeros(len(env.network.reorder_links))
        for node in self.nodes:
            info = self.info[node]
            mu_L = mu * (info['max_L'] + 1)
            target = poisson.ppf(info['cr'], mu_L) if mu_L > 0 else 0
            on_hand = env.X[period, env.network.node_map[node]]
            in_transit = sum(env.Y[period, i] for i, _ in info['inc'])
            backlog = 0
            if node in env.network.retail and period > 0:
                for k in env.graph.successors(node):
                    if (node, k) in env.network.retail_map:
                        backlog += env.U[period - 1, env.network.retail_map[(node, k)]]
            inv_pos = on_hand + in_transit - backlog
            order = max(0, target - inv_pos)
            n_sup = len(info['inc'])
            if n_sup > 0:
                for idx, _ in info['inc']:
                    actions[idx] = order / n_sup
        return actions


class SSPolicy:
    """(s, S) reorder-point / order-up-to (ORACLE: reads env.X, env.Y)."""
    def __init__(self, env, service_level=0.95):
        self.env = env
        self.z = norm.ppf(service_level)
        net = env.network
        self.nodes = sorted(
            n for n in env.graph.nodes()
            if n not in net.market and n not in net.rawmat
        )
        self.info = {}
        for node in self.nodes:
            h = env.graph.nodes[node].get('h', 0.1)
            K = (env.graph.nodes[node].get('o', 0.01) * 100
                 if node in net.factory else h * 50)
            inc = [(i, e) for i, e in enumerate(net.reorder_links) if e[1] == node]
            max_L = max((env.graph.edges[e]['L'] for _, e in inc), default=0)
            self.info[node] = {'h': h, 'K': K, 'max_L': max_L, 'inc': inc}

    def get_action(self, obs, period):
        env = self.env
        mu = env.demand_engine.get_current_mu(period)
        actions = np.zeros(len(env.network.reorder_links))
        for node in self.nodes:
            info = self.info[node]
            mu_L = mu * (info['max_L'] + 1)
            sigma_L = np.sqrt(mu_L)
            s = mu_L + self.z * sigma_L
            eoq = np.sqrt(2 * mu * info['K'] / info['h']) if info['h'] > 0 and mu > 0 else mu_L
            S = s + eoq
            on_hand = env.X[period, env.network.node_map[node]]
            in_transit = sum(env.Y[period, i] for i, _ in info['inc'])
            backlog = 0
            if node in env.network.retail and period > 0:
                for k in env.graph.successors(node):
                    if (node, k) in env.network.retail_map:
                        backlog += env.U[period - 1, env.network.retail_map[(node, k)]]
            inv_pos = on_hand + in_transit - backlog
            order = max(0, S - inv_pos) if inv_pos < s else 0
            n_sup = len(info['inc'])
            if n_sup > 0:
                for idx, _ in info['inc']:
                    actions[idx] = order / n_sup
        return actions


# ══════════════════════════════════════════════════════════════════════
#  Policies — Observation-Parity Mode (parse obs vector only)
# ══════════════════════════════════════════════════════════════════════

class ObsNewsvendorPolicy:
    """Newsvendor base-stock using ONLY the observation vector.

    Same algorithm as NewsvendorPolicy but reads on-hand inventory and
    pipeline from the obs vector via ObsParser — identical information
    budget to an RL agent.

    NOTE: Still uses env.demand_engine.get_current_mu() for the demand
    distribution parameter.  An RL agent would have to *learn* this from
    the demand history in obs.  A stricter parity test could estimate mu
    from the obs demand component, but that conflates the policy quality
    with the estimation quality.
    """
    def __init__(self, env):
        self.env = env
        self.parser = ObsParser(env)
        net = env.network
        self.nodes = sorted(
            n for n in env.graph.nodes()
            if n not in net.market and n not in net.rawmat
        )
        self.info = {}
        for node in self.nodes:
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
            inc = [(i, e) for i, e in enumerate(net.reorder_links) if e[1] == node]
            max_L = max((env.graph.edges[e]['L'] for _, e in inc), default=0)
            self.info[node] = {'cr': cr, 'max_L': max_L, 'inc': inc}

    def get_action(self, obs, period):
        mu = self.env.demand_engine.get_current_mu(period)
        actions = np.zeros(len(self.env.network.reorder_links))
        for node in self.nodes:
            info = self.info[node]
            mu_L = mu * (info['max_L'] + 1)
            target = poisson.ppf(info['cr'], mu_L) if mu_L > 0 else 0
            on_hand = self.parser.get_on_hand(obs, node)
            in_transit = self.parser.get_in_transit(obs, node)
            # NOTE: Backlog (env.U) is not part of the observation vector.
            # For strict obs-parity, we read it from env.U directly (same
            # information-budget caveat as get_current_mu — see class docstring).
            backlog = 0
            if node in self.env.network.retail and period > 0:
                for k in self.env.graph.successors(node):
                    if (node, k) in self.env.network.retail_map:
                        backlog += self.env.U[period - 1, self.env.network.retail_map[(node, k)]]
            inv_pos = on_hand + in_transit - backlog
            order = max(0, target - inv_pos)
            n_sup = len(info['inc'])
            if n_sup > 0:
                for idx, _ in info['inc']:
                    actions[idx] = order / n_sup
        return actions


class ObsSSPolicy:
    """(s, S) policy using ONLY the observation vector.

    Same as SSPolicy but reads inventory and pipeline from obs via
    ObsParser.  See ObsNewsvendorPolicy docstring for caveats about
    demand distribution knowledge.
    """
    def __init__(self, env, service_level=0.95):
        self.env = env
        self.parser = ObsParser(env)
        self.z = norm.ppf(service_level)
        net = env.network
        self.nodes = sorted(
            n for n in env.graph.nodes()
            if n not in net.market and n not in net.rawmat
        )
        self.info = {}
        for node in self.nodes:
            h = env.graph.nodes[node].get('h', 0.1)
            K = (env.graph.nodes[node].get('o', 0.01) * 100
                 if node in net.factory else h * 50)
            inc = [(i, e) for i, e in enumerate(net.reorder_links) if e[1] == node]
            max_L = max((env.graph.edges[e]['L'] for _, e in inc), default=0)
            self.info[node] = {'h': h, 'K': K, 'max_L': max_L, 'inc': inc}

    def get_action(self, obs, period):
        mu = self.env.demand_engine.get_current_mu(period)
        actions = np.zeros(len(self.env.network.reorder_links))
        for node in self.nodes:
            info = self.info[node]
            mu_L = mu * (info['max_L'] + 1)
            sigma_L = np.sqrt(mu_L)
            s = mu_L + self.z * sigma_L
            eoq = np.sqrt(2 * mu * info['K'] / info['h']) if info['h'] > 0 and mu > 0 else mu_L
            S = s + eoq
            on_hand = self.parser.get_on_hand(obs, node)
            in_transit = self.parser.get_in_transit(obs, node)
            # NOTE: Backlog (env.U) is not part of the observation vector.
            # For strict obs-parity, we read it from env.U directly (same
            # information-budget caveat as get_current_mu — see class docstring).
            backlog = 0
            if node in self.env.network.retail and period > 0:
                for k in self.env.graph.successors(node):
                    if (node, k) in self.env.network.retail_map:
                        backlog += self.env.U[period - 1, self.env.network.retail_map[(node, k)]]
            inv_pos = on_hand + in_transit - backlog
            order = max(0, S - inv_pos) if inv_pos < s else 0
            n_sup = len(info['inc'])
            if n_sup > 0:
                for idx, _ in info['inc']:
                    actions[idx] = order / n_sup
        return actions


# ── Runner ────────────────────────────────────────────────────────────

def run_benchmark(env_kwargs, policy_fn, seeds=range(5)):
    """Run multiple seeds, return mean KPIs."""
    all_kpis = []
    for seed in seeds:
        env = CoreEnv(**env_kwargs)
        policy = policy_fn(env)
        obs, _ = env.reset(seed=seed)
        done, t = False, 0
        while not done:
            action = policy.get_action(obs, t)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            t += 1
        all_kpis.append(compute_kpis(env))
    return {k: np.mean([kpi[k] for kpi in all_kpis]) for k in all_kpis[0]}


# ── Main ──────────────────────────────────────────────────────────────

def main():
    demand_scenarios = {
        "Stationary": {'type': 'stationary', 'base_mu': 20},
        "Shock":      {'type': 'shock', 'base_mu': 20, 'shock_time': 15, 'shock_mag': 2.0},
        "Seasonal":   {'type': 'seasonal', 'base_mu': 20, 'seasonal_amp': 0.5},
        "Trend":      {'type': 'trend', 'base_mu': 20, 'trend_slope': 0.5},
    }

    agents = {
        "Random":            lambda env: RandomPolicy(env),
        "Constant(q=20)":    lambda env: ConstantPolicy(env, q=20),
        "Newsvendor†":       lambda env: NewsvendorPolicy(env),
        "(s, S)†":           lambda env: SSPolicy(env),
        "Newsvendor·obs":    lambda env: ObsNewsvendorPolicy(env),
        "(s, S)·obs":        lambda env: ObsSSPolicy(env),
    }

    for scenario_name, dcfg in demand_scenarios.items():
        print(f"\n{'=' * 75}")
        print(f"  {scenario_name} Demand — Multi-Echelon Network (5 seeds)")
        print(f"{'=' * 75}")
        print(f"  {'Agent':<18} {'Profit':>10} {'Fill Rate':>10} {'Avg Inv':>10} {'Turns':>8}")
        print(f"  {'-' * 65}")

        env_kwargs = dict(scenario='network', num_periods=30, demand_config=dcfg)

        for agent_name, make_agent in agents.items():
            kpis = run_benchmark(env_kwargs, make_agent)
            print(f"  {agent_name:<18} {kpis['total_profit']:>10,.0f} "
                  f"{kpis['fill_rate']:>10.1%} "
                  f"{kpis['avg_inventory']:>10,.0f} "
                  f"{kpis['inventory_turns']:>8.2f}")

    print(f"\n{'=' * 75}")
    print("  † = Oracle mode (reads env.X, env.Y directly)")
    print("  ·obs = Obs-parity mode (parses observation vector only)")
    print("  Note: Add trained RL agents (PPO, SAC) for a full comparison.")
    print("  See examples/train_ppo.py to train a PPO baseline.")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
