"""Shared utilities for gym-invmgmt.

This module contains:
- ``assign_env_config``: Apply kwargs / Ray-style env_config to an environment.
- ``DIST_REGISTRY``: Mapping of distribution names to scipy.stats objects.
- ``run_episode``: Run a full episode with a given policy function.
- ``compute_kpis``: Extract standard supply-chain KPIs from a completed episode.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import norm, poisson, uniform

# ── Distribution Registry ─────────────────────────────────────────────
# Maps human-readable names (used in YAML configs) to scipy distribution
# objects.  Extend this dict to support additional distributions.

DIST_REGISTRY: dict[str, Any] = {
    "poisson": poisson,
    "normal": norm,
    "uniform": uniform,
}


# ── Config Helper ─────────────────────────────────────────────────────


def assign_env_config(self: Any, kwargs: dict[str, Any]) -> None:
    """Apply keyword arguments and Ray-style ``env_config`` to an environment.

    For each key in *kwargs*, the corresponding attribute on *self* is set.
    If *self* has an ``env_config`` dict (e.g. when used with Ray RLlib),
    those entries override existing attributes with type coercion.

    Raises:
        AttributeError: If ``env_config`` contains a key that does not
            already exist on *self*.
    """
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, "env_config"):
        for key, value in self.env_config.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), np.ndarray):
                    setattr(self, key, value)
                else:
                    setattr(self, key, type(getattr(self, key))(value))
            else:
                raise AttributeError(f"{self} has no attribute, {key}")


# ── Episode Runner ────────────────────────────────────────────────────


def run_episode(
    env,
    policy: Optional[Callable] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Run a single episode and return a trajectory summary.

    Args:
        env: A Gymnasium-compatible environment (e.g. ``CoreEnv``).
        policy: A callable ``policy(obs) -> action``.  If *None*, uses
            ``env.action_space.sample()`` (random policy).
        seed: Optional seed passed to ``env.reset(seed=...)``.

    Returns:
        A dict containing:

        - ``observations``: list of observations (length T+1, includes reset obs)
        - ``actions``: list of actions taken (length T)
        - ``rewards``: list of step rewards (length T)
        - ``infos``: list of info dicts (length T)
        - ``total_reward``: float, sum of all rewards
        - ``steps``: int, number of steps taken

    Example::

        from gym_invmgmt import CoreEnv
        from gym_invmgmt.utils import run_episode

        env = CoreEnv(scenario='serial', num_periods=30)
        result = run_episode(env, seed=42)
        print(f"Total reward: {result['total_reward']:.2f}")
    """
    if policy is None:
        policy = lambda obs: env.action_space.sample()  # noqa: E731

    obs, info = env.reset(seed=seed)

    observations = [obs]
    actions = []
    rewards = []
    infos = []

    done = False
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        done = terminated or truncated

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "infos": infos,
        "total_reward": float(np.sum(rewards)),
        "steps": len(actions),
    }


# ── KPI Computation ──────────────────────────────────────────────────


def compute_kpis(env, *, partial: bool = False) -> dict[str, float]:
    """Compute standard supply-chain KPIs from a completed episode.

    By default, this must be called **after** an episode has finished (i.e.
    after the final ``env.step()`` returns ``truncated=True``).  Reads the
    trajectory matrices directly from the unwrapped ``CoreEnv``.  Pass
    ``partial=True`` to compute streaming/in-progress KPIs explicitly.

    Returns:
        A dict containing:

        - ``total_profit``: Sum of ``P[0:T, :]`` across all nodes and periods.
        - ``avg_step_profit``: ``total_profit / T``.
        - ``total_demand``: ``sum(D[0:T, :])`` — total demand realized across
          all retail links.
        - ``total_sold``: ``sum(S[0:T, retail_edges])`` — actual retail sales
          from the **S matrix** (flow variable). This is **not** derived from
          backlog arithmetic.
        - ``fill_rate``: ``total_sold / total_demand`` (1.0 = perfect service,
          0.0 = no demand fulfilled). Always in ``[0, 1]``.
        - ``total_backlog``: ``sum(U[T-1, :])`` — **final-period** standing
          backlog (stock variable, not cumulative unit-periods).
        - ``avg_inventory``: Mean of ``X[1:T+1, main_nodes]`` — post-delivery
          on-hand inventory averaged across steps and managed nodes.
        - ``inventory_turns``: ``total_sold / avg_inventory`` (higher = leaner).
        - ``final_sentiment``: Goodwill sentiment at episode end (from
          ``demand_engine``).

    Example::

        from gym_invmgmt import CoreEnv
        from gym_invmgmt.utils import run_episode, compute_kpis

        env = CoreEnv(scenario='network', num_periods=30)
        run_episode(env, seed=42)
        kpis = compute_kpis(env)
        for k, v in kpis.items():
            print(f"  {k}: {v:.4f}")
    """
    # Unwrap to get raw CoreEnv regardless of wrapper stack
    base = env.unwrapped

    T = base.period  # number of steps actually taken
    if T == 0:
        raise ValueError("compute_kpis() requires a completed episode. " "No steps have been taken yet.")
    if not partial and T < base.num_periods:
        raise ValueError(
            "compute_kpis() requires a completed episode by default. " "Pass partial=True to compute in-progress KPIs."
        )

    # Profit
    step_profits = np.sum(base.P[:T, :], axis=1)
    total_profit = float(np.sum(step_profits))
    avg_step_profit = total_profit / T

    # Demand & fulfillment
    total_demand = float(np.sum(base.D[:T, :]))

    # total_sold: sum actual retail sales from the S matrix.
    # S[t, net_idx] = min(effective_demand, available_inventory) on retail edges.
    # NOTE: S also records inter-node shipments; we only want retail→market sales.
    retail_sales = 0.0
    for j, k in base.network.retail_links:
        net_idx = base.network.network_map[(j, k)]
        retail_sales += float(np.sum(base.S[:T, net_idx]))
    total_sold = retail_sales

    # Backlog: final standing backlog (units still unfulfilled at end of episode)
    total_backlog = float(np.sum(base.U[T - 1, :]))
    fill_rate = total_sold / total_demand if total_demand > 0 else 1.0

    # Inventory
    main_indices = [base.network.node_map[n] for n in base.network.main_nodes]
    # Average over steps 1..T (post-fulfillment inventory)
    avg_inventory = float(np.mean(base.X[1 : T + 1, main_indices]))
    inventory_turns = total_sold / avg_inventory if avg_inventory > 0 else float("inf")

    # Goodwill
    final_sentiment = float(base.demand_engine.sentiment)

    return {
        "total_profit": total_profit,
        "avg_step_profit": avg_step_profit,
        "total_demand": total_demand,
        "total_sold": total_sold,
        "fill_rate": fill_rate,
        "total_backlog": total_backlog,
        "avg_inventory": avg_inventory,
        "inventory_turns": inventory_turns,
        "final_sentiment": final_sentiment,
    }


# ── Bullwhip Effect Measurement ──────────────────────────────────────


def compute_bullwhip(env, *, partial: bool = False) -> dict[str, Any]:
    """Compute the bullwhip effect from a completed episode.

    The **bullwhip effect** is the amplification of order variance as
    it propagates upstream through a supply chain.  It is measured
    as the coefficient of variation (CV = std / mean) of ordering
    quantities on each reorder link.

    A bullwhip ratio > 1.0 between an upstream link and its corresponding
    downstream link indicates demand amplification (the classic bullwhip).

    By default, must be called **after** ``env.step()`` has returned
    ``truncated=True``. Pass ``partial=True`` to compute in-progress metrics.

    Returns:
        A dict containing:

        - ``per_link_cv``: dict mapping ``(supplier, buyer)`` → float CV.
        - ``demand_cv``: float, CV of realized end-customer demand.
        - ``max_cv``: float, highest CV across all reorder links.
        - ``bullwhip_ratio``: float, ``max_cv / demand_cv`` (> 1 = amplification).

    Reference:
        Inspired by the MADRL Serial codebase (Liu et al., 2022) which
        computes ``std(action_history) / mean(action_history)`` per echelon.
    """
    base = env.unwrapped
    T = base.period
    if T == 0:
        raise ValueError("compute_bullwhip() requires a completed episode. " "No steps have been taken yet.")
    if not partial and T < base.num_periods:
        raise ValueError(
            "compute_bullwhip() requires a completed episode by default. "
            "Pass partial=True to compute in-progress metrics."
        )

    per_link_cv: dict[tuple, float] = {}
    for edge in base.network.reorder_links:
        idx = base.network.reorder_map[edge]
        orders = base.action_log[:T, idx]
        mean_val = float(np.mean(orders))
        if mean_val > 1e-8:
            per_link_cv[edge] = float(np.std(orders) / mean_val)
        else:
            per_link_cv[edge] = 0.0

    # Demand CV (end-customer)
    all_demand = base.D[:T, :].sum(axis=1)  # aggregate across retail links
    demand_mean = float(np.mean(all_demand))
    demand_cv = float(np.std(all_demand) / demand_mean) if demand_mean > 1e-8 else 0.0

    max_cv = max(per_link_cv.values()) if per_link_cv else 0.0
    bullwhip_ratio = max_cv / demand_cv if demand_cv > 1e-8 else 0.0

    return {
        "per_link_cv": per_link_cv,
        "demand_cv": demand_cv,
        "max_cv": max_cv,
        "bullwhip_ratio": bullwhip_ratio,
    }
