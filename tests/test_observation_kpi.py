"""Regression tests for observation semantics and KPI computation.

These tests verify the specific fixes from the gym-invmgmt audit:
- Demand observation is lag-1 (realized), not future
- Unfulfilled-demand observation is lag-1 and explicit in the base state
- Pipeline observation uses R (filled orders), not action_log (requested)
- fill_rate KPI uses S (sales matrix), not cumulative backlog arithmetic
"""

import numpy as np
import pytest

from gym_invmgmt import CoreEnv
from gym_invmgmt.utils import compute_kpis

# ── Observation Regression Tests ─────────────────────────────────────


class TestDemandObservation:
    """Verify demand component of observation is lag-1 realized demand."""

    def test_reset_demand_is_zero(self):
        """At reset (t=0), no demand has been realized yet → demand obs = 0."""
        env = CoreEnv(scenario="network", num_periods=10)
        obs, _ = env.reset(seed=42)

        n_retail = len(env.network.retail_links)
        demand_slice = obs[:n_retail]

        np.testing.assert_array_equal(
            demand_slice,
            np.zeros(n_retail),
            err_msg="Reset observation demand should be all zeros (no demand realized yet)",
        )

    def test_step0_demand_matches_D_matrix(self):
        """After step 0, demand obs should equal D[0] (the demand just realized)."""
        env = CoreEnv(scenario="network", num_periods=10)
        env.reset(seed=42)

        action = np.zeros(env.action_space.shape)
        obs, _, _, _, _ = env.step(action)

        n_retail = len(env.network.retail_links)
        demand_in_obs = obs[:n_retail]

        # D[0] was sampled during step 0; now period=1, so obs reads D[0]
        np.testing.assert_array_almost_equal(
            demand_in_obs, env.D[0, :], err_msg="Post-step-0 demand obs should be D[0] (lag-1 realized demand)"
        )

    def test_demand_never_reads_future(self):
        """At every step, demand in obs should equal D[period-1], never D[period]."""
        env = CoreEnv(
            scenario="network",
            num_periods=10,
            demand_config={"type": "stationary", "base_mu": 20, "use_goodwill": False},
        )
        env.reset(seed=42)

        n_retail = len(env.network.retail_links)

        for t in range(10):
            action = np.zeros(env.action_space.shape)
            obs, _, term, trunc, _ = env.step(action)

            if term or trunc:
                # Terminal obs may differ at truncation boundary; skip
                break

            demand_in_obs = obs[:n_retail]
            expected = env.D[t, :]  # D[t] was just realized; period is now t+1

            np.testing.assert_array_almost_equal(
                demand_in_obs, expected, err_msg=f"At period {env.period}, demand obs should be D[{t}]"
            )


class TestPipelineObservation:
    """Verify pipeline uses R (filled) not action_log (requested)."""

    def test_pipeline_zeros_at_reset(self):
        """At reset, no orders placed → pipeline obs = 0."""
        env = CoreEnv(scenario="network", num_periods=10)
        obs, _ = env.reset(seed=42)

        n_retail = len(env.network.retail_links)
        n_main = len(env.network.main_nodes)
        n_extra = env.extra_features_dim

        # Base obs layout:
        # [lagged demand | lagged unfulfilled demand | inventory | pipeline | extras]
        pipeline_start = 2 * n_retail + n_main
        pipeline_slice = obs[pipeline_start:-n_extra]

        np.testing.assert_array_equal(
            pipeline_slice, np.zeros(len(pipeline_slice)), err_msg="Reset pipeline observation should be all zeros"
        )

    def test_pipeline_uses_R_not_action_log(self):
        """Pipeline obs should reflect filled orders (R), not raw requests (action_log).

        Strategy: Issue a huge order that will be constrained by capacity/inventory.
        Then verify that R (filled) differs from action_log (requested) on at least
        one constrained link, and that the pipeline obs matches R.
        """
        env = CoreEnv(scenario="network", num_periods=10)
        env.reset(seed=42)

        # Issue a very large order (will be constrained by inventory/capacity)
        huge_action = env.action_space.high * 10
        obs, _, _, _, _ = env.step(huge_action)

        # R should be constrained below action_log on at least one link
        # (factory and distributor links have inventory/capacity constraints)
        action_log_0 = env.action_log[0, :]
        r_filled_0 = env.R[0, :]

        # At least one link should have R < action_log (unless all are raw material)
        constrained = np.any(r_filled_0 < action_log_0)
        if constrained:
            # Verify the pipeline observation uses R values, not action_log values
            # Build expected pipeline from R (same logic as _update_state)
            expected_pipeline = []
            for i, edge_tuple in enumerate(env.network.reorder_links):
                L = env.network.lead_times[edge_tuple]
                if L == 0:
                    continue
                arrivals = np.zeros(L)
                for k in range(L):
                    order_time = env.period - L + k
                    if 0 <= order_time < env.period:
                        arrivals[k] = env.R[order_time, i]
                expected_pipeline.append(arrivals)

            if expected_pipeline:
                expected_pipeline = np.hstack(expected_pipeline)
                n_retail = len(env.network.retail_links)
                n_main = len(env.network.main_nodes)
                n_extra = env.extra_features_dim
                pipeline_start = 2 * n_retail + n_main
                pipeline_in_obs = obs[pipeline_start:-n_extra]

                np.testing.assert_array_almost_equal(
                    pipeline_in_obs,
                    expected_pipeline,
                    err_msg="Pipeline obs should match R-based arrivals, not action_log",
                )

    def test_pipeline_sums_match_Y(self):
        """Total pipeline obs per link should equal Y (running balance) for that link."""
        env = CoreEnv(scenario="network", num_periods=10)
        env.reset(seed=42)

        # Run a few steps to populate pipeline
        for _ in range(3):
            action = env.action_space.sample()
            env.step(action)

        # Now check: sum of arrival slots per link == Y[period, link]
        for i, edge_tuple in enumerate(env.network.reorder_links):
            L = env.network.lead_times[edge_tuple]
            if L == 0:
                continue

            # Sum arrivals for this link from R matrix
            total_from_R = 0
            for k in range(L):
                order_time = env.period - L + k
                if 0 <= order_time < env.period:
                    total_from_R += env.R[order_time, i]

            # Should match Y
            np.testing.assert_almost_equal(
                total_from_R,
                env.Y[env.period, i],
                err_msg=f"R-arrival sum ({total_from_R}) != Y[{env.period},{i}] "
                f"({env.Y[env.period, i]}) for link {edge_tuple}",
            )


class TestObservationDimensions:
    """Verify observation structure matches documented layout."""

    @pytest.mark.parametrize("scenario", ["network", "serial"])
    def test_obs_dim_breakdown(self, scenario):
        """obs = [demand | unfulfilled | inventory | pipeline | extra_features]."""
        env = CoreEnv(scenario=scenario, num_periods=5)
        obs, _ = env.reset(seed=42)

        n_retail = len(env.network.retail_links)
        n_main = len(env.network.main_nodes)
        pipeline_len = env.network.pipeline_length
        n_extra = env.extra_features_dim

        expected_dim = 2 * n_retail + n_main + pipeline_len + n_extra
        assert obs.shape[0] == expected_dim, (
            f"Obs dim {obs.shape[0]} != "
            f"demand({n_retail}) + unfulfilled({n_retail}) + main({n_main}) "
            f"+ pipeline({pipeline_len}) + extra({n_extra}) "
            f"= {expected_dim}"
        )


# ── KPI Regression Tests ────────────────────────────────────────────


class TestFillRate:
    """Verify fill_rate KPI uses S (sales matrix) correctly."""

    def test_zero_inventory_zero_action_fill_rate(self):
        """If initial inventory is 0 and agent never orders → fill rate = 0%.

        This regression check ensures fill rate is computed from actual sales
        rather than cumulative backlog unit-periods.
        """
        env = CoreEnv(
            scenario="network",
            num_periods=10,
            demand_config={"type": "stationary", "base_mu": 20, "use_goodwill": False},
        )
        env.reset(seed=42)

        # Force zero initial inventory
        env.X[0, :] = 0

        # Never order anything
        for _ in range(10):
            obs, r, term, trunc, _ = env.step(np.zeros(env.action_space.shape))
            if term or trunc:
                break

        kpis = compute_kpis(env)

        assert kpis["fill_rate"] == 0.0, (
            f"With zero inventory and zero orders, fill_rate should be 0.0, " f"got {kpis['fill_rate']}"
        )
        assert kpis["fill_rate"] >= 0.0, "fill_rate should never be negative when computed from retail sales"
        assert kpis["total_sold"] == 0.0, f"With zero inventory, total_sold should be 0.0, got {kpis['total_sold']}"

    def test_abundant_inventory_fill_rate_100(self):
        """With huge initial inventory, all demand should be met → fill rate ≈ 100%."""
        env = CoreEnv(
            scenario="network",
            num_periods=10,
            demand_config={"type": "stationary", "base_mu": 5, "use_goodwill": False},
        )
        env.reset(seed=42)

        # Set very high initial inventory at all nodes
        env.X[0, :] = 10000

        for _ in range(10):
            action = np.zeros(env.action_space.shape)
            obs, r, term, trunc, _ = env.step(action)
            if term or trunc:
                break

        kpis = compute_kpis(env)

        assert kpis["fill_rate"] == pytest.approx(
            1.0, abs=0.01
        ), f"With abundant inventory, fill_rate should be ~1.0, got {kpis['fill_rate']}"

    def test_fill_rate_equals_sold_over_demand(self):
        """fill_rate should always equal total_sold / total_demand."""
        env = CoreEnv(scenario="network", num_periods=15)
        env.reset(seed=99)

        for _ in range(15):
            action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            if term or trunc:
                break

        kpis = compute_kpis(env)

        if kpis["total_demand"] > 0:
            expected_fr = kpis["total_sold"] / kpis["total_demand"]
            assert kpis["fill_rate"] == pytest.approx(expected_fr), (
                f"fill_rate ({kpis['fill_rate']}) != " f"total_sold/total_demand ({expected_fr})"
            )

    def test_fill_rate_bounded_0_to_1(self):
        """fill_rate should always be in [0, 1]."""
        for seed in range(20):
            env = CoreEnv(scenario="network", num_periods=10)
            env.reset(seed=seed)
            for _ in range(10):
                obs, r, term, trunc, _ = env.step(env.action_space.sample())
                if term or trunc:
                    break

            kpis = compute_kpis(env)
            assert 0.0 <= kpis["fill_rate"] <= 1.0, f"fill_rate out of bounds: {kpis['fill_rate']} (seed={seed})"


class TestKPIConsistency:
    """Verify KPI values are internally consistent."""

    def test_total_sold_matches_S_matrix(self):
        """total_sold should equal sum of S on retail edges."""
        env = CoreEnv(scenario="network", num_periods=10)
        env.reset(seed=42)

        for _ in range(10):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break

        kpis = compute_kpis(env)
        T = env.period

        manual_sold = 0.0
        for j, k in env.network.retail_links:
            net_idx = env.network.network_map[(j, k)]
            manual_sold += float(np.sum(env.S[:T, net_idx]))

        assert kpis["total_sold"] == pytest.approx(
            manual_sold
        ), f"total_sold ({kpis['total_sold']}) != manual S sum ({manual_sold})"

    def test_total_backlog_is_final_period(self):
        """total_backlog should be the final-period standing backlog, not cumulative."""
        env = CoreEnv(scenario="network", num_periods=10)
        env.reset(seed=42)

        for _ in range(10):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break

        kpis = compute_kpis(env)
        T = env.period

        manual_backlog = float(np.sum(env.U[T - 1, :]))
        assert kpis["total_backlog"] == pytest.approx(
            manual_backlog
        ), f"total_backlog ({kpis['total_backlog']}) != U[{T-1}] sum ({manual_backlog})"

    def test_kpis_require_completed_steps(self):
        """compute_kpis should raise ValueError if no steps taken."""
        env = CoreEnv(scenario="network", num_periods=10)
        env.reset(seed=42)

        with pytest.raises(ValueError, match="No steps"):
            compute_kpis(env)


# ── Property Alias Tests ────────────────────────────────────────────


class TestPropertyAliases:
    """Verify @property aliases point to the correct underlying arrays."""

    def test_aliases_are_same_object(self):
        """Property aliases should return the exact same ndarray (not a copy)."""
        env = CoreEnv(scenario="network", num_periods=5)
        env.reset(seed=42)
        env.step(env.action_space.sample())

        assert env.inventory is env.X
        assert env.pipeline is env.Y
        assert env.orders_filled is env.R
        assert env.orders_requested is env.action_log
        assert env.shipments is env.S
        assert env.demand is env.D
        assert env.standing_backlog is env.U
        assert env.profit is env.P

    def test_alias_mutation_propagates(self):
        """Writing through an alias should modify the underlying array."""
        env = CoreEnv(scenario="network", num_periods=5)
        env.reset(seed=42)

        original_val = env.X[0, 0]
        env.inventory[0, 0] = 99999.0
        assert env.X[0, 0] == 99999.0

        # Restore
        env.X[0, 0] = original_val
