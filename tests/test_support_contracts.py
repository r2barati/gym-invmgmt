import importlib
import warnings

import numpy as np
import pytest

import gym_invmgmt
from gym_invmgmt import CoreEnv
from gym_invmgmt.demand_engine import DemandEngine
from gym_invmgmt.utils import compute_bullwhip, compute_kpis


def test_demand_type_aliases_enable_composable_effects():
    alias_engine = DemandEngine({"type": "trend+seasonal+shock", "base_mu": 20})
    explicit_engine = DemandEngine({"effects": ["trend", "seasonal", "shock"], "base_mu": 20})

    assert alias_engine.effects == ["trend", "seasonal", "shock"]
    assert alias_engine.get_current_mu(20) == pytest.approx(explicit_engine.get_current_mu(20))
    assert alias_engine.get_current_mu(20) > alias_engine.get_current_mu(0)


def test_demand_engine_rejects_unknown_effect_aliases():
    with pytest.raises(ValueError, match="Unknown demand type"):
        DemandEngine({"type": "trend+holiday"})

    with pytest.raises(ValueError, match="Unknown demand effect"):
        DemandEngine({"effects": ["trend", "holiday"]})


def test_goodwill_bounds_are_configurable():
    engine = DemandEngine(
        {
            "use_goodwill": True,
            "gw_growth": 2.0,
            "gw_decay": 0.1,
            "gw_cap": 1.2,
            "gw_floor": 0.5,
        }
    )

    engine.update_goodwill(0.0)
    assert engine.sentiment == pytest.approx(1.2)
    engine.update_goodwill(1.0)
    assert engine.sentiment == pytest.approx(0.5)


def test_kpi_helpers_require_completed_episode_unless_partial_requested():
    env = CoreEnv(scenario="serial", num_periods=3)
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape))

    with pytest.raises(ValueError, match="partial=True"):
        compute_kpis(env)
    with pytest.raises(ValueError, match="partial=True"):
        compute_bullwhip(env)

    partial_kpis = compute_kpis(env, partial=True)
    partial_bullwhip = compute_bullwhip(env, partial=True)
    assert partial_kpis["total_demand"] >= 0.0
    assert "bullwhip_ratio" in partial_bullwhip

    while env.period < env.num_periods:
        env.step(np.zeros(env.action_space.shape))

    assert compute_kpis(env)["total_demand"] >= partial_kpis["total_demand"]
    assert "bullwhip_ratio" in compute_bullwhip(env)


def test_gym_registration_is_reload_safe():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(gym_invmgmt)

    override_warnings = [warning for warning in caught if "Overriding environment GymInvMgmt" in str(warning.message)]
    assert override_warnings == []


def test_plot_network_includes_market_level(tmp_path):
    env = CoreEnv(scenario="serial", num_periods=3)
    fig = env.plot_network(save_path=str(tmp_path / "serial_topology.png"))

    labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    assert "Market" in labels
