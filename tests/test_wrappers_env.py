import numpy as np
import pytest

from gym_invmgmt import CoreEnv
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from gym_invmgmt.wrappers.domain_randomization import DomainRandomizationWrapper
from gym_invmgmt.wrappers.graph_only_wrapper import GraphOnlyWrapper
from gym_invmgmt.wrappers.multi_agent import MultiAgentWrapper
from gym_invmgmt.wrappers.residual_action import ProportionalResidualWrapper
from gym_invmgmt.wrappers.residual_graph_wrapper import ResidualGraphWrapper
from gym_invmgmt.wrappers.temporal_frame_stack import TemporalFrameStack


def test_domain_feature_wrapper_v1_v2():
    core_env = CoreEnv(scenario="base")
    n_main = len(
        [
            n
            for n in core_env.network.graph.nodes()
            if n not in core_env.network.market and n not in core_env.network.rawmat
        ]
    )
    base_dim = core_env.observation_space.shape[0]

    env_v1 = DomainFeatureWrapper(core_env, enhanced=False)
    obs_v1, _ = env_v1.reset()
    assert obs_v1.shape == (base_dim + 3 * n_main + 2,)

    core_env = CoreEnv(scenario="base")
    env_v2 = DomainFeatureWrapper(core_env, enhanced=True, grouped=False)
    obs_v2, _ = env_v2.reset()
    assert obs_v2.shape == (core_env.observation_space.shape[0] + 8 * n_main + 10,)


def test_blind_wrapper_shapes():
    core_env = CoreEnv(scenario="base")
    sighted = DomainFeatureWrapper(core_env, is_blind=False)
    blind = DomainFeatureWrapper(core_env, is_blind=True)
    obs_s, _ = sighted.reset()
    obs_b, _ = blind.reset()
    assert obs_s.shape == obs_b.shape
    assert sighted.observation_space.shape == blind.observation_space.shape


def test_temporal_framestack():
    core_env = CoreEnv(scenario="base")
    env = TemporalFrameStack(core_env, n_history=3)
    obs, _ = env.reset()
    assert obs.shape == (core_env.observation_space.shape[0] * 3,)


def test_domain_randomization_sets_demand_engine_parameters():
    env = DomainRandomizationWrapper(
        CoreEnv(scenario="base", num_periods=5),
        effect_options=("seasonal", "shock"),
        effect_prob=1.0,
        goodwill_options=(False,),
        backlog_options=(True,),
        noise_scale_range=(1.0, 1.0),
        external_series_prob=0.0,
        seasonal_amp=0.3,
        seasonal_freq=0.3,
        shock_time=2,
        shock_mag=1.7,
    )

    obs, _ = env.reset(seed=123)
    engine = env.unwrapped.demand_engine

    assert obs.shape == env.observation_space.shape
    assert engine.effects == ["seasonal", "shock"]
    assert engine.use_goodwill is False
    assert engine.seasonal_amp == pytest.approx(0.3)
    assert engine.shock_time == 2
    assert env.unwrapped.backlog is True


def test_graph_only_lost_sales_does_not_subtract_prior_unfulfilled():
    env = GraphOnlyWrapper(CoreEnv(scenario="serial", num_periods=3, backlog=False))
    env.reset(seed=123)
    core = env.unwrapped
    core.period = 1

    retail = next(n for n in env.main_nodes if n in core.network.retail)
    retail_pos = env.main_nodes.index(retail)
    retail_idx = core.network.node_map[retail]
    core.X[1, retail_idx] = 20.0
    core.Y[1, :] = 0.0
    core.U[0, :] = 7.0

    obs = env._compute_graph_obs()
    assert obs[retail_pos] == pytest.approx(20.0)


def test_multi_agent_terminal_and_lost_sales_contracts():
    core_env = CoreEnv(
        scenario="serial",
        num_periods=3,
        backlog=False,
        demand_config={"type": "stationary", "base_mu": 5},
    )
    env = MultiAgentWrapper(core_env)
    obs, _ = env.reset(seed=123)

    for _ in range(core_env.num_periods):
        obs, _, _, truncated, _ = env.step(np.zeros(env.action_space.shape))

    assert truncated
    assert obs.shape == env.observation_space.shape
    assert not np.allclose(obs, 0.0)

    core_env.period = 1
    core_env.U[0, :] = 9.0
    retail = next(n for n in env.agent_nodes if n in core_env.network.retail)
    retail_agent_pos = env.agent_nodes.index(retail)
    local_obs = env._build_local_obs()
    backlog_feature_idx = retail_agent_pos * 5 + 2
    assert local_obs[backlog_feature_idx] == pytest.approx(0.0)


def test_residual_graph_observation_space_matches_heuristic_features():
    class OnesHeuristic:
        def get_action(self, obs, t):
            return np.ones(len(core_env.network.reorder_links))

    core_env = CoreEnv(scenario="base", num_periods=3)
    env = ResidualGraphWrapper(core_env, heuristic_agent=OnesHeuristic())
    obs, _ = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_residual_actions_are_clipped():
    class HugeHeuristic:
        def __init__(self, core):
            self.core = core

        def get_action(self, obs, t):
            return self.core.action_space.high * 10.0

    core_env = CoreEnv(scenario="base", num_periods=3)
    graph_env = ResidualGraphWrapper(core_env, heuristic_agent=HugeHeuristic(core_env))
    graph_env.reset(seed=123)
    graph_env.step(np.ones(graph_env.action_space.shape) * 1e6)
    assert np.all(core_env.action_log[0] <= core_env.action_space.high)

    core_env = CoreEnv(scenario="base", num_periods=3)
    prop_env = ProportionalResidualWrapper(core_env, heuristic_agent=HugeHeuristic(core_env))
    prop_env.reset(seed=123)
    prop_env.step(np.ones(prop_env.action_space.shape) * 1e6)
    assert np.all(core_env.action_log[0] <= core_env.action_space.high)
