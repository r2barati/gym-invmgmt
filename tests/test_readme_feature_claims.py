import json
import warnings

import gymnasium as gym
import numpy as np
import pytest

import gym_invmgmt
from gym_invmgmt import (
    CoreEnv,
    EpisodeLoggerWrapper,
    IntegerActionWrapper,
    compute_kpis,
    external_series_spec,
    run_episode,
    write_metadata,
)
from gym_invmgmt.demand_engine import DemandEngine


def test_registered_envs_match_readme_table_shapes_and_defaults():
    cases = [
        ("GymInvMgmt/MultiEchelon-v0", (11,), (71,), "network"),
        ("GymInvMgmt/Serial-v0", (3,), (15,), "serial"),
    ]

    for env_id, action_shape, obs_shape, scenario in cases:
        env = gym.make(env_id)
        obs, _ = env.reset(seed=42)
        base = env.unwrapped

        assert base.env_kwargs["scenario"] == scenario
        assert base.num_periods == 30
        assert base.demand_engine.type == "stationary"
        assert base.demand_engine.base_mu == 20
        assert base.demand_engine.use_goodwill is False
        assert env.action_space.shape == action_shape
        assert env.observation_space.shape == obs_shape
        assert obs.shape == obs_shape
        assert env.observation_space.contains(obs)
        env.close()


def test_gym_make_accepts_readme_demand_kwargs_and_lost_sales_mode():
    env = gym.make(
        "GymInvMgmt/MultiEchelon-v0",
        demand_config={
            "type": "stationary",
            "base_mu": 20,
            "use_goodwill": False,
            "external_series": np.asarray([2.0, 3.0, 4.0, 5.0]),
            "noise_scale": 0.0,
        },
        num_periods=4,
        backlog=False,
    )

    obs, _ = env.reset(seed=7)
    assert obs.shape == env.observation_space.shape
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))

    base = env.unwrapped
    assert base.backlog is False
    assert base.D[0, 0] == pytest.approx(2.0)
    assert isinstance(float(reward), float)
    assert terminated is False
    assert truncated is False
    env.close()


def test_demand_engine_claimed_modes_are_composable_and_deterministic():
    trend = DemandEngine({"type": "trend", "base_mu": 10.0, "trend_slope": 0.2})
    seasonal = DemandEngine({"type": "seasonal", "base_mu": 10.0, "seasonal_amp": 0.5, "seasonal_freq": np.pi / 2})
    shock = DemandEngine({"type": "shock", "base_mu": 10.0, "shock_time": 3, "shock_mag": 2.5})
    combined = DemandEngine(
        {
            "effects": ["trend", "seasonal", "shock"],
            "base_mu": 10.0,
            "trend_slope": 0.2,
            "seasonal_amp": 0.5,
            "seasonal_freq": np.pi / 2,
            "shock_time": 3,
            "shock_mag": 2.5,
        }
    )
    external = DemandEngine({"external_series": np.asarray([0.0, 4.0]), "noise_scale": 0.0})

    assert trend.get_current_mu(2) == pytest.approx(14.0)
    assert seasonal.get_current_mu(1) == pytest.approx(15.0)
    assert shock.get_current_mu(2) == pytest.approx(10.0)
    assert shock.get_current_mu(3) == pytest.approx(25.0)
    assert combined.get_current_mu(3) == pytest.approx(10.0 * 1.6 * 0.5 * 2.5)
    assert external.get_current_mu(0) == pytest.approx(0.0)
    assert external.get_current_mu(99) == pytest.approx(4.0)


def test_external_series_spec_write_metadata_and_core_env_round_trip(tmp_path):
    spec = external_series_spec(
        [2.0, 4.0, 6.0],
        scenario="serial",
        scale_to_mean=12.0,
        metadata={"source": "unit-test"},
    )
    metadata_path = tmp_path / "metadata.json"
    write_metadata(spec.metadata, metadata_path)

    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["adapter"] == "external_series"
    assert metadata["source"] == "unit-test"

    env = CoreEnv(**spec.core_env_kwargs(), num_periods=3)
    env.reset(seed=0)
    for _ in range(3):
        env.step(np.zeros(env.action_space.shape))

    np.testing.assert_allclose(env.D[:, 0], [6.0, 12.0, 18.0])


def test_integer_action_wrapper_and_rescale_action_examples_hit_valid_core_actions():
    raw_env = CoreEnv(scenario="serial", num_periods=1)
    int_env = IntegerActionWrapper(raw_env)
    int_env.reset(seed=0)
    action = np.asarray([1.2, 2.6, 3.5], dtype=np.float32)
    int_env.step(action)
    np.testing.assert_allclose(raw_env.action_log[0], np.round(action))

    base_env = CoreEnv(scenario="serial", num_periods=1)
    scaled_env = gym.wrappers.RescaleAction(base_env, min_action=-1.0, max_action=1.0)
    obs, _ = scaled_env.reset(seed=0)
    assert scaled_env.action_space.low.min() == pytest.approx(-1.0)
    assert scaled_env.action_space.high.max() == pytest.approx(1.0)
    obs, *_ = scaled_env.step(np.zeros(scaled_env.action_space.shape, dtype=np.float32))
    assert scaled_env.observation_space.contains(obs)
    assert np.all(base_env.action_log[0] >= base_env.action_space.low)
    assert np.all(base_env.action_log[0] <= base_env.action_space.high)


def test_episode_logger_file_contains_documented_trajectory_matrices(tmp_path):
    base_env = CoreEnv(scenario="serial", num_periods=3)
    env = EpisodeLoggerWrapper(base_env, log_dir=str(tmp_path), run_name="readme")
    env.reset(seed=0)
    for _ in range(base_env.num_periods):
        env.step(np.zeros(env.action_space.shape))

    [log_file] = list(tmp_path.glob("readme_ep*.npz"))
    with np.load(log_file) as data:
        expected_keys = {
            "inventory_X",
            "pipeline_Y",
            "orders_R",
            "sales_S",
            "demand_D",
            "unfulfilled_U",
            "profit_P",
            "actions",
            "goodwill_GW",
            "episode_rewards",
        }
        assert expected_keys.issubset(data.files)
        assert data["inventory_X"].shape[0] == base_env.num_periods + 1
        assert data["demand_D"].shape[0] == base_env.num_periods
        assert data["actions"].shape == (base_env.num_periods, base_env.action_space.shape[0])


def test_rgb_array_render_mode_returns_image_array():
    env = CoreEnv(scenario="serial", num_periods=2, render_mode="rgb_array")
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape))

    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert frame.size > 0
    assert frame.max() > frame.min()


def test_public_run_episode_and_compute_kpis_work_from_top_level_exports():
    env = CoreEnv(scenario="serial", num_periods=4)
    result = run_episode(env, policy=lambda obs: np.zeros(env.action_space.shape), seed=0)
    kpis = compute_kpis(env)

    assert result["steps"] == 4
    assert len(result["observations"]) == 5
    assert "total_reward" in result
    assert kpis["total_demand"] >= 0.0
    assert 0.0 <= kpis["fill_rate"] <= 1.0
    assert "CoreEnv" in gym_invmgmt.__all__
    assert "run_episode" in gym_invmgmt.__all__


def test_stable_baselines3_vectorized_smoke_if_installed():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = CoreEnv(scenario="serial", num_periods=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env, warn=True)

    vec_env = DummyVecEnv(
        [
            lambda: gym.wrappers.RescaleAction(
                CoreEnv(scenario="serial", num_periods=2),
                min_action=-1.0,
                max_action=1.0,
            )
        ]
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    obs = vec_env.reset()
    obs, rewards, dones, infos = vec_env.step(np.zeros((1, 3), dtype=np.float32))

    assert obs.shape == (1, 15)
    assert rewards.shape == (1,)
    assert dones.shape == (1,)
