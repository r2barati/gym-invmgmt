import numpy as np
import pytest

from gym_invmgmt import CoreEnv


@pytest.mark.parametrize("topology", ["base", "serial", "network"])
def test_env_topology_initialization(topology):
    """Test that the environment initializes properly across supported topologies."""
    env = CoreEnv(scenario=topology)
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape

    # Take a step
    action = env.action_space.sample()
    assert action.shape == env.action_space.shape

    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_kpi_array_dimensions():
    """Test that internal state tracking arrays have the correct dimensions."""
    env = CoreEnv(scenario="base", num_periods=30)
    env.reset(seed=42)

    # Take a few steps
    for _ in range(5):
        env.step(env.action_space.sample())

    num_nodes = len(env.network.graph.nodes())
    num_edges = len(env.network.graph.edges())
    num_retail = len(env.network.retail)

    # Demand arrays should be (T, num_retail)
    assert env.D.ndim == 2
    assert env.D.shape[1] == num_retail

    # Inventory state array should be (T, num_nodes)
    assert env.X.ndim == 2
    assert env.X.shape[1] == num_nodes

    # Pipeline/transit array should be (_, M) where M <= num_edges
    assert env.Y.ndim == 2
    assert env.Y.shape[1] <= num_edges

    # Unmet demand (backlog/lost sales) array
    assert env.U.ndim == 2
    assert env.U.shape[1] == num_retail


def test_terminal_observation_reflects_final_state():
    """The final transition should return the state after the final action."""
    env = CoreEnv(
        scenario="serial",
        num_periods=3,
        demand_config={"type": "stationary", "base_mu": 5},
    )
    obs, _ = env.reset(seed=42)

    for _ in range(env.num_periods):
        previous_obs = obs.copy()
        obs, _, _, truncated, _ = env.step(np.zeros(env.action_space.shape))

    assert truncated
    assert env.period == env.num_periods
    assert obs.shape == env.observation_space.shape
    assert not np.allclose(obs, previous_obs)
    np.testing.assert_allclose(obs, env.state)
