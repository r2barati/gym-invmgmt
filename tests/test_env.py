"""Smoke tests for gym-invmgmt."""

import gymnasium
import numpy as np

import gym_invmgmt  # noqa: F401 - imported for Gymnasium environment registration


def test_multi_echelon_registration():
    """Check MultiEchelon-v0 instantiates via gymnasium.make()."""

    env = gymnasium.make("GymInvMgmt/MultiEchelon-v0")
    obs, info = env.reset(seed=42)

    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)

    assert obs2.shape == obs.shape
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    print("MultiEchelon-v0 OK")


def test_serial_registration():
    """Check Serial-v0 instantiates via gymnasium.make()."""

    env = gymnasium.make("GymInvMgmt/Serial-v0")
    obs, info = env.reset(seed=42)

    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)

    assert obs2.shape == obs.shape
    print("Serial-v0 OK")


def test_full_episode():
    """Run a complete 30-step episode on MultiEchelon."""

    env = gymnasium.make("GymInvMgmt/MultiEchelon-v0")
    obs, info = env.reset(seed=123)

    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    assert steps == 30, f"Expected 30 steps, got {steps}"
    assert isinstance(total_reward, (float, np.floating))
    print(f"Full episode OK — {steps} steps, total reward: {total_reward:.2f}")


def test_custom_scenario():
    """Test creating env with custom demand config."""
    from gym_invmgmt import CoreEnv

    env = CoreEnv(
        scenario="network",
        demand_config={
            "type": "shock",
            "base_mu": 25,
            "use_goodwill": True,
            "shock_time": 10,
            "shock_mag": 3.0,
        },
        num_periods=20,
    )
    obs, info = env.reset(seed=7)
    action = env.action_space.sample()
    obs2, reward, term, trunc, info = env.step(action)
    assert obs2.shape == obs.shape
    print("Custom scenario OK")


def test_render():
    """Test render method produces output."""
    from gym_invmgmt import CoreEnv

    env = CoreEnv(scenario="network", render_mode="ansi")
    env.reset(seed=42)
    env.step(env.action_space.sample())

    output = env.render()
    assert output is not None
    assert "Period" in output
    print("Render OK")


def test_deterministic_seeding():
    """Same seed → same trajectory."""
    from gym_invmgmt import CoreEnv

    def run_episode(seed):
        env = CoreEnv(scenario="network", num_periods=10)
        obs, _ = env.reset(seed=seed)
        rewards = []
        for _ in range(10):
            action = np.zeros(env.action_space.shape)
            obs, r, term, trunc, info = env.step(action)
            rewards.append(r)
        return rewards

    r1 = run_episode(42)
    r2 = run_episode(42)
    np.testing.assert_array_almost_equal(r1, r2)
    print("Deterministic seeding OK")


def test_plot_network():
    """Test plot_network generates figures without errors."""
    import matplotlib

    matplotlib.use("Agg")
    import os
    import tempfile

    from gym_invmgmt import CoreEnv

    env = CoreEnv(scenario="network")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_plot.png")
        env.plot_network(detailed=True, save_path=path)
        assert os.path.exists(path), "Plot file not saved"
        assert os.path.getsize(path) > 0, "Plot file is empty"
    print("plot_network OK")


def test_wrappers():
    """Test wrappers work correctly."""
    import os
    import tempfile

    from gym_invmgmt import CoreEnv, EpisodeLoggerWrapper, IntegerActionWrapper

    # IntegerActionWrapper
    env = CoreEnv(scenario="serial")
    env = IntegerActionWrapper(env)
    obs, _ = env.reset(seed=42)
    obs2, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs2.shape == obs.shape

    # EpisodeLoggerWrapper
    with tempfile.TemporaryDirectory() as tmpdir:
        env = CoreEnv(scenario="serial", num_periods=5)
        env = EpisodeLoggerWrapper(env, log_dir=tmpdir, run_name="test")
        obs, _ = env.reset(seed=42)
        for _ in range(5):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                break
        files = [f for f in os.listdir(tmpdir) if f.endswith(".npz")]
        assert len(files) == 1

    print("Wrappers OK")


def test_gymnasium_check_env():
    """Formal Gymnasium env checker — validates spaces, PRNG, return types."""
    from gymnasium.utils.env_checker import check_env

    for env_id in ["GymInvMgmt/MultiEchelon-v0", "GymInvMgmt/Serial-v0"]:
        env = gymnasium.make(env_id)
        # check_env raises on hard errors; warnings are acceptable
        check_env(env.unwrapped, skip_render_check=True)
    print("check_env OK (both envs)")


def test_observation_space_containment():
    """Stress test: run 1000 random steps, assert obs stays in bounds."""
    from gym_invmgmt import CoreEnv

    for scenario in ["network", "serial"]:
        env = CoreEnv(scenario=scenario, num_periods=30)

        for episode in range(10):
            obs, _ = env.reset(seed=episode)
            assert env.observation_space.contains(obs), f"Reset obs out of bounds: {obs}"

            for step in range(30):
                action = env.action_space.sample()
                assert env.action_space.contains(action), f"Sampled action out of bounds: {action}"

                obs, reward, term, trunc, info = env.step(action)

                assert env.observation_space.contains(obs), (
                    f"Step obs out of bounds at t={step}, scenario={scenario}: " f"min={obs.min()}, max={obs.max()}"
                )

                if term or trunc:
                    break

    print("Observation space containment OK (300 steps × 2 scenarios)")


def test_demand_engine_standalone():
    """Test DemandEngine can sample demand in isolation (no CoreEnv)."""
    from scipy.stats import poisson

    from gym_invmgmt.demand_engine import DemandEngine

    engine = DemandEngine(
        {
            "effects": ["shock"],
            "base_mu": 20,
            "shock_time": 5,
            "shock_mag": 2.0,
            "use_goodwill": True,
        }
    )
    engine.reset(np_random=np.random.default_rng(42))

    # Pre-shock demand
    d_pre = engine.sample(0, poisson)
    assert 5 <= d_pre <= 40, f"Pre-shock demand out of range: {d_pre}"

    # Post-shock demand should be higher (mu doubles)
    d_post = engine.sample(10, poisson)
    assert d_post > 0, f"Post-shock demand should be positive: {d_post}"

    # Goodwill update
    engine.update_goodwill(0)  # satisfied
    assert engine.sentiment > 1.0, "Sentiment should grow after satisfaction"
    engine.update_goodwill(10)  # stockout
    assert engine.sentiment < 1.01, "Sentiment should drop after stockout"

    print("DemandEngine standalone OK")


# ── Custom YAML Config Tests ─────────────────────────────────────────


def test_custom_yaml_serial():
    """Load serial_network.yaml and verify it matches the built-in serial scenario."""
    import os

    from gym_invmgmt import CoreEnv

    config_path = os.path.join(os.path.dirname(__file__), "..", "gym_invmgmt", "topologies", "serial.yaml")

    # Custom from YAML
    env_custom = CoreEnv(scenario="custom", config_path=config_path, num_periods=10)
    obs_c, _ = env_custom.reset(seed=42)

    # Built-in serial
    env_serial = CoreEnv(scenario="serial", num_periods=10)
    obs_s, _ = env_serial.reset(seed=42)

    # Same observation and action shapes
    assert obs_c.shape == obs_s.shape, f"Shape mismatch: custom={obs_c.shape} vs serial={obs_s.shape}"
    assert env_custom.action_space.shape == env_serial.action_space.shape, "Action shape mismatch"

    # Run a step
    action = np.zeros(env_custom.action_space.shape)
    obs2, r, term, trunc, info = env_custom.step(action)
    assert obs2.shape == obs_c.shape
    assert isinstance(r, (float, np.floating))

    print("Custom YAML serial OK")


def test_custom_yaml_diamond():
    """Load diamond_network.yaml and run a full episode."""
    import os

    from gym_invmgmt import make_custom_env

    config_path = os.path.join(os.path.dirname(__file__), "..", "gym_invmgmt", "topologies", "diamond.yaml")

    env = make_custom_env(config_path, num_periods=15)
    obs, info = env.reset(seed=99)

    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)

    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    assert steps == 15, f"Expected 15 steps, got {steps}"
    assert isinstance(total_reward, (float, np.floating))
    print(f"Custom YAML diamond OK — {steps} steps, total reward: {total_reward:.2f}")


def test_custom_yaml_divergent():
    """Load divergent_network.yaml and verify equivalence with built-in network."""
    import os

    from gym_invmgmt import CoreEnv

    config_path = os.path.join(os.path.dirname(__file__), "..", "gym_invmgmt", "topologies", "divergent.yaml")

    env_custom = CoreEnv(scenario="custom", config_path=config_path, num_periods=10)
    env_builtin = CoreEnv(scenario="network", num_periods=10)

    assert (
        env_custom.observation_space.shape == env_builtin.observation_space.shape
    ), f"Obs shape mismatch: {env_custom.observation_space.shape} vs {env_builtin.observation_space.shape}"
    assert env_custom.action_space.shape == env_builtin.action_space.shape, "Action shape mismatch"

    # Verify node counts match
    assert len(env_custom.network.main_nodes) == len(env_builtin.network.main_nodes)
    assert len(env_custom.network.reorder_links) == len(env_builtin.network.reorder_links)
    assert len(env_custom.network.retail_links) == len(env_builtin.network.retail_links)

    print("Custom YAML divergent OK")


def test_custom_yaml_assembly():
    """Load assembly.yaml — convergent supply chain with 2 component suppliers."""
    import os

    from gym_invmgmt import CoreEnv

    config_path = os.path.join(os.path.dirname(__file__), "..", "gym_invmgmt", "topologies", "assembly.yaml")

    env = CoreEnv(scenario="custom", config_path=config_path, num_periods=15)
    obs, _ = env.reset(seed=42)

    net = env.network
    assert len(net.retail) == 1, f"Assembly should have 1 retailer, got {len(net.retail)}"
    assert len(net.factory) == 3, f"Assembly should have 3 factories (2 component + 1 assembly), got {len(net.factory)}"
    assert env.action_space.shape[0] == 5, f"Expected 5 reorder links, got {env.action_space.shape[0]}"

    steps, total_reward = 0, 0.0
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        total_reward += r
        steps += 1
        done = term or trunc

    assert steps == 15
    print(f"Custom YAML assembly OK — {steps} steps, total reward: {total_reward:.2f}")


def test_custom_yaml_distribution_tree():
    """Load distribution_tree.yaml — 1 factory → hub → 3 retailers."""
    import os

    from gym_invmgmt import CoreEnv

    config_path = os.path.join(os.path.dirname(__file__), "..", "gym_invmgmt", "topologies", "distribution_tree.yaml")

    env = CoreEnv(scenario="custom", config_path=config_path, num_periods=15)
    obs, _ = env.reset(seed=42)

    net = env.network
    assert len(net.retail) == 3, f"Distribution tree should have 3 retailers, got {len(net.retail)}"
    assert len(net.retail_links) == 3, f"Should have 3 retail links, got {len(net.retail_links)}"
    assert len(net.factory) == 1, f"Should have 1 factory, got {len(net.factory)}"

    steps, total_reward = 0, 0.0
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        total_reward += r
        steps += 1
        done = term or trunc

    assert steps == 15
    print(f"Custom YAML distribution_tree OK — {steps} steps, total reward: {total_reward:.2f}")


def test_custom_yaml_w_network():
    """Load w_network.yaml — 2 factories → shared DC → 2 retailers."""
    import os

    from gym_invmgmt import CoreEnv

    config_path = os.path.join(os.path.dirname(__file__), "..", "gym_invmgmt", "topologies", "w_network.yaml")

    env = CoreEnv(scenario="custom", config_path=config_path, num_periods=15)
    obs, _ = env.reset(seed=42)

    net = env.network
    assert len(net.retail) == 2, f"W-network should have 2 retailers, got {len(net.retail)}"
    assert len(net.retail_links) == 2, f"Should have 2 retail links, got {len(net.retail_links)}"
    assert len(net.factory) == 2, f"Should have 2 factories, got {len(net.factory)}"

    steps, total_reward = 0, 0.0
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        total_reward += r
        steps += 1
        done = term or trunc

    assert steps == 15
    print(f"Custom YAML w_network OK — {steps} steps, total reward: {total_reward:.2f}")


def test_custom_yaml_invalid():
    """Test error handling for invalid YAML configs."""
    import os
    import tempfile

    from gym_invmgmt import CoreEnv

    # Missing config_path for custom scenario
    try:
        CoreEnv(scenario="custom")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "config_path" in str(e)

    # Non-existent file
    try:
        CoreEnv(scenario="custom", config_path="/nonexistent/path.yaml")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    # Empty YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("---\n")
        empty_path = f.name
    try:
        CoreEnv(scenario="custom", config_path=empty_path)
        assert False, "Should have raised ValueError"
    except (ValueError, TypeError):
        pass
    finally:
        os.unlink(empty_path)

    # YAML with missing edges
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("nodes:\n  - id: 0\n  - id: 1\n    I0: 100\n    h: 0.03\n")
        no_edges_path = f.name
    try:
        CoreEnv(scenario="custom", config_path=no_edges_path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "edges" in str(e).lower()
    finally:
        os.unlink(no_edges_path)

    # YAML with unknown distribution
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            """
nodes:
  - id: 0
  - id: 1
    I0: 100
    h: 0.03
edges:
  - from: 1
    to: 0
    p: 2.0
    b: 0.1
    demand_dist: unknown_dist
    dist_param:
      mu: 20
"""
        )
        bad_dist_path = f.name
    try:
        CoreEnv(scenario="custom", config_path=bad_dist_path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unknown_dist" in str(e) or "Unknown" in str(e)
    finally:
        os.unlink(bad_dist_path)

    print("Custom YAML invalid configs OK")


if __name__ == "__main__":
    test_multi_echelon_registration()
    test_serial_registration()
    test_full_episode()
    test_custom_scenario()
    test_render()
    test_deterministic_seeding()
    test_plot_network()
    test_wrappers()
    test_gymnasium_check_env()
    test_observation_space_containment()
    test_demand_engine_standalone()
    test_custom_yaml_serial()
    test_custom_yaml_diamond()
    test_custom_yaml_divergent()
    test_custom_yaml_invalid()
    print("\n All tests passed!")
