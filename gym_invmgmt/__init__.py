"""gym-invmgmt — Multi-Echelon Inventory Management Gymnasium Environment."""

import gymnasium

from gym_invmgmt.core_env import CoreEnv
from gym_invmgmt.data_adapters import (
    DatasetScenarioSpec,
    external_series_spec,
    hierarchy_csv_to_tree_topology_yaml,
    long_demand_csv_to_spec,
    m5_wide_csv_to_spec,
    retail_store_csv_to_star_topology_yaml,
    topology_csvs_to_yaml,
    wide_demand_csv_to_spec,
    write_metadata,
)
from gym_invmgmt.demand_engine import DemandEngine
from gym_invmgmt.network_topology import SupplyChainNetwork
from gym_invmgmt.utils import compute_kpis, run_episode
from gym_invmgmt.visualization import plot_network
from gym_invmgmt.wrappers import EpisodeLoggerWrapper, IntegerActionWrapper


def make_custom_env(config_path: str, **kwargs):
    """Create a CoreEnv from a YAML network config file.

    Args:
        config_path: Path to the YAML network configuration file.
        **kwargs: Additional keyword arguments passed to CoreEnv
                  (e.g. demand_config, num_periods, render_mode).

    Returns:
        A CoreEnv instance with the custom network topology.

    Example::

        env = make_custom_env('examples/configs/diamond_network.yaml', num_periods=30)
        obs, info = env.reset(seed=42)
    """
    return CoreEnv(scenario="custom", config_path=config_path, **kwargs)


def _register_env_once(env_id: str, **kwargs) -> None:
    """Register a Gymnasium env unless another import already did so."""
    if env_id not in gymnasium.envs.registry:
        gymnasium.register(id=env_id, **kwargs)


# ── Register environments with Gymnasium ─────────────────────────────
# Users can create environments via:
#   gymnasium.make("GymInvMgmt/MultiEchelon-v0")
#   gymnasium.make("GymInvMgmt/Serial-v0")

_register_env_once(
    "GymInvMgmt/MultiEchelon-v0",
    entry_point="gym_invmgmt.core_env:CoreEnv",
    kwargs={
        "scenario": "network",
        "demand_config": {
            "type": "stationary",
            "base_mu": 20,
            "use_goodwill": False,
        },
        "num_periods": 30,
    },
)

_register_env_once(
    "GymInvMgmt/Serial-v0",
    entry_point="gym_invmgmt.core_env:CoreEnv",
    kwargs={
        "scenario": "serial",
        "demand_config": {
            "type": "stationary",
            "base_mu": 20,
            "use_goodwill": False,
        },
        "num_periods": 30,
    },
)

__version__ = "0.2.1"

__all__ = [
    "CoreEnv",
    "DemandEngine",
    "DatasetScenarioSpec",
    "SupplyChainNetwork",
    "IntegerActionWrapper",
    "EpisodeLoggerWrapper",
    "external_series_spec",
    "hierarchy_csv_to_tree_topology_yaml",
    "long_demand_csv_to_spec",
    "m5_wide_csv_to_spec",
    "plot_network",
    "retail_store_csv_to_star_topology_yaml",
    "make_custom_env",
    "run_episode",
    "compute_kpis",
    "topology_csvs_to_yaml",
    "wide_demand_csv_to_spec",
    "write_metadata",
]
