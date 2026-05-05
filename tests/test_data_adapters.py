import csv

import numpy as np
import pytest

from gym_invmgmt import CoreEnv
from gym_invmgmt.data_adapters import (
    hierarchy_csv_to_tree_topology_yaml,
    long_demand_csv_to_spec,
    m5_wide_csv_to_spec,
    retail_store_csv_to_star_topology_yaml,
    topology_csvs_to_yaml,
    wide_demand_csv_to_spec,
)


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_m5_wide_csv_to_external_series_spec(tmp_path):
    day_cols = [f"d_{i}" for i in range(1, 41)]
    fieldnames = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + day_cols
    quiet = {c: 10 for c in day_cols}
    volatile_values = [
        8,
        9,
        8,
        10,
        9,
        11,
        12,
        8,
        9,
        10,
        8,
        9,
        10,
        11,
        12,
        9,
        8,
        10,
        11,
        9,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        25,
        30,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        24,
        28,
        30,
    ]
    volatile = {c: volatile_values[i] for i, c in enumerate(day_cols)}
    path = tmp_path / "sales_train_evaluation.csv"
    _write_csv(
        path,
        fieldnames,
        [
            {"id": "quiet", "item_id": "q", "dept_id": "d", "cat_id": "c", "store_id": "s", "state_id": "st", **quiet},
            {
                "id": "volatile",
                "item_id": "v",
                "dept_id": "d",
                "cat_id": "c",
                "store_id": "s",
                "state_id": "st",
                **volatile,
            },
        ],
    )

    spec = m5_wide_csv_to_spec(
        path,
        num_periods=10,
        base_mu=20.0,
        search_days=20,
        min_window_mean=8.0,
        min_nonzero_frac=1.0,
        max_to_mean_cap=4.0,
    )

    series = np.asarray(spec.demand_config["external_series"], dtype=float)
    assert series.shape == (10,)
    assert np.isclose(series.mean(), 20.0)
    assert spec.metadata["adapter"] == "m5_wide_csv"
    assert spec.metadata["series_id"] == "volatile"

    env = CoreEnv(**spec.core_env_kwargs(), num_periods=10)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape


def test_generic_wide_demand_csv_to_external_series_spec(tmp_path):
    period_cols = [f"week_{i:02d}" for i in range(1, 13)]
    fieldnames = ["sku", "store", "category"] + period_cols
    steady = {c: 10 for c in period_cols}
    volatile_values = [8, 9, 8, 9, 8, 24, 28, 30, 8, 9, 8, 9]
    volatile = {c: volatile_values[i] for i, c in enumerate(period_cols)}
    path = tmp_path / "generic_wide_sales.csv"
    _write_csv(
        path,
        fieldnames,
        [
            {"sku": "steady", "store": "A", "category": "grocery", **steady},
            {"sku": "volatile", "store": "B", "category": "grocery", **volatile},
        ],
    )

    spec = wide_demand_csv_to_spec(
        path,
        value_col_prefix="week_",
        metadata_cols=["sku", "store", "category"],
        num_periods=6,
        scale_to_mean=20.0,
        search_periods=12,
        min_window_mean=8.0,
        min_nonzero_frac=1.0,
        max_to_mean_cap=4.0,
        source_name="generic_weekly_sales",
    )

    series = np.asarray(spec.demand_config["external_series"], dtype=float)
    assert series.shape == (6,)
    assert np.isclose(series.mean(), 20.0)
    assert spec.metadata["adapter"] == "wide_csv_window"
    assert spec.metadata["source"] == "generic_weekly_sales"
    assert spec.metadata["sku"] == "volatile"
    assert spec.metadata["start_period"].startswith("week_")


def test_long_demand_csv_to_external_series_spec(tmp_path):
    path = tmp_path / "sales_long.csv"
    _write_csv(
        path,
        ["date", "store", "item", "demand"],
        [
            {"date": "2026-01-02", "store": "A", "item": "sku", "demand": 12},
            {"date": "2026-01-01", "store": "A", "item": "sku", "demand": 8},
            {"date": "2026-01-01", "store": "B", "item": "sku", "demand": 99},
        ],
    )

    spec = long_demand_csv_to_spec(
        path,
        demand_col="demand",
        time_col="date",
        group_filter={"store": "A"},
        scale_to_mean=20.0,
    )

    series = np.asarray(spec.demand_config["external_series"], dtype=float)
    assert np.allclose(series, [16.0, 24.0])
    assert spec.metadata["adapter"] == "long_csv_external_series"


def test_topology_csv_and_edge_mapped_demand(tmp_path):
    nodes_csv = tmp_path / "nodes.csv"
    edges_csv = tmp_path / "edges.csv"
    topo_yaml = tmp_path / "two_retailers.yaml"
    _write_csv(
        nodes_csv,
        ["id", "I0", "h"],
        [
            {"id": 0, "I0": "", "h": ""},
            {"id": 1, "I0": 50, "h": 0.02},
            {"id": 2, "I0": 50, "h": 0.02},
            {"id": 3, "I0": "", "h": ""},
        ],
    )
    _write_csv(
        edges_csv,
        ["from", "to", "L", "p", "g", "b", "demand_dist", "dist_param_mu"],
        [
            {"from": 1, "to": 0, "L": "", "p": 2.0, "g": "", "b": 0.1, "demand_dist": "poisson", "dist_param_mu": 20},
            {"from": 2, "to": 0, "L": "", "p": 2.0, "g": "", "b": 0.1, "demand_dist": "poisson", "dist_param_mu": 20},
            {"from": 3, "to": 1, "L": 0, "p": 0.5, "g": 0.0, "b": "", "demand_dist": "", "dist_param_mu": ""},
            {"from": 3, "to": 2, "L": 0, "p": 0.5, "g": 0.0, "b": "", "demand_dist": "", "dist_param_mu": ""},
        ],
    )
    cfg = topology_csvs_to_yaml(nodes_csv, edges_csv, topo_yaml)
    assert len(cfg["nodes"]) == 4
    assert len(cfg["edges"]) == 4

    demand_csv = tmp_path / "edge_demand.csv"
    _write_csv(
        demand_csv,
        ["t", "market", "demand"],
        [
            {"t": 0, "market": "A", "demand": 5},
            {"t": 1, "market": "A", "demand": 6},
            {"t": 0, "market": "B", "demand": 7},
            {"t": 1, "market": "B", "demand": 8},
        ],
    )
    spec = long_demand_csv_to_spec(
        demand_csv,
        demand_col="demand",
        time_col="t",
        group_cols=["market"],
        edge_map={"A": (1, 0), "B": (2, 0)},
        scenario="custom",
    )
    kwargs = spec.core_env_kwargs()
    kwargs["config_path"] = str(topo_yaml)
    env = CoreEnv(**kwargs, num_periods=2)
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape))

    assert env.D.shape == (2, 2)
    assert np.allclose(env.D[0], [5, 7])


def test_edge_mapped_long_demand_aligns_on_common_time_grid(tmp_path):
    demand_csv = tmp_path / "edge_demand_missing_day.csv"
    _write_csv(
        demand_csv,
        ["date", "market", "demand"],
        [
            {"date": "2026-01-01", "market": "A", "demand": 5},
            {"date": "2026-01-02", "market": "A", "demand": 6},
            {"date": "2026-01-03", "market": "A", "demand": 9},
            {"date": "2026-01-01", "market": "B", "demand": 7},
            {"date": "2026-01-03", "market": "B", "demand": 11},
        ],
    )

    spec = long_demand_csv_to_spec(
        demand_csv,
        demand_col="demand",
        time_col="date",
        group_cols=["market"],
        edge_map={"A": (1, 0), "B": (2, 0)},
        scenario="custom",
    )

    assert np.allclose(spec.env_kwargs["user_D"][(1, 0)], [5, 9])
    assert np.allclose(spec.env_kwargs["user_D"][(2, 0)], [7, 11])
    assert spec.metadata["alignment_policy"] == "intersection"
    assert spec.metadata["aligned_time_count"] == 2


def test_user_d_empirical_demand_respects_goodwill(tmp_path):
    nodes_csv = tmp_path / "nodes.csv"
    edges_csv = tmp_path / "edges.csv"
    topo_yaml = tmp_path / "one_retailer.yaml"
    _write_csv(
        nodes_csv,
        ["id", "I0", "h"],
        [
            {"id": 0, "I0": "", "h": ""},
            {"id": 1, "I0": 0, "h": 0.02},
            {"id": 2, "I0": "", "h": ""},
        ],
    )
    _write_csv(
        edges_csv,
        ["from", "to", "L", "p", "g", "b", "demand_dist", "dist_param_mu"],
        [
            {"from": 1, "to": 0, "L": "", "p": 2.0, "g": "", "b": 0.1, "demand_dist": "poisson", "dist_param_mu": 20},
            {"from": 2, "to": 1, "L": 0, "p": 0.5, "g": 0.0, "b": "", "demand_dist": "", "dist_param_mu": ""},
        ],
    )
    topology_csvs_to_yaml(nodes_csv, edges_csv, topo_yaml)

    env = CoreEnv(
        scenario="custom",
        config_path=str(topo_yaml),
        demand_config={"type": "stationary", "base_mu": 20, "use_goodwill": True},
        user_D={(1, 0): np.asarray([10.0, 10.0])},
        num_periods=2,
    )
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape))
    env.step(np.zeros(env.action_space.shape))

    assert env.D[0, 0] == 10.0
    assert env.GW[1] == 0.9
    assert env.D[1, 0] == 9.0


def test_all_zero_user_d_remains_empirical_zero_demand(tmp_path):
    nodes_csv = tmp_path / "nodes.csv"
    edges_csv = tmp_path / "edges.csv"
    topo_yaml = tmp_path / "zero_demand.yaml"
    _write_csv(
        nodes_csv,
        ["id", "I0", "h"],
        [
            {"id": 0, "I0": "", "h": ""},
            {"id": 1, "I0": 5, "h": 0.02},
            {"id": 2, "I0": "", "h": ""},
        ],
    )
    _write_csv(
        edges_csv,
        ["from", "to", "L", "p", "g", "b", "demand_dist", "dist_param_mu"],
        [
            {"from": 1, "to": 0, "L": "", "p": 2.0, "g": "", "b": 0.1, "demand_dist": "poisson", "dist_param_mu": 20},
            {"from": 2, "to": 1, "L": 1, "p": 0.5, "g": 0.0, "b": "", "demand_dist": "", "dist_param_mu": ""},
        ],
    )
    topology_csvs_to_yaml(nodes_csv, edges_csv, topo_yaml)

    env = CoreEnv(
        scenario="custom",
        config_path=str(topo_yaml),
        user_D={(1, 0): np.zeros(3)},
        num_periods=3,
    )
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape))

    assert env.D[0, 0] == 0.0


def test_custom_topology_rejects_lead_time_on_retail_edge(tmp_path):
    topo_yaml = tmp_path / "bad_retail_l.yaml"
    topo_yaml.write_text(
        """
nodes:
  - id: 0
  - id: 1
    I0: 10
    h: 0.1
  - id: 2
edges:
  - from: 1
    to: 0
    L: 1
    p: 2.0
    b: 1.0
    demand_dist: poisson
    dist_param: {mu: 9}
  - from: 2
    to: 1
    L: 1
    p: 1.0
    g: 0.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not define lead time"):
        CoreEnv(scenario="custom", config_path=str(topo_yaml), num_periods=2)


def test_custom_topology_rejects_non_integer_lead_time(tmp_path):
    topo_yaml = tmp_path / "bad_float_l.yaml"
    topo_yaml.write_text(
        """
nodes:
  - id: 0
  - id: 1
    I0: 10
    h: 0.1
  - id: 2
edges:
  - from: 1
    to: 0
    p: 2.0
    b: 1.0
    demand_dist: poisson
    dist_param: {mu: 9}
  - from: 2
    to: 1
    L: 1.5
    p: 1.0
    g: 0.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-negative integer"):
        CoreEnv(scenario="custom", config_path=str(topo_yaml), num_periods=2)


def test_hierarchy_adapter_adds_explicit_root_source(tmp_path):
    hierarchy_csv = tmp_path / "hierarchy.csv"
    _write_csv(
        hierarchy_csv,
        ["state", "store", "sku"],
        [
            {"state": "CA", "store": "CA_1", "sku": "A"},
            {"state": "TX", "store": "TX_1", "sku": "B"},
        ],
    )
    topo_yaml = tmp_path / "hierarchy.yaml"
    cfg = hierarchy_csv_to_tree_topology_yaml(
        hierarchy_csv,
        topo_yaml,
        hierarchy_cols=["state", "store", "sku"],
    )
    env = CoreEnv(scenario="custom", config_path=str(topo_yaml), num_periods=2)

    assert cfg["metadata"]["root_source_id"] in env.network.rawmat
    assert "state=CA" not in env.network.rawmat
    assert "state=CA" in env.network.distrib


def test_retail_star_adapter_uses_finite_supplier_with_raw_source(tmp_path):
    stores_csv = tmp_path / "stores.csv"
    topo_yaml = tmp_path / "star.yaml"
    _write_csv(stores_csv, ["Store", "Region"], [{"Store": 7, "Region": "A"}])

    cfg = retail_store_csv_to_star_topology_yaml(
        stores_csv,
        topo_yaml,
        store_id_col="Store",
        selected_store_ids=[7],
        supplier_id=8,
        source_id=9,
    )
    env = CoreEnv(scenario="custom", config_path=str(topo_yaml), num_periods=2)

    assert cfg["metadata"]["supplier_id"] == "8"
    assert 9 in env.network.rawmat
    assert 8 in env.network.distrib
    assert (9, 8) in env.network.reorder_links
