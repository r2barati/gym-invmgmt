import csv

import numpy as np

from gym_invmgmt import CoreEnv
from gym_invmgmt.data_adapters import (
    hierarchy_csv_to_tree_topology_yaml,
    long_demand_csv_to_spec,
    topology_csvs_to_yaml,
)


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_rossmann_style_daily_store_sales_can_drive_external_series(tmp_path):
    """Validate a Rossmann-like store/date/sales table without dataset-specific code."""
    sales_csv = tmp_path / "rossmann_train_like.csv"
    _write_csv(
        sales_csv,
        ["Date", "Store", "Sales", "Open", "Promo"],
        [
            {"Date": "2015-01-03", "Store": 7, "Sales": 10, "Open": 1, "Promo": 0},
            {"Date": "2015-01-01", "Store": 7, "Sales": 8, "Open": 1, "Promo": 1},
            {"Date": "2015-01-02", "Store": 7, "Sales": 12, "Open": 1, "Promo": 0},
            {"Date": "2015-01-04", "Store": 7, "Sales": 10, "Open": 1, "Promo": 0},
            {"Date": "2015-01-05", "Store": 7, "Sales": 10, "Open": 1, "Promo": 1},
            {"Date": "2015-01-06", "Store": 7, "Sales": 10, "Open": 1, "Promo": 0},
            {"Date": "2015-01-07", "Store": 7, "Sales": 10, "Open": 1, "Promo": 0},
            {"Date": "2015-01-08", "Store": 7, "Sales": 10, "Open": 1, "Promo": 1},
            {"Date": "2015-01-09", "Store": 7, "Sales": 10, "Open": 1, "Promo": 0},
            {"Date": "2015-01-10", "Store": 7, "Sales": 10, "Open": 1, "Promo": 0},
            {"Date": "2015-01-01", "Store": 8, "Sales": 9999, "Open": 1, "Promo": 1},
        ],
    )

    spec = long_demand_csv_to_spec(
        sales_csv,
        demand_col="Sales",
        time_col="Date",
        group_filter={"Store": 7},
        scale_to_mean=20.0,
    )

    series = np.asarray(spec.demand_config["external_series"], dtype=float)
    assert np.allclose(series[:3], [16.0, 24.0, 20.0])
    assert np.isclose(series.mean(), 20.0)
    assert spec.metadata["adapter"] == "long_csv_external_series"

    env = CoreEnv(**spec.core_env_kwargs(), num_periods=10)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape


def test_favorita_style_store_item_sales_can_drive_multi_retailer_graph(tmp_path):
    """Validate a Favorita-like sales table mapped to two retail demand edges."""
    nodes_csv = tmp_path / "nodes.csv"
    edges_csv = tmp_path / "edges.csv"
    topology_yaml = tmp_path / "favorita_two_store_network.yaml"
    _write_csv(
        nodes_csv,
        ["id", "I0", "h"],
        [
            {"id": 0, "I0": "", "h": ""},
            {"id": 1, "I0": 60, "h": 0.02},
            {"id": 2, "I0": 60, "h": 0.02},
            {"id": 3, "I0": 150, "h": 0.01},
        ],
    )
    _write_csv(
        edges_csv,
        ["from", "to", "L", "p", "g", "b", "demand_dist", "dist_param_mu"],
        [
            {"from": 1, "to": 0, "L": "", "p": 3.0, "g": "", "b": 0.2, "demand_dist": "poisson", "dist_param_mu": 20},
            {"from": 2, "to": 0, "L": "", "p": 3.0, "g": "", "b": 0.2, "demand_dist": "poisson", "dist_param_mu": 20},
            {"from": 3, "to": 1, "L": 1, "p": 1.0, "g": 0.0, "b": "", "demand_dist": "", "dist_param_mu": ""},
            {"from": 3, "to": 2, "L": 1, "p": 1.0, "g": 0.0, "b": "", "demand_dist": "", "dist_param_mu": ""},
        ],
    )
    topology_csvs_to_yaml(nodes_csv, edges_csv, topology_yaml, name="favorita_two_store_network")

    sales_csv = tmp_path / "favorita_train_like.csv"
    _write_csv(
        sales_csv,
        ["date", "store_nbr", "family", "sales", "onpromotion"],
        [
            {"date": "2017-01-01", "store_nbr": 1, "family": "GROCERY I", "sales": 11, "onpromotion": 0},
            {"date": "2017-01-02", "store_nbr": 1, "family": "GROCERY I", "sales": 13, "onpromotion": 1},
            {"date": "2017-01-01", "store_nbr": 2, "family": "GROCERY I", "sales": 17, "onpromotion": 0},
            {"date": "2017-01-02", "store_nbr": 2, "family": "GROCERY I", "sales": 19, "onpromotion": 1},
            {"date": "2017-01-01", "store_nbr": 1, "family": "BEVERAGES", "sales": 99, "onpromotion": 0},
        ],
    )

    spec = long_demand_csv_to_spec(
        sales_csv,
        demand_col="sales",
        time_col="date",
        group_cols=["store_nbr"],
        group_filter={"family": "GROCERY I"},
        edge_map={"1": (1, 0), "2": (2, 0)},
        scenario="custom",
    )

    kwargs = spec.core_env_kwargs()
    kwargs["config_path"] = str(topology_yaml)
    env = CoreEnv(**kwargs, num_periods=2)
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape))
    env.step(np.zeros(env.action_space.shape))

    assert env.D.shape == (2, 2)
    assert np.allclose(env.D[0], [11, 17])
    assert np.allclose(env.D[1], [13, 19])


def test_hierarchy_csv_can_infer_tree_topology(tmp_path):
    hierarchy_csv = tmp_path / "hierarchical_sales.csv"
    _write_csv(
        hierarchy_csv,
        ["state", "store", "category", "sku"],
        [
            {"state": "CA", "store": "CA_1", "category": "FOODS", "sku": "FOODS_1"},
            {"state": "CA", "store": "CA_1", "category": "FOODS", "sku": "FOODS_2"},
            {"state": "TX", "store": "TX_1", "category": "HOBBIES", "sku": "HOBBIES_1"},
        ],
    )
    topology_yaml = tmp_path / "hierarchy.yaml"
    cfg = hierarchy_csv_to_tree_topology_yaml(
        hierarchy_csv,
        topology_yaml,
        hierarchy_cols=["state", "store", "category", "sku"],
        group_filter={"state": "CA"},
    )

    assert cfg["metadata"]["adapter"] == "hierarchy_csv_tree_topology"
    assert cfg["metadata"]["path_count"] == 2
    assert cfg["metadata"]["leaf_count"] == 2

    env = CoreEnv(scenario="custom", config_path=str(topology_yaml), num_periods=2)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert len(env.network.retail_links) == 2
