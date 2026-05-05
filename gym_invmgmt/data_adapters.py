"""Dataset adapters for turning external demand/topology files into CoreEnv inputs.

The core environment already has two generic hooks:

* ``demand_config["external_series"]`` for one empirical retail-demand trace.
* ``user_D={(retailer, market): series}`` for edge-specific demand traces.

This module provides small, explicit parsers that convert common dataset shapes
into those hooks without making the benchmark code depend on one dataset such as
M5.  The adapters intentionally return ordinary dictionaries and metadata so the
resulting scenario can be inspected, serialized, or passed directly to
``CoreEnv``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import yaml

EdgeKey = Tuple[Any, Any]


@dataclass
class DatasetScenarioSpec:
    """CoreEnv-ready scenario produced by a dataset adapter."""

    scenario: str = "network"
    demand_config: dict[str, Any] = field(default_factory=dict)
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    topology_config: dict[str, Any] | None = None
    topology_path: str | None = None

    def core_env_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments suitable for ``CoreEnv(**kwargs)``."""
        kwargs = dict(self.env_kwargs)
        kwargs["scenario"] = self.scenario
        kwargs["demand_config"] = self.demand_config
        if self.topology_path is not None:
            kwargs["scenario"] = "custom"
            kwargs["config_path"] = self.topology_path
        return kwargs


def _coerce_scalar(value: Any) -> Any:
    """Coerce CSV string values into bool/int/float where unambiguous."""
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    v = value.strip()
    if v == "":
        return None
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _scale_series(series: Sequence[float], scale_to_mean: float | None) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Demand series must be one-dimensional, got shape {arr.shape}.")
    if len(arr) == 0:
        raise ValueError("Demand series must not be empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("Demand series contains NaN or infinite values.")
    arr = np.maximum(arr, 0.0)
    if scale_to_mean is not None:
        mean = float(np.mean(arr))
        if mean <= 0:
            raise ValueError("Cannot scale a zero-mean demand series.")
        arr = arr * (float(scale_to_mean) / mean)
    return arr.astype(float)


def external_series_spec(
    series: Sequence[float],
    *,
    scenario: str = "network",
    scale_to_mean: float | None = None,
    use_goodwill: bool = False,
    deterministic: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> DatasetScenarioSpec:
    """Build a CoreEnv-ready spec from a single empirical demand trace."""
    arr = _scale_series(series, scale_to_mean)
    demand_config = {
        "type": "stationary",
        "base_mu": float(np.mean(arr)),
        "use_goodwill": bool(use_goodwill),
        "external_series": arr,
        "noise_scale": 0.0 if deterministic else 1.0,
    }
    md = {
        "adapter": "external_series",
        "num_periods": int(len(arr)),
        "scaled_mean": float(np.mean(arr)),
        "scaled_std": float(np.std(arr)),
        "scaled_cv": float(np.std(arr) / max(np.mean(arr), 1e-9)),
    }
    if metadata:
        md.update(dict(metadata))
    return DatasetScenarioSpec(scenario=scenario, demand_config=demand_config, metadata=md)


def long_demand_csv_to_spec(
    csv_path: str | Path,
    *,
    demand_col: str,
    time_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    group_filter: Mapping[str, Any] | None = None,
    edge_map: Mapping[Any, EdgeKey] | None = None,
    scenario: str = "network",
    scale_to_mean: float | None = None,
    use_goodwill: bool = False,
) -> DatasetScenarioSpec:
    """Parse a generic long-format demand CSV.

    For a single selected group, the result uses ``external_series``.  If
    ``edge_map`` is provided, each group is mapped to a retail edge and returned
    through ``env_kwargs["user_D"]``.

    Expected minimal long format::

        date, store, item, demand
        2026-01-01, A, sku1, 12
    """
    path = Path(csv_path)
    group_cols = list(group_cols or [])
    group_filter = dict(group_filter or {})
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if demand_col not in (reader.fieldnames or []):
            raise ValueError(f"Column '{demand_col}' not found in {path}.")
        for row in reader:
            if any(str(row.get(k, "")) != str(v) for k, v in group_filter.items()):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows matched filters {group_filter} in {path}.")
    if time_col:
        rows.sort(key=lambda r: r[time_col])

    if edge_map:
        if not group_cols:
            raise ValueError("edge_map requires at least one group column.")

        def _group_key(row: Mapping[str, Any]) -> Any:
            return tuple(row[c] for c in group_cols) if len(group_cols) > 1 else row[group_cols[0]]

        user_D: dict[EdgeKey, np.ndarray] = {}
        metadata_extra: dict[str, Any] = {}
        if time_col:
            grouped_by_time: dict[Any, dict[Any, float]] = {k: {} for k in edge_map}
            for row in rows:
                key = _group_key(row)
                if key not in grouped_by_time:
                    continue
                t_value = row[time_col]
                grouped_by_time[key][t_value] = grouped_by_time[key].get(t_value, 0.0) + float(row[demand_col])

            missing_groups = [key for key, values in grouped_by_time.items() if not values]
            if missing_groups:
                raise ValueError(f"No demand rows found for mapped group(s): {missing_groups!r}.")
            common_times = set.intersection(*(set(values) for values in grouped_by_time.values()))
            if not common_times:
                raise ValueError("Mapped demand groups have no shared time periods.")

            def _time_sort_key(value: Any) -> tuple[int, Any]:
                try:
                    return (0, float(value))
                except (TypeError, ValueError):
                    return (1, str(value))

            aligned_times = sorted(common_times, key=_time_sort_key)
            for key, edge in edge_map.items():
                user_D[edge] = _scale_series(
                    [grouped_by_time[key][t_value] for t_value in aligned_times],
                    scale_to_mean,
                )
            metadata_extra = {
                "alignment_time_col": time_col,
                "alignment_policy": "intersection",
                "aligned_time_count": len(aligned_times),
                "dropped_unaligned_rows": sum(len(values) - len(aligned_times) for values in grouped_by_time.values()),
            }
        else:
            grouped: dict[Any, list[float]] = {k: [] for k in edge_map}
            for row in rows:
                key = _group_key(row)
                if key in grouped:
                    grouped[key].append(float(row[demand_col]))
            lengths = {key: len(values) for key, values in grouped.items()}
            missing_groups = [key for key, length in lengths.items() if length == 0]
            if missing_groups:
                raise ValueError(f"No demand rows found for mapped group(s): {missing_groups!r}.")
            if len(set(lengths.values())) != 1:
                raise ValueError("Edge-mapped demand groups must have equal lengths when time_col is not provided.")
            for key, edge in edge_map.items():
                user_D[edge] = _scale_series(grouped[key], scale_to_mean)
        return DatasetScenarioSpec(
            scenario=scenario,
            demand_config={"type": "stationary", "base_mu": 20, "use_goodwill": bool(use_goodwill)},
            env_kwargs={"user_D": user_D},
            metadata={
                "adapter": "long_csv_edge_map",
                "source": str(path),
                "edge_count": len(user_D),
                "group_cols": group_cols,
                **metadata_extra,
            },
        )

    series = [float(row[demand_col]) for row in rows]
    return external_series_spec(
        series,
        scenario=scenario,
        scale_to_mean=scale_to_mean,
        use_goodwill=use_goodwill,
        metadata={
            "adapter": "long_csv_external_series",
            "source": str(path),
            "row_count": len(rows),
            "group_filter": group_filter,
        },
    )


def wide_demand_csv_to_spec(
    csv_path: str | Path,
    *,
    scenario: str = "network",
    value_cols: Sequence[str] | None = None,
    value_col_prefix: str | None = None,
    metadata_cols: Sequence[str] | None = None,
    group_filter: Mapping[str, Any] | None = None,
    num_periods: int = 30,
    scale_to_mean: float = 20.0,
    search_periods: int | None = None,
    min_window_mean: float = 8.0,
    min_nonzero_frac: float = 0.80,
    max_to_mean_cap: float = 4.0,
    use_goodwill: bool = False,
    source_name: str | None = None,
    source_url: str | None = None,
) -> DatasetScenarioSpec:
    """Parse a generic wide-format demand CSV into one empirical demand trace.

    This handles datasets where each row is one entity (store, item-store, SKU,
    etc.) and repeated time periods are stored in separate columns, such as
    ``d_1, d_2, ...`` or ``week_001, week_002, ...``. The adapter selects a
    reproducible rolling window by maximizing coefficient of variation subject
    to simple viability filters.
    """
    path = Path(csv_path)
    group_filter = dict(group_filter or {})
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if value_cols is None:
            if value_col_prefix is None:
                raise ValueError("Provide either value_cols or value_col_prefix.")
            selected_value_cols = [c for c in fieldnames if c.startswith(value_col_prefix)]
        else:
            selected_value_cols = list(value_cols)
            missing = [c for c in selected_value_cols if c not in fieldnames]
            if missing:
                raise ValueError(f"Value columns not found in {path}: {missing}")

        if len(selected_value_cols) < num_periods:
            raise ValueError(
                f"Wide demand file has {len(selected_value_cols)} value columns; need at least {num_periods}."
            )
        search_count = max(search_periods or len(selected_value_cols), num_periods)
        search_cols = selected_value_cols[-search_count:]
        rows: list[dict[str, str]] = []
        values_list: list[list[float]] = []
        for row in reader:
            if any(str(row.get(k, "")) != str(v) for k, v in group_filter.items()):
                continue
            rows.append(dict(row))
            values_list.append([float(row[c]) for c in search_cols])
    if not rows:
        raise ValueError(f"No rows matched filters {group_filter} in {path}.")

    values = np.asarray(values_list, dtype=float)
    if values.ndim != 2 or values.shape[1] < num_periods:
        raise ValueError(f"Unexpected wide demand matrix shape: {values.shape}")

    # Vectorized rolling mean/std/nonzero calculations. This keeps the public
    # adapter fast enough for full retail datasets while avoiding a pandas
    # dependency.
    padded = np.pad(values, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
    csum = np.cumsum(padded, axis=1)
    csum2 = np.cumsum(padded * padded, axis=1)
    sums = csum[:, num_periods:] - csum[:, :-num_periods]
    sums2 = csum2[:, num_periods:] - csum2[:, :-num_periods]
    means = sums / float(num_periods)
    variances = np.maximum(sums2 / float(num_periods) - means * means, 0.0)
    stds = np.sqrt(variances)
    cvs = stds / np.maximum(means, 1e-9)

    nz = (values > 0).astype(float)
    nz_padded = np.pad(nz, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
    nz_csum = np.cumsum(nz_padded, axis=1)
    nonzero_frac = (nz_csum[:, num_periods:] - nz_csum[:, :-num_periods]) / float(num_periods)

    max_values = np.empty_like(means)
    for win_idx in range(means.shape[1]):
        max_values[:, win_idx] = np.max(values[:, win_idx : win_idx + num_periods], axis=1)
    max_to_mean = max_values / np.maximum(means, 1e-9)

    valid = (
        (means >= float(min_window_mean))
        & (nonzero_frac >= float(min_nonzero_frac))
        & (max_to_mean <= float(max_to_mean_cap))
    )
    if not np.any(valid):
        raise ValueError("No wide demand window satisfied the selection filters.")

    scores = np.where(valid, cvs, -np.inf)
    row_idx, win_idx = np.unravel_index(int(np.argmax(scores)), scores.shape)
    raw_series = values[row_idx, win_idx : win_idx + num_periods].astype(float).tolist()
    row = rows[row_idx]
    arr = _scale_series(raw_series, scale_to_mean)
    raw_mean = float(np.mean(raw_series))
    raw_std = float(np.std(raw_series))
    raw_nonzero_frac = float(np.mean(np.asarray(raw_series, dtype=float) > 0))
    raw_max_to_mean = float(np.max(raw_series) / max(raw_mean, 1e-9))
    raw_cv = float(raw_std / max(raw_mean, 1e-9))
    metadata_cols = list(metadata_cols or [])
    selected_metadata = {col: row.get(col) for col in metadata_cols if col in row}
    metadata = {
        "adapter": "wide_csv_window",
        "source": source_name or str(path),
        "source_path": str(path),
        "source_url": source_url,
        "selection_rule": (
            "highest rolling coefficient of variation among "
            f"{num_periods}-period row windows in the last {len(search_cols)} "
            f"available periods, subject to mean >= {min_window_mean}, "
            f"nonzero fraction >= {min_nonzero_frac}, and max/mean <= {max_to_mean_cap}"
        ),
        "num_periods": int(num_periods),
        "base_mu": float(scale_to_mean),
        "search_periods": int(len(search_cols)),
        "min_window_mean": float(min_window_mean),
        "min_nonzero_frac": float(min_nonzero_frac),
        "max_to_mean_cap": float(max_to_mean_cap),
        "row_index": int(row_idx),
        "window_index": int(win_idx),
        "start_period": search_cols[win_idx],
        "end_period": search_cols[win_idx + num_periods - 1],
        **selected_metadata,
        "raw_series": list(map(float, raw_series)),
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "raw_cv": raw_cv,
        "raw_nonzero_frac": raw_nonzero_frac,
        "raw_max_to_mean": raw_max_to_mean,
        "scaled_series": arr.tolist(),
    }
    return external_series_spec(arr, scenario=scenario, use_goodwill=use_goodwill, metadata=metadata)


def m5_wide_csv_to_spec(
    csv_path: str | Path,
    *,
    scenario: str = "network",
    num_periods: int = 30,
    base_mu: float = 20.0,
    search_days: int = 365,
    min_window_mean: float = 8.0,
    min_nonzero_frac: float = 0.80,
    max_to_mean_cap: float = 4.0,
) -> DatasetScenarioSpec:
    """Parse M5 ``sales_train_evaluation.csv`` wide format into a demand spec.

    This is a convenience wrapper over :func:`wide_demand_csv_to_spec`; it is not
    a separate benchmark-specific parser.
    """
    spec = wide_demand_csv_to_spec(
        csv_path,
        scenario=scenario,
        value_col_prefix="d_",
        metadata_cols=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        num_periods=num_periods,
        scale_to_mean=base_mu,
        search_periods=search_days,
        min_window_mean=min_window_mean,
        min_nonzero_frac=min_nonzero_frac,
        max_to_mean_cap=max_to_mean_cap,
        source_name="M5 sales_train_evaluation.csv",
        source_url="https://huggingface.co/datasets/kashif/M5/blob/main/sales_train_evaluation.csv",
    )
    metadata = dict(spec.metadata)
    metadata.update(
        {
            "adapter": "m5_wide_csv",
            "search_days": int(metadata["search_periods"]),
            "start_day": metadata["start_period"],
            "end_day": metadata["end_period"],
            "series_id": metadata.get("id"),
        }
    )
    spec.metadata = metadata
    return spec


def topology_csvs_to_yaml(
    nodes_csv: str | Path,
    edges_csv: str | Path,
    output_path: str | Path,
    *,
    name: str = "dataset_topology",
) -> dict[str, Any]:
    """Convert simple node/edge CSV files into a gym-invmgmt topology YAML.

    Node CSV requires an ``id`` column.  Edge CSV requires ``from`` and ``to``.
    Blank cells are ignored.  Columns named ``dist_param_<name>`` are grouped
    into the edge's ``dist_param`` mapping.
    """
    nodes_path = Path(nodes_csv)
    edges_path = Path(edges_csv)
    out = Path(output_path)

    nodes: list[dict[str, Any]] = []
    with nodes_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "id" not in (reader.fieldnames or []):
            raise ValueError("Node CSV must contain an 'id' column.")
        for row in reader:
            node: dict[str, Any] = {}
            for k, v in row.items():
                parsed = _coerce_scalar(v)
                if parsed is not None:
                    node[k] = parsed
            nodes.append(node)

    edges: list[dict[str, Any]] = []
    with edges_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"from", "to"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Edge CSV must contain 'from' and 'to' columns.")
        for row in reader:
            edge: dict[str, Any] = {}
            dist_param: dict[str, Any] = {}
            for k, v in row.items():
                parsed = _coerce_scalar(v)
                if parsed is None:
                    continue
                if k.startswith("dist_param_"):
                    dist_param[k.replace("dist_param_", "", 1)] = parsed
                else:
                    edge[k] = parsed
            if dist_param:
                edge["dist_param"] = dist_param
            edges.append(edge)

    cfg = {"name": name, "nodes": nodes, "edges": edges}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def retail_store_csv_to_star_topology_yaml(
    stores_csv: str | Path,
    output_path: str | Path,
    *,
    store_id_col: str,
    selected_store_ids: Sequence[Any],
    name: str = "retail_star_topology",
    market_id: int = 0,
    supplier_id: int | None = None,
    source_id: Any | None = None,
    initial_inventory: float = 80.0,
    retailer_holding_cost: float = 0.02,
    supplier_inventory: float = 500.0,
    supplier_holding_cost: float = 0.005,
    lead_time: int = 1,
    retail_price: float = 3.0,
    backlog_cost: float = 0.2,
    purchase_cost: float = 1.0,
    demand_mu: float = 20.0,
) -> dict[str, Any]:
    """Infer a simple warehouse-to-store topology from retail-store metadata.

    Retail datasets such as Rossmann and Favorita expose store metadata and
    demand, but not the physical replenishment network.  This adapter therefore
    creates an explicit, documented star topology: one synthetic upstream
    supplier replenishes the selected store nodes, and each store serves the
    market node.  A separate unlimited raw-source node feeds the finite
    supplier so supplier inventory and holding cost are meaningful in CoreEnv.
    The graph is useful for standalone environment construction, but should be
    described as inferred rather than observed from the dataset.
    """
    stores_path = Path(stores_csv)
    selected = {str(s) for s in selected_store_ids}
    stores: dict[str, dict[str, Any]] = {}
    with stores_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if store_id_col not in (reader.fieldnames or []):
            raise ValueError(f"Store CSV must contain '{store_id_col}'.")
        for row in reader:
            store_id = str(row[store_id_col])
            if store_id in selected:
                stores[store_id] = {k: _coerce_scalar(v) for k, v in row.items() if _coerce_scalar(v) is not None}

    missing = selected.difference(stores)
    if missing:
        raise ValueError(f"Selected store IDs not found in {stores_path}: {sorted(missing)}")

    parsed_store_ids = [_coerce_scalar(s) for s in selected_store_ids]
    if supplier_id is None:
        numeric_ids = [int(s) for s in parsed_store_ids if isinstance(s, int)]
        supplier_id = (max(numeric_ids) + 1) if numeric_ids else 999_999
    if source_id is None:
        numeric_ids = [int(s) for s in [market_id, supplier_id, *parsed_store_ids] if isinstance(s, int)]
        if numeric_ids:
            source_id = max(numeric_ids) + 1
        else:
            source_id = "RAW_SOURCE"
            used = {str(market_id), str(supplier_id), *(str(s) for s in parsed_store_ids)}
            while str(source_id) in used:
                source_id = f"{source_id}_UPSTREAM"

    nodes: list[dict[str, Any]] = [{"id": market_id}]
    for raw_id, parsed_id in zip(selected_store_ids, parsed_store_ids):
        store_meta = {
            k: v for k, v in stores[str(raw_id)].items() if k != store_id_col and isinstance(v, (str, int, float, bool))
        }
        nodes.append(
            {
                "id": parsed_id,
                "I0": float(initial_inventory),
                "h": float(retailer_holding_cost),
                **{f"meta_{k}": v for k, v in store_meta.items()},
            }
        )
    nodes.append({"id": supplier_id, "I0": float(supplier_inventory), "h": float(supplier_holding_cost)})
    nodes.append({"id": source_id})

    edges: list[dict[str, Any]] = []
    for parsed_id in parsed_store_ids:
        edges.append(
            {
                "from": parsed_id,
                "to": market_id,
                "p": float(retail_price),
                "b": float(backlog_cost),
                "demand_dist": "poisson",
                "dist_param": {"mu": float(demand_mu)},
            }
        )
        edges.append(
            {
                "from": supplier_id,
                "to": parsed_id,
                "L": int(lead_time),
                "p": float(purchase_cost),
                "g": 0.0,
            }
        )
    edges.append(
        {
            "from": source_id,
            "to": supplier_id,
            "L": int(lead_time),
            "p": 0.0,
            "g": 0.0,
        }
    )

    cfg = {
        "name": name,
        "metadata": {
            "adapter": "retail_store_csv_star_topology",
            "source": str(stores_path),
            "store_id_col": store_id_col,
            "selected_store_ids": list(map(str, selected_store_ids)),
            "source_id": str(source_id),
            "supplier_id": str(supplier_id),
            "topology_assumption": (
                "synthetic finite-supplier star graph inferred from store metadata, "
                "with an explicit unlimited upstream raw source"
            ),
        },
        "nodes": nodes,
        "edges": edges,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def hierarchy_csv_to_tree_topology_yaml(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    hierarchy_cols: Sequence[str],
    group_filter: Mapping[str, Any] | None = None,
    max_rows: int | None = None,
    name: str = "hierarchy_topology",
    market_id: str = "MARKET",
    initial_inventory: float = 80.0,
    holding_cost: float = 0.02,
    lead_time: int = 1,
    retail_price: float = 3.0,
    backlog_cost: float = 0.2,
    purchase_cost: float = 1.0,
    pipeline_holding_cost: float = 0.0,
    demand_mu: float = 20.0,
    root_id: str = "SOURCE",
) -> dict[str, Any]:
    """Infer an environment DAG from dataset hierarchy columns.

    Datasets such as M5 expose an official forecasting hierarchy
    (state/store/category/department/item), but this hierarchy is an
    aggregation/planning structure rather than a physically observed
    replenishment network.  This adapter maps each unique hierarchy path into a
    tree-shaped custom topology and connects every leaf to a synthetic market.
    The result is useful for topology-transfer and standalone-environment
    experiments, but should be described as hierarchy-inferred.
    """
    path = Path(csv_path)
    hierarchy_cols = list(hierarchy_cols)
    if not hierarchy_cols:
        raise ValueError("hierarchy_cols must contain at least one column.")
    group_filter = dict(group_filter or {})

    paths: set[tuple[str, ...]] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in hierarchy_cols if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Hierarchy columns not found in {path}: {missing}")
        for row_idx, row in enumerate(reader):
            if max_rows is not None and row_idx >= max_rows:
                break
            if any(str(row.get(k, "")) != str(v) for k, v in group_filter.items()):
                continue
            values = tuple(str(row[c]) for c in hierarchy_cols)
            if all(v != "" for v in values):
                paths.add(values)

    if not paths:
        raise ValueError(f"No hierarchy rows matched filters {group_filter} in {path}.")

    node_ids: set[str] = {market_id}
    edges: set[tuple[str, str]] = set()
    leaf_ids: set[str] = set()
    first_level_ids: set[str] = set()

    def node_id(level_idx: int, value: str) -> str:
        return f"{hierarchy_cols[level_idx]}={value}"

    for values in sorted(paths):
        ids = [node_id(i, value) for i, value in enumerate(values)]
        node_ids.update(ids)
        first_level_ids.add(ids[0])
        for parent, child in zip(ids[:-1], ids[1:]):
            edges.add((parent, child))
        leaf_ids.add(ids[-1])
        edges.add((ids[-1], market_id))

    source_id = root_id
    while source_id in node_ids or source_id == market_id:
        source_id = f"{source_id}_UPSTREAM"
    node_ids.add(source_id)
    for top_id in first_level_ids:
        edges.add((source_id, top_id))

    nodes = []
    for nid in sorted(node_ids):
        if nid == market_id or nid == source_id:
            nodes.append({"id": nid})
        else:
            nodes.append({"id": nid, "I0": float(initial_inventory), "h": float(holding_cost)})

    yaml_edges: list[dict[str, Any]] = []
    for src, dst in sorted(edges):
        if dst == market_id:
            yaml_edges.append(
                {
                    "from": src,
                    "to": dst,
                    "p": float(retail_price),
                    "b": float(backlog_cost),
                    "demand_dist": "poisson",
                    "dist_param": {"mu": float(demand_mu)},
                }
            )
        else:
            yaml_edges.append(
                {
                    "from": src,
                    "to": dst,
                    "L": int(lead_time),
                    "p": float(purchase_cost),
                    "g": float(pipeline_holding_cost),
                }
            )

    cfg = {
        "name": name,
        "metadata": {
            "adapter": "hierarchy_csv_tree_topology",
            "source": str(path),
            "hierarchy_cols": hierarchy_cols,
            "group_filter": group_filter,
            "path_count": len(paths),
            "leaf_count": len(leaf_ids),
            "root_source_id": source_id,
            "topology_assumption": (
                "tree-shaped replenishment DAG inferred from dataset forecasting "
                "hierarchy, with an explicit unlimited root source feeding finite "
                "hierarchy nodes; not a physically observed supply-chain network"
            ),
            "leaf_edges": [{"leaf": leaf, "market": market_id} for leaf in sorted(leaf_ids)],
        },
        "nodes": nodes,
        "edges": yaml_edges,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def write_metadata(metadata: Mapping[str, Any], output_path: str | Path) -> None:
    """Write adapter metadata as JSON for reproducibility."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(dict(metadata), f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
