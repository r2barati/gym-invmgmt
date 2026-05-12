from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Optional, Union

import networkx as nx
import numpy as np
import yaml
from scipy.stats import poisson

from gym_invmgmt.utils import DIST_REGISTRY


class SupplyChainNetwork:
    """
    The Network Context Class.

    Responsibilities:
    1. Defines the Graph Topology (Nodes, Edges, Parameters).
    2. Compiles static mappings (Node -> Index) for NumPy optimization.
    3. Pre-computes adjacency lists so the Env doesn't have to.

    Node Classification Rules (auto-detected from graph properties):
        - **Market**: Nodes with no successors (demand sinks).
        - **Retailer**: Nodes that supply at least one Market node.
        - **Distributor**: Nodes with initial inventory ``I0`` but no capacity ``C``.
        - **Factory**: Nodes with a production capacity attribute ``C``.
        - **Raw Material**: Nodes with no predecessors (supply sources).

    Scenarios:
        - ``network`` / ``base``: Built-in divergent multi-echelon network.
        - ``serial``: Built-in serial supply chain.
        - ``custom``: User-defined topology loaded from a YAML config file.
    """

    def __init__(
        self,
        scenario: str = "network",
        user_D: Optional[dict] = None,
        sample_path: Optional[dict] = None,
        num_periods: int = 30,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        # 1. Setup Configuration
        self.num_periods = num_periods
        self.user_D = user_D if user_D is not None else {}
        self.sample_path = sample_path if sample_path is not None else {}
        self.levels = {}
        self.graph = nx.DiGraph()

        # 2. Build Graph Topology
        if scenario in ("network", "base"):
            self._build_network_scenario()
        elif scenario == "serial":
            self._build_serial_scenario()
        elif scenario == "custom":
            if config_path is None:
                raise ValueError("scenario='custom' requires a config_path pointing to a YAML file.")
            self._build_custom_scenario(Path(config_path))
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # 3. Apply Metadata (User Demand / Sample Paths)
        self._apply_metadata()

        # 4. Compile NumPy Mappings (The heavy lifting)
        self._compile_indices()

        # 5. Compute Space Limits (for Observation/Action spaces)
        self._compute_space_limits()

    # ── Custom YAML Parser ────────────────────────────────────────────

    def _build_custom_scenario(self, config_path: Path) -> None:
        """Build a network topology from a YAML configuration file.

        Args:
            config_path: Path to the YAML config file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the config is malformed or the graph is invalid.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Check file extension
        if config_path.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError(
                f"Config file must be a YAML file (.yaml or .yml), " f"got '{config_path.suffix}': {config_path}"
            )

        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML config file '{config_path}': {e}") from e

        if cfg is None:
            raise ValueError(f"Config file is empty or contains no valid YAML: {config_path}")
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Config file must contain a YAML mapping (key-value pairs), "
                f"got {type(cfg).__name__}: {config_path}"
            )

        # ── Parse Nodes ───────────────────────────────────────────────
        if "nodes" not in cfg or not cfg["nodes"]:
            raise ValueError("Config must contain a non-empty 'nodes' list.")

        for node_def in cfg["nodes"]:
            if "id" not in node_def:
                raise ValueError(f"Each node must have an 'id' field. Got: {node_def}")
            node_id = node_def["id"]
            attrs = {k: v for k, v in node_def.items() if k != "id"}
            self.graph.add_node(node_id, **attrs)

        # ── Parse Edges ───────────────────────────────────────────────
        if "edges" not in cfg or not cfg["edges"]:
            raise ValueError("Config must contain a non-empty 'edges' list.")

        for edge_def in cfg["edges"]:
            if "from" not in edge_def or "to" not in edge_def:
                raise ValueError(f"Each edge must have 'from' and 'to' fields. Got: {edge_def}")

            src = edge_def["from"]
            dst = edge_def["to"]

            if src not in self.graph:
                raise ValueError(f"Edge references unknown source node {src}")
            if dst not in self.graph:
                raise ValueError(f"Edge references unknown target node {dst}")

            attrs = {k: v for k, v in edge_def.items() if k not in ("from", "to")}

            # Resolve distribution name → scipy object
            if "demand_dist" in attrs:
                dist_name = attrs["demand_dist"]
                if dist_name not in DIST_REGISTRY:
                    raise ValueError(
                        f"Unknown demand distribution '{dist_name}'. " f"Supported: {list(DIST_REGISTRY.keys())}"
                    )
                attrs["demand_dist"] = DIST_REGISTRY[dist_name]

            self.graph.add_edge(src, dst, **attrs)

        # ── Validate Graph Structure ──────────────────────────────────
        self._validate_custom_graph()

        # ── Auto-detect Levels for Visualization ──────────────────────
        self._auto_detect_levels()

    def _validate_custom_graph(self) -> None:
        """Validate the custom graph structure and required edge attributes."""
        # Must be a DAG
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Custom network must be a directed acyclic graph (DAG). Cycles detected.")

        # Must be weakly connected
        if not nx.is_weakly_connected(self.graph):
            raise ValueError("Custom network must be a single connected component.")

        # Check for at least one market (sink) and one raw material (source)
        markets = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        rawmats = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]

        if not markets:
            raise ValueError("Network must have at least one market node (a node with no successors).")
        if not rawmats:
            raise ValueError("Network must have at least one raw material node (a node with no predecessors).")

        # Validate node attributes
        for n, data in self.graph.nodes(data=True):
            if n in rawmats or n in markets:
                continue

            if "h" not in data:
                raise ValueError(f"Node '{n}' must have 'h' (holding cost).")
            if "I0" not in data:
                raise ValueError(f"Node '{n}' must have 'I0' (initial inventory).")

            if "C" in data:
                for required_attr in ("v", "o"):
                    if required_attr not in data:
                        raise ValueError(
                            f"Factory node '{n}' has capacity 'C' but is missing '{required_attr}'. "
                            f"Required factory attributes: C (capacity), v (yield), o (operating cost)."
                        )

        # Validate edge attributes
        for u, v, data in self.graph.edges(data=True):
            is_retail_edge = v in markets

            if is_retail_edge:
                if "L" in data:
                    raise ValueError(
                        f"Retail demand edge ({u} → {v}) must not define lead time 'L'. "
                        "Market-facing edges represent customer demand, not replenishment arcs."
                    )
                # Retail edges need demand distribution info
                if "demand_dist" not in data:
                    raise ValueError(
                        f"Retail edge ({u} → {v}) must have 'demand_dist'. " f"Example: demand_dist: poisson"
                    )
                if "dist_param" not in data:
                    raise ValueError(
                        f"Retail edge ({u} → {v}) must have 'dist_param'. " f"Example: dist_param: {{mu: 20}}"
                    )
                if "p" not in data:
                    raise ValueError(f"Retail edge ({u} → {v}) must have 'p' (unit price).")
                if "b" not in data:
                    raise ValueError(f"Retail edge ({u} → {v}) must have 'b' (backlog penalty).")
            else:
                # Reorder edges need lead time, price, pipeline holding cost
                for required_attr in ("L", "p", "g"):
                    if required_attr not in data:
                        raise ValueError(
                            f"Reorder edge ({u} → {v}) must have '{required_attr}'. "
                            f"Required attributes: L (lead time), p (price), g (pipeline holding cost)."
                        )
                L = data["L"]
                if isinstance(L, bool) or not isinstance(L, Integral) or int(L) < 0:
                    raise ValueError(
                        f"Reorder edge ({u} → {v}) must have a non-negative integer lead time 'L'. " f"Got {L!r}."
                    )

    def _auto_detect_levels(self) -> None:
        """Auto-detect echelon levels from graph structure for visualization."""
        markets = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        rawmats = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]

        # Retailers: nodes that directly supply a market
        retailers = []
        for m in markets:
            retailers.extend(self.graph.predecessors(m))
        retailers = list(set(retailers))

        # Factories: nodes with capacity
        factories = [n for n in self.graph.nodes() if "C" in self.graph.nodes[n]]

        # Distributors: have I0 but no capacity, and not retailers
        distributors = [
            n
            for n in self.graph.nodes()
            if "I0" in self.graph.nodes[n] and "C" not in self.graph.nodes[n] and n not in retailers
        ]

        self.levels = {
            "retailer": retailers,
            "distributor": distributors,
            "manufacturer": factories,
            "raw_materials": rawmats,
        }

    # ── Built-in Scenarios ────────────────────────────────────────────

    def _apply_metadata(self):
        """Attaches demand vectors to graph edges based on user config."""
        for link, d in self.user_D.items():
            if link in self.graph.edges:
                arr = np.asarray(d, dtype=float)
                if arr.ndim != 1:
                    raise ValueError(f"user_D for edge {link} must be one-dimensional, got shape {arr.shape}.")
                if len(arr) < self.num_periods:
                    raise ValueError(
                        f"user_D for edge {link} has length {len(arr)}; "
                        f"expected at least num_periods={self.num_periods}."
                    )
                if np.any(~np.isfinite(arr)):
                    raise ValueError(f"user_D for edge {link} contains NaN or infinite values.")
                self.graph.edges[link]["user_D"] = arr
                if link in self.sample_path:
                    self.graph.edges[link]["sample_path"] = self.sample_path[link]

    def _compile_indices(self) -> None:
        """Compiles static graph data into NumPy-friendly indices."""
        self.num_nodes = self.graph.number_of_nodes()

        # Node Classification
        # Market: Nodes with no successors
        self.market = [j for j in self.graph.nodes() if len(list(self.graph.successors(j))) == 0]
        # Distributor: Has Inventory (I0) but no Capacity (C)
        self.distrib = [j for j in self.graph.nodes() if "C" not in self.graph.nodes[j] and "I0" in self.graph.nodes[j]]
        # Retailer: Supplies the market
        self.retail = [
            j for j in self.graph.nodes() if len(set.intersection(set(self.graph.successors(j)), set(self.market))) > 0
        ]
        # Factory: Has Capacity (C)
        self.factory = [j for j in self.graph.nodes() if "C" in self.graph.nodes[j]]
        # Raw Material: Nodes with no predecessors
        self.rawmat = [j for j in self.graph.nodes() if len(list(self.graph.predecessors(j))) == 0]

        # Main Nodes for State Tracking (Sorted for consistency)
        self.main_nodes = np.sort(self.distrib + self.factory)

        # General Mappings
        self.all_nodes = list(self.graph.nodes())
        self.node_map = {n: i for i, n in enumerate(self.all_nodes)}
        self.idx_to_node = {i: n for n, i in self.node_map.items()}

        # Edge Lists
        self.reorder_links = [e for e in self.graph.edges() if e[1] not in self.market]
        self.retail_links = [e for e in self.graph.edges() if e[1] in self.market]
        self.network_links = list(self.graph.edges())

        # Edge Mappings
        self.reorder_map = {e: i for i, e in enumerate(self.reorder_links)}
        self.retail_map = {e: i for i, e in enumerate(self.retail_links)}
        self.network_map = {e: i for i, e in enumerate(self.network_links)}

        # Pre-compute Adjacency Indices for Fast Step execution
        self.pred_reorder_indices = {}
        self.succ_network_indices = {}

        for j in self.main_nodes:
            # Predecessors (Incoming Supply)
            preds = []
            for k in self.graph.predecessors(j):
                edge = (k, j)
                if edge in self.reorder_map:
                    # Store tuple: (predecessor_node_idx, reorder_edge_idx, Lead_Time)
                    preds.append((self.node_map[k], self.reorder_map[edge], self.graph.edges[edge]["L"]))
            self.pred_reorder_indices[j] = preds

            # Successors (Outgoing Sales)
            succs = []
            for k in self.graph.successors(j):
                edge = (j, k)
                if edge in self.network_map:
                    # Store tuple: (successor_node_idx, network_edge_idx)
                    succs.append((self.node_map[k], self.network_map[edge]))
            self.succ_network_indices[j] = succs

        # Validation
        expected = set.union(set(self.market), set(self.distrib), set(self.factory), set(self.rawmat))
        if set(self.graph.nodes()) != expected:
            raise ValueError(
                f"Node classification incomplete: graph has {set(self.graph.nodes())} "
                f"but classified only {expected}"
            )
        if self.graph.number_of_nodes() < 2:
            raise ValueError("Network must have at least 2 nodes")

    def _compute_space_limits(self):
        """Calculates limits needed for Gym spaces."""
        num_reorder_links = len(self.reorder_links)

        if num_reorder_links > 0:
            self.lt_max = np.max([self.graph.edges[e]["L"] for e in self.reorder_links])
            self.pipeline_length = sum([self.graph.edges[e]["L"] for e in self.reorder_links])
            self.lead_times = {e: self.graph.edges[e]["L"] for e in self.reorder_links}
        else:
            self.lt_max = 0
            self.pipeline_length = 0
            self.lead_times = {}

        self.init_inv_max = np.max([self.graph.nodes[j].get("I0", 0) for j in self.graph.nodes()])
        self.capacity_max = np.max([self.graph.nodes[j].get("C", 0) for j in self.graph.nodes()])

    def _build_network_scenario(self):
        """Builds the multi-echelon divergent supply-chain network."""
        # Nodes
        self.graph.add_nodes_from([0])
        self.graph.add_nodes_from([1], I0=100, h=0.030)
        self.graph.add_nodes_from([2], I0=110, h=0.020)
        self.graph.add_nodes_from([3], I0=80, h=0.015)
        self.graph.add_nodes_from([4], I0=400, C=90, o=0.010, v=1.000, h=0.012)
        self.graph.add_nodes_from([5], I0=350, C=90, o=0.015, v=1.000, h=0.013)
        self.graph.add_nodes_from([6], I0=380, C=80, o=0.012, v=1.000, h=0.011)
        self.graph.add_nodes_from([7, 8])

        # Edges
        self.graph.add_edges_from(
            [
                (1, 0, {"p": 2.000, "b": 0.100, "demand_dist": poisson, "dist_param": {"mu": 20}}),
                (2, 1, {"L": 5, "p": 1.500, "g": 0.010}),
                (3, 1, {"L": 3, "p": 1.600, "g": 0.015}),
                (4, 2, {"L": 8, "p": 1.000, "g": 0.008}),
                (4, 3, {"L": 10, "p": 0.800, "g": 0.006}),
                (5, 2, {"L": 9, "p": 0.700, "g": 0.005}),
                (6, 2, {"L": 11, "p": 0.750, "g": 0.007}),
                (6, 3, {"L": 12, "p": 0.800, "g": 0.004}),
                (7, 4, {"L": 0, "p": 0.150, "g": 0.000}),
                (7, 5, {"L": 1, "p": 0.050, "g": 0.005}),
                (8, 5, {"L": 2, "p": 0.070, "g": 0.002}),
                (8, 6, {"L": 0, "p": 0.200, "g": 0.000}),
            ]
        )

        # Levels for Visualization
        self.levels["retailer"] = np.array([1])
        self.levels["distributor"] = np.unique(
            np.hstack([list(self.graph.predecessors(i)) for i in self.levels["retailer"]])
        )
        self.levels["manufacturer"] = np.unique(
            np.hstack([list(self.graph.predecessors(i)) for i in self.levels["distributor"]])
        )
        self.levels["raw_materials"] = np.unique(
            np.hstack([list(self.graph.predecessors(i)) for i in self.levels["manufacturer"]])
        )

    def _build_serial_scenario(self) -> None:
        """Constructs a standard serial supply chain."""
        # Nodes
        self.graph.add_nodes_from([0])
        self.graph.add_nodes_from([1], I0=100, h=0.030)
        self.graph.add_nodes_from([2], I0=100, h=0.020)
        self.graph.add_nodes_from([3], I0=200, C=100, o=0.010, v=1.000, h=0.010)
        self.graph.add_nodes_from([4])

        # Edges
        self.graph.add_edges_from(
            [
                (1, 0, {"p": 2.0, "b": 0.1, "demand_dist": poisson, "dist_param": {"mu": 20}}),
                (2, 1, {"L": 4, "p": 1.5, "g": 0.010}),
                (3, 2, {"L": 4, "p": 1.0, "g": 0.005}),
                (4, 3, {"L": 0, "p": 0.5, "g": 0.000}),
            ]
        )

        # Levels for Visualization
        self.levels = {"retailer": [1], "distributor": [2], "manufacturer": [3], "raw_materials": [4]}
