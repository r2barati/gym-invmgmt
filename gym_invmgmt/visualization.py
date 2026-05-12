"""Visualization utilities for gym-invmgmt supply chain networks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from gym_invmgmt.network_topology import SupplyChainNetwork


def plot_network(
    network: SupplyChainNetwork,
    detailed: bool = False,
    save_path: Optional[str] = None,
    *,
    num_periods: int = 30,
    backlog: bool = True,
    n_actions: Optional[int] = None,
    n_obs: Optional[int] = None,
):
    """
    Visualize a supply chain network topology.

    Can be called standalone or via ``CoreEnv.plot_network()``.

    Args:
        network: A ``SupplyChainNetwork`` instance.
        detailed: If True, show node/edge parameters (costs, lead times, capacities).
        save_path: If provided, save the figure to this path instead of showing.
        num_periods: Episode length (for info box).
        backlog: Whether backlog mode is enabled (for info box).
        n_actions: Number of action dimensions (for info box).
        n_obs: Number of observation dimensions (for info box).

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    # Color palette
    level_colors = {
        "raw_materials": "#7f8c8d",  # Gray
        "manufacturer": "#2ecc71",  # Green
        "distributor": "#3498db",  # Blue
        "retailer": "#e74c3c",  # Red
        "market": "#f39c12",  # Orange
    }

    levels = {level: list(nodes) for level, nodes in network.levels.items()}
    markets = list(getattr(network, "market", []))
    if markets:
        levels["market"] = markets
    max_density = max(len(v) for v in levels.values())
    node_coords = {}

    fig, ax = plt.subplots(figsize=(14, 9) if detailed else (12, 7))

    # ── Draw Nodes ──
    for i, (level_name, node_ids) in enumerate(levels.items()):
        n = len(node_ids)
        node_ys = np.linspace(0.5, max_density - 0.5, n) if n > 1 else [max_density / 2]
        color = level_colors.get(level_name, "#95a5a6")

        for node_id, y in zip(node_ids, node_ys):
            node_coords[node_id] = (i, y)
            data = network.graph.nodes[node_id]

            # Node size based on role
            size = 400 if node_id in network.factory else (350 if node_id in network.retail else 300)
            ax.scatter(i, y, s=size, color=color, edgecolors="white", linewidths=2, zorder=5)

            # Label
            if detailed:
                parts = [f"Node {node_id}"]
                if "I0" in data:
                    parts.append(f"I₀={data['I0']}")
                if "C" in data:
                    parts.append(f"C={data['C']}")
                if "h" in data:
                    parts.append(f"h={data['h']:.3f}")
                if "o" in data:
                    parts.append(f"o={data['o']:.3f}")
                label = "\n".join(parts)
            else:
                label = str(node_id)

            ax.annotate(
                label,
                xy=(i, y),
                xytext=(0, 18 if detailed else 14),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9),
            )

    # ── Draw Edges ──
    for edge_idx, (u, v, data) in enumerate(network.graph.edges(data=True)):
        if u in node_coords and v in node_coords:
            start = node_coords[u]
            end = node_coords[v]

            # Find level of source for coloring
            source_level = next((k for k, val in levels.items() if u in val), None)
            color = level_colors.get(source_level, "#bdc3c7")

            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.6, connectionstyle="arc3,rad=0.05"),
            )

            if detailed:
                parts = []
                if "L" in data:
                    parts.append(f"L={data['L']}")
                if "p" in data:
                    parts.append(f"p=${data['p']:.2f}")
                if "g" in data and data["g"] > 0:
                    parts.append(f"g={data['g']:.3f}")
                if "b" in data:
                    parts.append(f"b=${data['b']:.2f}")
                if "dist_param" in data:
                    parts.append(f"μ={data['dist_param'].get('mu', '?')}")

                if parts:
                    mid_x = (start[0] + end[0]) / 2
                    mid_y = (start[1] + end[1]) / 2
                    offset_y = 0.08 * (edge_idx % 3 - 1)
                    ax.text(
                        mid_x,
                        mid_y + offset_y,
                        "  ".join(parts),
                        fontsize=7,
                        color="#2c3e50",
                        ha="center",
                        bbox=dict(facecolor="white", edgecolor="#bdc3c7", alpha=0.85, boxstyle="round,pad=0.2"),
                    )

    # ── Legend ──
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=level.replace("_", " ").title(),
            markerfacecolor=level_colors.get(level, "#95a5a6"),
            markersize=12,
        )
        for level in levels
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    # ── Formatting ──
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([k.replace("_", " ").title() for k in levels.keys()], fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.set_ylim(-0.3, max_density + 0.3)
    ax.set_title("Multi-Echelon Inventory Network Topology", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="x", linestyle="--", alpha=0.2)

    # Add global info box
    info_parts = [f"Periods: {num_periods}", f"Backlog: {backlog}"]
    if n_actions is not None:
        info_parts.append(f"Actions: {n_actions}")
    if n_obs is not None:
        info_parts.append(f"Obs: {n_obs}")
    ax.text(0.5, -0.06, "  |  ".join(info_parts), transform=ax.transAxes, ha="center", fontsize=10, color="#7f8c8d")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
    else:
        plt.show()

    return fig


def render_rgb_array(env, *, width: int = 640, height: int = 480) -> np.ndarray:
    """Render a compact dashboard frame for ``render_mode='rgb_array'``."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    base = env.unwrapped
    t = max(0, base.period - 1)
    main_nodes = list(base.network.main_nodes)
    main_indices = [base.network.node_map[n] for n in main_nodes]
    inv = base.X[t, main_indices] if t < len(base.X) else np.zeros(len(main_indices))
    demand = base.D[t, :] if t < len(base.D) else np.zeros(len(base.network.retail_links))
    backlog = base.U[t, :] if t < len(base.U) else np.zeros(len(base.network.retail_links))
    profit = float(np.sum(base.P[t, :])) if t < len(base.P) else 0.0

    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_title(f"Inventory State - Period {t + 1}/{base.num_periods}", fontsize=12, fontweight="bold")

    x = np.arange(len(main_nodes))
    ax.bar(x, inv, color="#3498db", label="Inventory")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in main_nodes], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Units")
    ax.grid(True, axis="y", alpha=0.2)

    info = (
        f"Demand: {np.round(demand, 1).tolist()}\n"
        f"Backlog: {np.round(backlog, 1).tolist()}\n"
        f"Profit: {profit:.2f}\n"
        f"Sentiment: {base.demand_engine.sentiment:.3f}"
    )
    ax.text(
        0.98,
        0.95,
        info,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#bdc3c7", boxstyle="round,pad=0.35", alpha=0.9),
    )
    fig.tight_layout()
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
