"""
generate_videos.py — Multi-scenario video generator for gym-invmgmt.

Runs 3 supply chain scenarios, renders a 6-panel live dashboard for each
period (with fixed axis scales), and saves them as .mp4 files.

Usage:
    PYTHONPATH=. python3 generate_videos.py

Output in demo_videos/:
    video_multiEchelon_stationary.mp4
    video_multiEchelon_shock.mp4
    video_serial_seasonal.mp4
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as iio
from pathlib import Path

import gymnasium as gym
import gym_invmgmt  # noqa: F401 (registers envs)

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0d0d1a"
PANEL   = "#12122b"
PANEL2  = "#181836"
TEXT    = "#e8eaf6"
GRAY    = "#546e7a"
GRID_C  = "#1e2240"
BLUE    = "#5c6bc0"
CYAN    = "#26c6da"
GREEN   = "#66bb6a"
RED     = "#ef5350"
ORANGE  = "#ffa726"
PURPLE  = "#ab47bc"
YELLOW  = "#ffee58"

# ── Config ────────────────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "title":      "Multi-Echelon Network  ·  Stationary Demand",
        "env_id":     "GymInvMgmt/MultiEchelon-v0",
        "env_kwargs": {},
        "filename":   "video_multiEchelon_stationary.mp4",
        "seed":       42,
    },
    {
        "title":      "Multi-Echelon Network  ·  Demand Shock at Period 15",
        "env_id":     "GymInvMgmt/MultiEchelon-v0",
        "env_kwargs": {
            "demand_config": {
                "type": "shock",
                "base_mu": 20,
                "shock_time": 15,
                "shock_mag": 2.5,
                "use_goodwill": False,
            }
        },
        "filename":   "video_multiEchelon_shock.mp4",
        "seed":       42,
    },
    {
        "title":      "Serial Supply Chain  ·  Seasonal Demand",
        "env_id":     "GymInvMgmt/Serial-v0",
        "env_kwargs": {
            "demand_config": {
                "type": "seasonal",
                "base_mu": 20,
                "seasonal_amp": 0.7,
                "use_goodwill": False,
            }
        },
        "filename":   "video_serial_seasonal.mp4",
        "seed":       7,
    },
]

REPEAT_FRAMES = 8    # each period frame is repeated N times
FPS           = 8    # so each period lasts REPEAT/FPS = 1.0 s
DPI           = 100
FIG_W, FIG_H  = 20, 11


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env_id: str, env_kwargs: dict, seed: int):
    """Run one full episode and return the unwrapped env (all history stored)."""
    env = gym.make(env_id, **env_kwargs)
    env.reset(seed=seed)
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.unwrapped


def compute_scales(env) -> dict:
    """Compute fixed y-axis bounds from the completed episode history."""
    all_inv = env.X[:env.num_periods + 1, :]
    all_dem = env.D
    all_s   = env.S
    all_y   = env.Y
    all_u   = env.U
    cum_p   = np.cumsum(np.sum(env.P, axis=1))

    return {
        "inv_max":          max(all_inv.max(), 1.0),
        "demand_max":       max(all_dem.max(), 1.0),
        "sales_max":        max(all_s.max(), 1.0),
        "pipeline_max":     max(all_y.max(), 1.0),
        "backlog_max":      max(all_u.max(), 1.0),
        "cum_profit_min":   cum_p.min(),
        "cum_profit_max":   max(cum_p.max(), 1.0),
    }


# ── Network flow panel ────────────────────────────────────────────────────────

def _node_positions(network) -> dict:
    """x∈[0,1] = echelon fraction, y∈[0,1] = position within echelon."""
    coords = {}
    levels = network.levels
    n_levels = max(len(levels) - 1, 1)
    for col_idx, (_, node_ids) in enumerate(levels.items()):
        n = len(node_ids)
        ys = np.linspace(0.12, 0.88, n) if n > 1 else [0.5]
        x  = col_idx / n_levels
        for node_id, y in zip(node_ids, ys):
            coords[node_id] = (x, y)
    return coords


ECHELON_COLORS = {
    "raw_materials": GRAY,
    "manufacturer":  GREEN,
    "distributor":   BLUE,
    "retailer":      RED,
}


def _node_color(inv_val: float, inv_max: float) -> tuple:
    """Green→Red gradient based on inventory depletion."""
    ratio = min(max(inv_val / max(inv_max, 1), 0), 1)
    return (0.9 * (1 - ratio), 0.75 * ratio, 0.15)


def draw_network_flow(ax, env, t: int, scales: dict) -> None:
    """Panel 1 — supply chain topology with animated flow and inventory colouring."""
    ax.set_facecolor(PANEL)
    ax.axis("off")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Supply Chain — Live Flow", color=TEXT, fontsize=9, pad=5)

    network   = env.network
    coords    = _node_positions(network)
    inv_max   = scales["inv_max"]

    # Log-scaled max for arrow width/alpha so small flows remain visible
    s_max_log = np.log1p(max(scales["sales_max"], 1))
    # Also compute log max for orders
    r_max_log = np.log1p(max(env.R.max(), 1)) if hasattr(env, 'R') else s_max_log

    # ── Edges: draw orders (orange, dashed) then shipments (cyan, solid) ──
    for u, v in network.graph.edges():
        if u not in coords or v not in coords:
            continue
        x0, y0 = coords[u]
        x1, y1 = coords[v]

        # Get order quantity on this edge (if it's a reorder link)
        order = 0.0
        if (u, v) in network.reorder_map and t < len(env.R):
            order = env.R[t, network.reorder_map[(u, v)]]

        # Get shipment quantity on this edge
        flow = 0.0
        if (u, v) in network.network_map and t < len(env.S):
            flow = env.S[t, network.network_map[(u, v)]]

        # ── Order arrow (orange, dashed, curved above) ────────────────
        if order > 0.5:
            o_ratio = np.log1p(order) / r_max_log if r_max_log > 0 else 0
            o_w     = 0.6 + 3.0 * o_ratio
            o_alpha = 0.35 + 0.65 * o_ratio
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", lw=o_w, color=ORANGE, alpha=o_alpha,
                    connectionstyle="arc3,rad=0.18",
                    linestyle="dashed",
                ),
            )

        # ── Shipment arrow (cyan, solid, curved below) ────────────────
        s_ratio = np.log1p(flow) / s_max_log if s_max_log > 0 else 0
        s_w     = 0.8 + 4.0 * s_ratio
        s_alpha = 0.30 + 0.70 * s_ratio
        s_col   = CYAN if flow > 0.5 else GRAY

        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", lw=s_w, color=s_col, alpha=s_alpha,
                connectionstyle="arc3,rad=-0.05",
            ),
        )

        # ── Label: "O:xx / S:xx" ──────────────────────────────────────
        if order > 0.5 or flow > 0.5:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.04
            parts = []
            if order > 0.5:
                parts.append(f"O:{order:.0f}")
            if flow > 0.5:
                parts.append(f"S:{flow:.0f}")
            label = " / ".join(parts)
            # Highlight unfulfilled orders in red
            label_col = RED if (order > 0.5 and flow < order - 0.5) else TEXT
            ax.text(mx, my, label, fontsize=5.5, color=label_col,
                    ha="center", va="center",
                    bbox=dict(facecolor=PANEL, edgecolor="none", alpha=0.7, pad=1))

    # ── Nodes ──────────────────────────────────────────────────────────
    read_t = min(t + 1, len(env.X) - 1)
    for node_id, (x, y) in coords.items():
        node_idx = network.node_map.get(node_id)
        inv  = env.X[read_t, node_idx] if node_idx is not None else 0.0
        col  = _node_color(inv, inv_max)
        size = 0.065

        circ = mpatches.Circle((x, y), size, color=col, zorder=5,
                                linewidth=1.5, ec="white")
        ax.add_patch(circ)
        ax.text(x, y,    f"{inv:.0f}", fontsize=7, color="white",
                ha="center", va="center", fontweight="bold", zorder=6)
        ax.text(x, y - 0.11, f"N{node_id}", fontsize=6, color=TEXT,
                ha="center", zorder=6)

    # ── Colour legend ──────────────────────────────────────────────────
    ax.text(0.0,  -0.04, "—", color=ORANGE, fontsize=9, transform=ax.transAxes)
    ax.text(0.04, -0.04, "Order", color=GRAY, fontsize=6.5, transform=ax.transAxes)
    ax.text(0.18, -0.04, "→", color=CYAN,   fontsize=9, transform=ax.transAxes)
    ax.text(0.22, -0.04, "Shipment", color=GRAY, fontsize=6.5, transform=ax.transAxes)
    ax.text(0.45, -0.04, "■", color=RED,    fontsize=9, transform=ax.transAxes)
    ax.text(0.49, -0.04, "Empty", color=GRAY, fontsize=6.5, transform=ax.transAxes)
    ax.text(0.65, -0.04, "■", color=GREEN,  fontsize=9, transform=ax.transAxes)
    ax.text(0.69, -0.04, "Full",  color=GRAY, fontsize=6.5, transform=ax.transAxes)


# ── Individual chart drawers ──────────────────────────────────────────────────

def draw_inventory_history(ax, env, t: int, scales: dict) -> None:
    """Panel 2 — inventory time series, all managed nodes, fixed y-scale."""
    network = env.network
    T       = env.num_periods

    # Assign a colour per echelon
    echelon_col = {}
    for k, node_ids in network.levels.items():
        c = ECHELON_COLORS.get(k, GRAY)
        for n in node_ids:
            echelon_col[n] = c

    x_end = min(t + 2, len(env.X))
    xs    = np.arange(x_end)
    for node_id in network.main_nodes:
        idx   = network.node_map[node_id]
        hist  = env.X[:x_end, idx]
        col   = echelon_col.get(node_id, GRAY)
        # bold for retailers, thin for raw-mat
        lw = 2.0 if node_id in network.retail else 1.0
        ax.plot(xs, hist, color=col, lw=lw, alpha=0.85, label=f"N{node_id}")

    ax.set_xlim(0, T)
    ax.set_ylim(0, scales["inv_max"] * 1.07)
    ax.axvline(t + 1, color=TEXT, lw=0.8, ls="--", alpha=0.4)
    ax.set_title("Inventory History — All Nodes", color=TEXT, fontsize=9, pad=5)
    ax.set_ylabel("Units", fontsize=8)
    ax.set_xlabel("Period", fontsize=8)

    if len(network.main_nodes) <= 9:
        leg = ax.legend(fontsize=5.5, facecolor=PANEL2, labelcolor=TEXT,
                        framealpha=0.9, ncol=3, loc="upper right",
                        handlelength=1.2)
        leg.get_frame().set_edgecolor(GRID_C)


def draw_pipeline(ax, env, t: int, scales: dict) -> None:
    """Panel 3 — pipeline (in-transit) per reorder link."""
    network = env.network
    T       = env.num_periods
    x_end   = min(t + 2, len(env.Y))
    xs      = np.arange(x_end)

    colors = plt.cm.tab20.colors
    for i, (u, v) in enumerate(network.reorder_links):
        ridx  = network.reorder_map[(u, v)]
        hist  = env.Y[:x_end, ridx]
        ax.plot(xs, hist, color=colors[i % len(colors)], lw=1.3,
                alpha=0.85, label=f"{u}→{v}")

    ax.set_xlim(0, T)
    ax.set_ylim(0, scales["pipeline_max"] * 1.07)
    ax.axvline(t + 1, color=TEXT, lw=0.8, ls="--", alpha=0.4)
    ax.set_title("Pipeline — In-Transit Units", color=TEXT, fontsize=9, pad=5)
    ax.set_ylabel("Units", fontsize=8)
    ax.set_xlabel("Period", fontsize=8)
    if len(network.reorder_links) <= 12:
        leg = ax.legend(fontsize=5.5, facecolor=PANEL2, labelcolor=TEXT,
                        framealpha=0.9, ncol=2, loc="upper right",
                        handlelength=1.2)
        leg.get_frame().set_edgecolor(GRID_C)


def draw_demand_sales(ax, env, t: int, scales: dict) -> None:
    """Panel 4 — current-period demand vs sales at retail links (fixed scale)."""
    network = env.network
    links   = network.retail_links
    n       = len(links)

    dem   = env.D[t, :]            if t < len(env.D) else np.zeros(n)
    sold  = np.array([
        env.S[t, network.network_map[lnk]]
        for lnk in links if lnk in network.network_map
    ]) if t < len(env.S) else np.zeros(n)
    if len(sold) < n:
        sold = np.zeros(n)

    x = np.arange(n)
    w = 0.36
    ax.bar(x - w / 2, dem,  w, color=ORANGE, alpha=0.85, label="Demand",
           edgecolor="none")
    ax.bar(x + w / 2, sold, w, color=CYAN,   alpha=0.85, label="Sales",
           edgecolor="none")

    # Annotate unfulfilled
    for i, (d, s) in enumerate(zip(dem, sold)):
        gap = d - s
        if gap > 0.5:
            ax.text(i, max(d, s) + 0.5, f"−{gap:.0f}", fontsize=6,
                    color=RED, ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in range(n)], fontsize=7)
    ax.set_ylim(0, scales["demand_max"] * 1.15)
    ax.set_title("Demand vs Sales — Retail Links", color=TEXT, fontsize=9, pad=5)
    ax.set_ylabel("Units", fontsize=8)
    leg = ax.legend(fontsize=7.5, facecolor=PANEL2, labelcolor=TEXT,
                    framealpha=0.9, loc="upper right")
    leg.get_frame().set_edgecolor(GRID_C)


def draw_profit(ax, env, t: int, scales: dict) -> None:
    """Panel 5 — cumulative profit over time (fixed full-episode scale)."""
    T = env.num_periods
    if t < 0:
        return

    cum = np.cumsum(np.sum(env.P[:t + 1, :], axis=1))
    xs  = np.arange(1, len(cum) + 1)

    col = GREEN if cum[-1] >= 0 else RED
    ax.fill_between(xs, cum, alpha=0.18, color=col)
    ax.plot(xs, cum, color=col, lw=2)
    ax.axhline(0, color=GRAY, lw=0.8, ls="--", alpha=0.5)
    ax.axvline(t + 1, color=TEXT, lw=0.8, ls="--", alpha=0.4)

    p_span = max(abs(scales["cum_profit_min"]), abs(scales["cum_profit_max"]))
    ax.set_ylim(-p_span * 1.12, p_span * 1.12)
    ax.set_xlim(1, T)
    ax.set_title("Cumulative Profit", color=TEXT, fontsize=9, pad=5)
    ax.set_ylabel("Profit ($)", fontsize=8)
    ax.set_xlabel("Period", fontsize=8)

    ax.text(0.03, 0.92, f"${cum[-1]:+,.0f}", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=col, va="top")


def draw_backlog_fillrate(ax, env, t: int, scales: dict) -> None:
    """Panel 6 — backlog history + fill-rate % on twin axis (fixed scale)."""
    T = env.num_periods
    if t < 0:
        return

    bl   = np.sum(env.U[:t + 1, :], axis=1)
    dem  = np.sum(env.D[:t + 1, :], axis=1)
    fr   = 1.0 - bl / np.maximum(dem, 1e-6)
    xs   = np.arange(1, len(bl) + 1)

    ax.fill_between(xs, bl, alpha=0.22, color=RED)
    ax.plot(xs, bl, color=RED, lw=2, label="Backlog")
    ax.axvline(t + 1, color=TEXT, lw=0.8, ls="--", alpha=0.4)
    ax.set_ylim(0, max(scales["backlog_max"] * 1.12, 1))
    ax.set_xlim(1, T)
    ax.set_title("Backlog & Fill Rate", color=TEXT, fontsize=9, pad=5)
    ax.set_ylabel("Units Backordered", color=RED, fontsize=7)
    ax.set_xlabel("Period", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(xs, fr * 100, color=GREEN, lw=1.8, ls="--", alpha=0.9,
             label="Fill %")
    ax2.set_ylim(-5, 115)
    ax2.set_ylabel("Fill Rate %", color=GREEN, fontsize=7)
    ax2.tick_params(colors=TEXT, labelsize=6.5)
    for sp in ax2.spines.values():
        sp.set_edgecolor(GRID_C)

    cur_fr = fr[-1] * 100
    ax2.text(0.97, 0.06, f"{cur_fr:.1f}%", transform=ax.transAxes,
             fontsize=10, color=GREEN, ha="right", va="bottom", fontweight="bold")


# ── Frame renderer ────────────────────────────────────────────────────────────

def render_frame(env, t: int, scales: dict, title: str) -> np.ndarray:
    T = env.num_periods

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG, dpi=DPI)
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30,
                            top=0.87, bottom=0.07, left=0.06, right=0.98)

    ax_net    = fig.add_subplot(gs[0, 0])
    ax_inv    = fig.add_subplot(gs[0, 1])
    ax_pipe   = fig.add_subplot(gs[0, 2])
    ax_demand = fig.add_subplot(gs[1, 0])
    ax_profit = fig.add_subplot(gs[1, 1])
    ax_blog   = fig.add_subplot(gs[1, 2])

    for ax in [ax_inv, ax_pipe, ax_demand, ax_profit, ax_blog]:
        ax.set_facecolor(PANEL2)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.xaxis.label.set_color(GRAY)
        ax.yaxis.label.set_color(GRAY)
        ax.grid(True, color=GRID_C, lw=0.5, alpha=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_C)

    # ── Header ─────────────────────────────────────────────────────────
    cum = np.cumsum(np.sum(env.P[:t + 1, :], axis=1))
    step_p  = np.sum(env.P[t, :]) if t < len(env.P) else 0
    dem_sum = np.sum(env.D[t, :]) if t < len(env.D) else 0
    bl_sum  = np.sum(env.U[t, :]) if t < len(env.U) else 0
    fill_r  = (1 - bl_sum / max(dem_sum, 1e-6)) * 100

    fig.text(0.5, 0.947, title, ha="center", fontsize=14,
             fontweight="bold", color=TEXT)
    fig.text(
        0.5, 0.912,
        f"Period {t + 1} / {T}   "
        f"│   Step Profit: ${step_p:+,.0f}   "
        f"│   Cumulative: ${cum[-1]:+,.0f}   "
        f"│   Fill Rate: {fill_r:.1f}%   "
        f"│   Sentiment: {env.GW[t]:.3f}",
        ha="center", fontsize=9, color=GRAY,
    )

    # Thin progress bar
    pbar = fig.add_axes([0.06, 0.904, 0.92, 0.007])
    pbar.barh(0.5, 1.0,           height=1, color=PANEL2)
    pbar.barh(0.5, (t + 1) / T,  height=1, color=BLUE)
    pbar.set_xlim(0, 1); pbar.set_ylim(0, 1); pbar.axis("off")

    # ── Draw panels ────────────────────────────────────────────────────
    draw_network_flow(ax_net,       env, t, scales)
    draw_inventory_history(ax_inv,  env, t, scales)
    draw_pipeline(ax_pipe,          env, t, scales)
    draw_demand_sales(ax_demand,    env, t, scales)
    draw_profit(ax_profit,          env, t, scales)
    draw_backlog_fillrate(ax_blog,  env, t, scales)

    # ── Rasterise ──────────────────────────────────────────────────────
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape((h, w, 3))
    plt.close(fig)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path("demo_videos")
    out_dir.mkdir(exist_ok=True)

    for sc in SCENARIOS:
        print(f"\n── {sc['title']}")
        print("   Running episode ...", flush=True)
        env = run_episode(sc["env_id"], sc["env_kwargs"], sc["seed"])
        scales = compute_scales(env)
        T = env.num_periods

        print(f"   Rendering {T} periods × {REPEAT_FRAMES} repeats "
              f"@ {FPS} fps → {T * REPEAT_FRAMES / FPS:.1f}s video ...", flush=True)

        frames = []
        for t in range(T):
            img = render_frame(env, t, scales, sc["title"])
            for _ in range(REPEAT_FRAMES):
                frames.append(img)
            if (t + 1) % 5 == 0:
                print(f"   ... period {t + 1}/{T}", flush=True)

        out_path = out_dir / sc["filename"]
        with iio.get_writer(str(out_path), format="FFMPEG", fps=FPS,
                            codec="libx264", quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)

        size_mb = out_path.stat().st_size / 1_048_576
        print(f"   OK Saved → {out_path}  ({size_mb:.1f} MB, "
              f"{len(frames)} frames)")

        env.close()

    print("\nAll done! Videos are in demo_videos/")


if __name__ == "__main__":
    main()
