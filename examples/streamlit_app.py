"""
Supply Chain RL Explorer — Interactive Streamlit Dashboard
===========================================================
Launch:  streamlit run examples/streamlit_app.py
"""
from __future__ import annotations

import sys, os, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
from gym_invmgmt import CoreEnv
from examples.run_or_baselines import (
    ConstantOrderPolicy, NewsvendorPolicy, SSPolicy,
)
from examples.generate_videos import render_frame, compute_scales

# ═══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Supply Chain RL Explorer",
    page_icon="🔗",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Dark premium theme
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(160deg, #0a0a1a 0%, #0d1117 40%, #0f1923 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1529 0%, #111b2e 100%);
        border-right: 1px solid rgba(100, 140, 255, 0.1);
    }
    section[data-testid="stSidebar"] * {
        color: #c8d6e5 !important;
    }

    /* Headers */
    h1, h2, h3 { color: #e2e8f0 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(20, 30, 60, 0.6);
        border: 1px solid rgba(100, 140, 255, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetric"] label { color: #8899aa !important; font-size: 12px !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e8edf5 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #8899aa !important; }
    .stTabs [aria-selected="true"] { color: #64b5f6 !important; border-bottom-color: #64b5f6 !important; }

    /* Expanders */
    .streamlit-expanderHeader { color: #b0c4de !important; }

    /* General text */
    .stMarkdown p, .stMarkdown li, .stCaption { color: #94a3b8 !important; }

    /* Selectbox/slider labels */
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #8899aa !important; }

    /* Dividers — subtle */
    hr { border-color: rgba(100, 140, 255, 0.12) !important; }

    /* Status cards */
    .run-status {
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 13px;
        margin-bottom: 12px;
    }
    .run-status.success {
        background: rgba(46, 125, 50, 0.2);
        border: 1px solid rgba(46, 125, 50, 0.4);
        color: #81c784;
    }
    .run-status.info {
        background: rgba(30, 90, 180, 0.2);
        border: 1px solid rgba(30, 90, 180, 0.4);
        color: #90caf9;
    }

    /* Video container */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(100, 140, 255, 0.15);
    }

    /* Remove top padding */
    .block-container { padding-top: 1.5rem; }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1565c0, #1976d2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🔗 Supply Chain Explorer")
st.sidebar.markdown("---")

# ── Topology ──
st.sidebar.markdown("### 🏗️ Topology")
topo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "gym_invmgmt", "topologies")
yaml_files = sorted([f for f in os.listdir(topo_dir) if f.endswith(".yaml")])

scenario_options = ["network", "serial"] + [f"📄 {f}" for f in yaml_files]
scenario_choice = st.sidebar.selectbox("Scenario", scenario_options, index=0)

custom_yaml_path = None
if scenario_choice.startswith("📄"):
    fname = scenario_choice.replace("📄 ", "")
    custom_yaml_path = os.path.join(topo_dir, fname)
    scenario = "network"
else:
    scenario = scenario_choice

num_periods = st.sidebar.slider("Episode length", 10, 100, 30)

# ── Demand ──
st.sidebar.markdown("### Summary Demand")
demand_type = st.sidebar.selectbox("Pattern", ["stationary", "shock", "seasonal", "trend"])
base_mu = st.sidebar.slider("Base demand (μ)", 5, 100, 20)

demand_config: dict = {"type": demand_type, "base_mu": base_mu, "use_goodwill": False}
if demand_type == "shock":
    demand_config["shock_time"] = st.sidebar.slider("Shock at period", 1, num_periods - 1, num_periods // 2)
    demand_config["shock_mag"]  = st.sidebar.slider("Shock multiplier", 1.0, 5.0, 2.5, 0.1)
elif demand_type == "seasonal":
    demand_config["seasonal_amp"] = st.sidebar.slider("Amplitude", 0.1, 1.0, 0.5, 0.05)
elif demand_type == "trend":
    demand_config["trend_slope"] = st.sidebar.slider("Trend slope", -2.0, 2.0, 0.5, 0.1)

# Policy
st.sidebar.markdown("### Policy")
policy_options = [
    "Newsvendor (base-stock)",
    "(s, S) Reorder-point",
    "Constant Order",
    "Random",
    "Custom Pulse",
    "Zero (do nothing)",
]
ppo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models", "ppo_invmgmt.zip")
if os.path.exists(ppo_path):
    policy_options.insert(0, "Trained PPO (RL)")

policy_type = st.sidebar.selectbox("Agent", policy_options)

# Policy params
constant_q = 20.0
ss_service = 0.95
pulse_qty = 50
pulse_period = 0

if policy_type == "Constant Order":
    constant_q = st.sidebar.slider("Order qty per link", 1.0, 200.0, 20.0, 1.0)
elif policy_type == "(s, S) Reorder-point":
    ss_service = st.sidebar.slider("Service level", 0.80, 0.999, 0.95, 0.005, format="%.3f")
elif policy_type == "Custom Pulse":
    pulse_qty = st.sidebar.slider("Pulse quantity", 1, 500, 50)
    pulse_period = st.sidebar.slider("At period #", 1, num_periods, 2) - 1

seed = st.sidebar.number_input("Seed", 0, 9999, 42)

st.sidebar.markdown("---")
run_clicked = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_simulation():
    """Run the episode with current sidebar params and generate video."""
    env_kwargs = dict(scenario=scenario, demand_config=demand_config, num_periods=num_periods)
    if custom_yaml_path:
        env_kwargs["config_path"] = custom_yaml_path

    env = CoreEnv(**env_kwargs)

    # Build policy
    policy = None
    if policy_type == "Newsvendor (base-stock)":
        policy = NewsvendorPolicy(env)
    elif policy_type == "(s, S) Reorder-point":
        policy = SSPolicy(env, service_level=ss_service)
    elif policy_type == "Constant Order":
        policy = ConstantOrderPolicy(env, q=constant_q)
    elif policy_type == "Trained PPO (RL)":
        from stable_baselines3 import PPO
        model = PPO.load(ppo_path)

    obs, _ = env.reset(seed=seed)
    done, t = False, 0
    n_links = len(env.network.reorder_links)

    while not done:
        if policy_type == "Random":
            action = env.action_space.sample()
        elif policy_type == "Custom Pulse":
            action = np.full(n_links, pulse_qty, dtype=np.float32) if t == pulse_period else np.zeros(n_links, dtype=np.float32)
        elif policy_type == "Zero (do nothing)":
            action = np.zeros(n_links, dtype=np.float32)
        elif policy_type == "Trained PPO (RL)":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = policy.get_action(obs, t)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        t += 1

    return env


def generate_video(env, title):
    """Generate MP4 video from render_frame, return path."""
    import imageio.v2 as iio
    scales = compute_scales(env)
    T = env.num_periods
    fps = 8
    repeat = 6  # ~0.75s per period

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False,
                                      dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_videos"))
    with iio.get_writer(tmp.name, format='FFMPEG', fps=fps, codec='libx264',
                        quality=8, macro_block_size=1) as writer:
        for t_step in range(T):
            frame = render_frame(env, t_step, scales, title)
            for _ in range(repeat):
                writer.append_data(frame)

    return tmp.name, scales


# ═══════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

if "env" not in st.session_state:
    st.session_state.env = None
    st.session_state.video_path = None
    st.session_state.scales = None
    st.session_state.run_label = None


# ═══════════════════════════════════════════════════════════════════════
#  HANDLE RUN BUTTON
# ═══════════════════════════════════════════════════════════════════════

if run_clicked:
    label = f"{scenario_choice} · {demand_type} · {policy_type}"

    with st.spinner("⏳ Running simulation..."):
        env = run_simulation()

    with st.spinner("🎬 Generating video..."):
        video_path, scales = generate_video(env, label)

    st.session_state.env = env
    st.session_state.video_path = video_path
    st.session_state.scales = scales
    st.session_state.run_label = label
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════

st.markdown("# 🔗 Supply Chain RL Explorer")

env = st.session_state.env

if env is None:
    # ── Empty state ──
    st.markdown("""
    <div style="text-align: center; padding: 80px 40px;">
        <div style="font-size: 64px; margin-bottom: 20px;">📦</div>
        <h2 style="color: #64b5f6 !important; margin-bottom: 12px;">Ready to Explore</h2>
        <p style="color: #78909c !important; font-size: 16px; max-width: 500px; margin: 0 auto;">
            Configure the <strong>topology</strong>, <strong>demand</strong>, and <strong>policy</strong>
            in the sidebar, then click <strong>Run Simulation</strong> to begin.
        </p>
        <div style="margin-top: 32px; padding: 20px; background: rgba(30, 60, 120, 0.15);
                    border-radius: 12px; border: 1px solid rgba(100, 140, 255, 0.1);
                    display: inline-block; text-align: left;">
            <p style="color: #90caf9 !important; font-size: 13px; margin: 0;">
                <strong>Available Policies:</strong><br>
                Newsvendor · (s,S) Reorder-point · Constant Order<br>
                Random · Custom Pulse · Trained PPO (RL)
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════════════
#  RESULTS — Simulation has been run
# ════════════════════════════════════════════════════════════════════════

label = st.session_state.run_label
T = env.num_periods
net = env.network

# ── Run status ──
st.markdown(f"""
<div class="run-status success">
    OK &nbsp;<strong>{label}</strong> &nbsp;—&nbsp; {T} periods &nbsp;|&nbsp;
    Seed: {seed} &nbsp;|&nbsp;
    Nodes: {len(net.main_nodes)} &nbsp;|&nbsp;
    Links: {len(net.reorder_links)}
</div>
""", unsafe_allow_html=True)

# ── KPI summary ──
total_profit = np.sum(env.P)
total_dem = np.sum(env.D)
total_bl = np.sum(env.U)
fill_rate = (1 - total_bl / max(total_dem, 1)) * 100
avg_inv = np.mean(env.X[1:T + 1])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Profit", f"${total_profit:+,.0f}")
k2.metric("Fill Rate", f"{fill_rate:.1f}%")
k3.metric("Total Demand", f"{total_dem:,.0f}")
k4.metric("Avg Inventory", f"{avg_inv:,.0f}")
k5.metric("Total Backlog", f"{total_bl:,.0f}")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  TABS: Video | Frame Explorer | Charts | Data
# ════════════════════════════════════════════════════════════════════════

tab_video, tab_frames, tab_charts, tab_data = st.tabs([
    "🎬 Video Playback",
    "🔍 Frame Explorer",
    "📈 Interactive Charts",
    "📋 Raw Data",
])

# ── TAB 1: Video ──────────────────────────────────────────────────────
with tab_video:
    st.markdown(f"### 🎬 {label}")
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        with open(st.session_state.video_path, "rb") as f:
            st.video(f.read(), format="video/mp4")
        st.caption(f"📏 {T} periods × ~0.75s each = ~{T * 0.75:.0f}s video  |  "
                   f"6-panel dashboard: Flow · Inventory · Pipeline · Demand · Profit · Backlog")
    else:
        st.warning("No video available. Click Run Simulation.")


# ── TAB 2: Frame Explorer ────────────────────────────────────────────
with tab_frames:
    st.markdown("### 🔍 Explore Individual Frames")
    period = st.slider("Select Period", 1, T, 1, key="frame_period") - 1
    scales = st.session_state.scales

    # Render single frame
    frame_img = render_frame(env, period, scales, label)
    st.image(frame_img, use_container_width=True)

    # Period KPI bar
    step_p = np.sum(env.P[period])
    cum_p = np.sum(env.P[:period + 1])
    dem_t = np.sum(env.D[period])
    bl_t = np.sum(env.U[period])

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Step Profit", f"${step_p:+,.0f}")
    fc2.metric("Cumulative", f"${cum_p:+,.0f}")
    fc3.metric("Demand", f"{dem_t:.0f}")
    fc4.metric("Backlog", f"{bl_t:.0f}")


# ── TAB 3: Interactive Charts ─────────────────────────────────────────
with tab_charts:
    st.markdown("### 📈 Interactive Charts")

    # Row 1: Inventory + Profit
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 📦 Inventory Over Time")
        main_indices = [net.node_map[n] for n in net.main_nodes]
        main_names = [f"N{n}" for n in net.main_nodes]
        inv_df = pd.DataFrame(env.X[:T + 1, main_indices], columns=main_names)
        inv_df.index.name = "Period"
        st.line_chart(inv_df, height=300)

    with c2:
        st.markdown("#### 💰 Cumulative Profit")
        cum = np.cumsum(np.sum(env.P, axis=1))
        profit_df = pd.DataFrame({"Profit ($)": cum}, index=range(1, T + 1))
        profit_df.index.name = "Period"
        color = "#4caf50" if cum[-1] >= 0 else "#f44336"
        st.line_chart(profit_df, height=300, color=[color])

    # Row 2: Pipeline + Orders
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### 🚛 Pipeline (In-Transit)")
        link_names = [f"N{u}→N{v}" for u, v in net.reorder_links]
        pipe_df = pd.DataFrame(env.Y[:T], columns=link_names)
        pipe_df.index.name = "Period"
        st.line_chart(pipe_df, height=300)

    with c4:
        st.markdown("#### 📋 Orders Placed")
        ord_df = pd.DataFrame(env.R[:T], columns=link_names)
        ord_df.index.name = "Period"
        st.line_chart(ord_df, height=300)

    # Row 3: Backlog + Demand vs Sales
    c5, c6 = st.columns(2)

    with c5:
        st.markdown("#### 📉 Fill Rate & Backlog")
        bl_s = np.sum(env.U, axis=1)
        dem_s = np.sum(env.D, axis=1)
        fr_s = (1 - bl_s / np.maximum(dem_s, 1e-6)) * 100
        bl_df = pd.DataFrame({
            "Backlog": bl_s,
            "Fill Rate (%)": fr_s,
        }, index=range(1, T + 1))
        st.line_chart(bl_df, height=300, color=["#f44336", "#4caf50"])

    with c6:
        st.markdown("#### 🛒 Demand vs Sales Over Time")
        total_dem_t = np.sum(env.D, axis=1)
        # Total retail sales
        retail_ship_idx = [net.network_map[lnk] for lnk in net.retail_links if lnk in net.network_map]
        total_sales_t = np.sum(env.S[:, retail_ship_idx], axis=1) if retail_ship_idx else np.zeros(T)
        ds_df = pd.DataFrame({
            "Demand": total_dem_t,
            "Sales": total_sales_t,
        }, index=range(1, T + 1))
        st.line_chart(ds_df, height=300, color=["#ff9800", "#00bcd4"])


# ── TAB 4: Raw Data ──────────────────────────────────────────────────
with tab_data:
    st.markdown("### 📋 Raw Data Tables")

    # Network info
    with st.expander("🗺️ Network Details & Lead Times", expanded=False):
        lt_rows = []
        for (u, v), idx in sorted(net.reorder_map.items()):
            L = env.graph.edges[(u, v)].get('L', 0)
            lt_rows.append({"Link": f"N{u} → N{v}", "Lead Time": L, "Index": idx})
        st.dataframe(pd.DataFrame(lt_rows), hide_index=True, use_container_width=True)

    data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs(
        ["Inventory (X)", "Orders (R)", "Shipments (S)", "Profit (P)"]
    )

    all_nodes = [f"N{n}" for n in sorted(net.node_map.keys())]
    link_names = [f"N{u}→N{v}" for u, v in net.reorder_links]

    with data_tab1:
        st.dataframe(
            pd.DataFrame(env.X[:T + 1], columns=all_nodes).round(1),
            use_container_width=True,
        )
    with data_tab2:
        st.dataframe(
            pd.DataFrame(env.R[:T], columns=link_names).round(1),
            use_container_width=True,
        )
    with data_tab3:
        edge_names = [f"N{u}→N{v}" for u, v in net.graph.edges()]
        st.dataframe(
            pd.DataFrame(env.S[:T], columns=edge_names[:env.S.shape[1]]).round(1),
            use_container_width=True,
        )
    with data_tab4:
        st.dataframe(
            pd.DataFrame(env.P[:T], columns=all_nodes[:env.P.shape[1]]).round(2),
            use_container_width=True,
        )
