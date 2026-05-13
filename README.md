# gym-invmgmt: An Open Benchmarking Framework for Inventory Management Methods

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/gym-invmgmt.svg)](https://pypi.org/project/gym-invmgmt/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-%E2%89%A50.26-orange.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **Gymnasium-compatible multi-echelon inventory management environment** for reinforcement learning and operations research.

This repository contains the standalone environment package only. The paper
benchmark agents, trained weights, result tables, and manuscript source live in
the companion benchmark repository.

## Project Links

- PyPI package: [gym-invmgmt](https://pypi.org/project/gym-invmgmt/)
- Project page: [r2barati.github.io/gym-invmgmt](https://r2barati.github.io/gym-invmgmt/)
- arXiv paper: [arXiv:2605.11355](https://arxiv.org/abs/2605.11355)
- Standalone environment package: [r2barati/gym-invmgmt](https://github.com/r2barati/gym-invmgmt)
- Paper/code repository: [r2barati/gym-invmgmt-paper](https://github.com/r2barati/gym-invmgmt-paper)
- Trained checkpoint archive: [rezabarati/gym-invmgmt-weights](https://huggingface.co/datasets/rezabarati/gym-invmgmt-weights)

**Tested with:**

| Framework | Version |
|---|---|
| [Stable Baselines3](https://stable-baselines3.readthedocs.io/) | ≥2.0 |
| [Gymnasium](https://gymnasium.farama.org/) | ≥0.26 |
| [Ray RLlib](https://docs.ray.io/en/latest/rllib/) | ≥2.0 |
| [CleanRL](https://docs.cleanrl.dev/) | — |

The environment simulates a configurable supply chain network with realistic logistics — production capacities, pipeline lead times, holding costs, backlog penalties — driven by a composable demand engine supporting non-stationary patterns and endogenous customer goodwill dynamics.

![Multi-Echelon Network Topology](https://raw.githubusercontent.com/r2barati/gym-invmgmt/main/assets/network_topology.png)

---

## Installation

```bash
pip install gym-invmgmt
```

To upgrade an existing installation:

```bash
pip install -U gym-invmgmt
```

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/r2barati/gym-invmgmt.git
```

**For development (editable install):**
```bash
git clone https://github.com/r2barati/gym-invmgmt.git
cd gym-invmgmt
pip install -e .
```

Release instructions for maintainers are in
[`docs/releasing_to_pypi.md`](docs/releasing_to_pypi.md).

---

## Repository Structure

```
gym-invmgmt/
├── gym_invmgmt/               ← Source code (all Python modules)
│   ├── core_env.py            ←   Gymnasium environment (step, reset, reward)
│   ├── demand_engine.py       ←   Non-stationary demand generation
│   ├── network_topology.py    ←   Graph builder (presets + YAML parser)
│   ├── visualization.py       ←   Network plotting
│   ├── utils.py               ←   Shared helpers (run_episode, compute_kpis)
│   ├── topologies/             ←   YAML network topology definitions
│   │   ├── assembly.yaml
│   │   ├── diamond.yaml
│   │   ├── distribution_tree.yaml
│   │   ├── divergent.yaml
│   │   ├── serial.yaml
│   │   └── w_network.yaml
│   └── wrappers/               ←   Action rounding, episode logging
├── examples/                  ← Runnable example scripts
│   ├── quickstart.py          ←   Minimal env usage
│   ├── run_heuristic.py       ←   Newsvendor base-stock policy
│   ├── run_or_baselines.py    ←   Classical OR policies comparison
│   ├── train_ppo.py           ←   PPO training with SB3
│   ├── benchmark_agents.py    ←   Multi-agent benchmark across scenarios
│   └── generate_videos.py     ←   Dashboard video generator
├── tests/                     ← Test suite
├── docs/                      ← Documentation
│   ├── guides/                ←   Tutorials and walkthroughs
│   └── reference/             ←   Technical reference docs
└── assets/                    ← Images for documentation
```

---

## Quick Start

```python
import gymnasium as gym
import gym_invmgmt

env = gym.make("GymInvMgmt/MultiEchelon-v0")
obs, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

---

## Environments

| Environment ID | Topology | Nodes | Echelons | Action Dim | Obs Dim |
|---|---|---|---|---|---|
| `GymInvMgmt/MultiEchelon-v0` | Divergent Network | 9 | 4 (Raw → Factory → Distributor → Retail) | 11 | 71 |
| `GymInvMgmt/Serial-v0` | Serial Chain | 5 | 4 (Raw → Factory → Distributor → Retail) | 3 | 15 |

Both environments default to 30-period episodes with stationary Poisson demand (μ=20).

---

## Environment Details

### Network Topology

The multi-echelon network features factories with production capacities, distributors as intermediary holding points, and retailers facing stochastic customer demand:

![Detailed Network Parameters](https://raw.githubusercontent.com/r2barati/gym-invmgmt/main/assets/network_detailed.png)

The serial chain provides a simpler linear topology for focused experiments:

![Serial Chain Topology](https://raw.githubusercontent.com/r2barati/gym-invmgmt/main/assets/serial_topology.png)

### Observation Space
`Box(-inf, inf, shape=(obs_dim,))` — a flat vector containing:
- **Demand**: Current realized demand at each retail link
- **Inventory**: On-hand inventory at each managed node (distributors + factories)
- **Pipeline**: In-transit quantities for each supply link, broken out by lead-time position
- **Extra Features**: Current time period `t` and demand sentiment `s` (goodwill multiplier)

### Action Space
`Box(0, max_order, shape=(n_reorder_links,))` — continuous order quantities for each supply link.

> **Note for RL practitioners**: The upper bound is a conservative theoretical maximum (initial inventory + capacity × horizon). In practice, meaningful orders are much smaller. If using PPO/SAC out-of-the-box, consider wrapping with `gymnasium.wrappers.RescaleAction(env, -1, 1)` to normalize the action range, or use `IntegerActionWrapper` for discrete order quantities.

### Reward Function
Each step returns the network-wide profit:

```
Reward = Σ (Revenue − Purchasing Cost − Holding Cost − Operating Cost − Backlog Penalty)
```

- **Revenue**: Selling price × units sold at retail
- **Purchasing Cost**: Unit cost × units ordered from upstream
- **Holding Cost**: Per-unit cost for on-hand inventory and in-transit pipeline
- **Operating Cost**: Factory variable cost per unit produced
- **Backlog Penalty**: Per-unit penalty for unmet demand

### Episode Termination
Episodes truncate after `num_periods` steps (default 30). There is no early termination.

---

## Configuration

All parameters are configurable via `gymnasium.make()` kwargs or direct `CoreEnv()` instantiation:

### Network Topologies
- `scenario='network'` — Multi-echelon divergent network (3 factories, 2 distributors, 1 retailer)
- `scenario='serial'` — Serial supply chain (1 factory → 1 distributor → 1 retailer)
- `scenario='custom'` — **User-defined topology** loaded from a YAML config file (see below)

### Demand Scenarios
The `DemandEngine` supports composable non-stationary effects:

| Parameter | Description | Default |
|---|---|---|
| `type` | Demand profile: `'stationary'`, `'trend'`, `'seasonal'`, `'shock'` | `'stationary'` |
| `effects` | Composable list: `['trend', 'seasonal']` applies both simultaneously | — |
| `base_mu` | Base mean demand | `20` |
| `external_series` | NumPy array of per-period demand (replaces `base_mu` with real data) | `None` |
| `use_goodwill` | Enable endogenous demand–service feedback loop | `False` |
| `noise_scale` | Variance multiplier (0.0 = deterministic, 1.0 = default) | `1.0` |

**Example — shock demand with goodwill:**
```python
env = gym.make("GymInvMgmt/MultiEchelon-v0",
    demand_config={
        'type': 'shock',
        'base_mu': 25,
        'use_goodwill': True,
        'shock_time': 15,
        'shock_mag': 2.0,
    },
    num_periods=50,
)
```

**Example — real-world demand data (e.g., M5 competition):**
```python
import numpy as np
real_demand = np.loadtxt("my_sales_data.csv")  # one value per period

env = gym.make("GymInvMgmt/MultiEchelon-v0",
    demand_config={
        'type': 'stationary',
        'base_mu': float(np.mean(real_demand)),
        'external_series': real_demand,
    },
    num_periods=len(real_demand),
)
```

> **Note:** When overriding `demand_config` via `gym.make()`, provide the **full dictionary** — partial dictionaries replace the registered defaults entirely rather than merging recursively.

### Fulfillment Modes
- `backlog=True` (default) — Unmet demand accumulates as backlog, penalized each period
- `backlog=False` — Unmet demand is lost immediately (lost sales model)

---

## Custom Network Topologies

Beyond the two built-in presets, you can define **any** topology via a YAML config file:

| Approach | When to Use | How |
|----------|------------|-----|
| **Built-in presets** | Benchmarking, reproducible experiments | `scenario='network'` or `scenario='serial'` |
| **YAML config file** | Custom topologies without changing Python code | `scenario='custom', config_path='...'` |

```python
from gym_invmgmt import make_custom_env

# Load a custom topology — diamond network with parallel factories
env = make_custom_env('gym_invmgmt/topologies/diamond.yaml', num_periods=30)
obs, info = env.reset(seed=42)
```

The built-in presets (`_build_network_scenario()`, `_build_serial_scenario()`) define topologies in Python. The YAML parser (`_build_custom_scenario()`) reads the same node/edge structure from a config file, auto-detects node roles, and validates the graph.

See [`gym_invmgmt/topologies/`](gym_invmgmt/topologies/) for ready-to-use YAML files, and the full [YAML schema reference](docs/reference/network_topologies.md#defining-custom-topologies) for all supported parameters.

---

## Visualization

```python
from gym_invmgmt import CoreEnv

env = CoreEnv(scenario='network')
env.plot_network()              # Simple topology view
env.plot_network(detailed=True) # With costs, lead times, capacities
```

---

## Wrappers

The package includes two agent-agnostic wrappers:

### `IntegerActionWrapper`
Rounds continuous actions to integers for physical realism (you can't order 3.7 units):
```python
from gym_invmgmt import IntegerActionWrapper
env = IntegerActionWrapper(env)
```

### `EpisodeLoggerWrapper`
Saves full trajectory matrices (inventory, demand, orders, profit, backlog) as `.npz` files:
```python
from gym_invmgmt import EpisodeLoggerWrapper
env = EpisodeLoggerWrapper(env, log_dir="./logs", run_name="experiment_1")
```

### Action Scaling for Deep RL
When training with PPO, SAC, or other deep RL algorithms, normalize the action space to `[-1, 1]`:
```python
from gymnasium.wrappers import RescaleAction

env = CoreEnv(scenario='network')
env = RescaleAction(env, min_action=-1.0, max_action=1.0)
# Agent now outputs actions in [-1, 1], automatically mapped to valid order quantities
```

---

## Architecture

```
CoreEnv (Gymnasium Environment)
├── Simulation Dynamics         # Order placement, delivery, demand fulfillment
│   ├── Allocation Logic        # Raw-material, distribution, factory constraints
│   ├── Pipeline Advancement    # Lead-time delayed deliveries & inventory update
│   └── Profit Calculation      # Revenue – procurement – holding – ops – penalty
├── State & Spaces              # Observation vector, action/observation spaces
├── SupplyChainNetwork          # Topology: nodes, edges, lead times, capacities
│   ├── _build_network_scenario()   # Built-in: multi-echelon divergent graph
│   ├── _build_serial_scenario()    # Built-in: serial supply chain
│   └── _build_custom_scenario()    # Custom: loads from YAML config file
├── DemandEngine                # Non-stationary demand generation
│   ├── Composable Effects      # Trend / Seasonal / Shock (can combine)
│   ├── External Series         # Real-world data injection (M5, Favorita, etc.)
│   └── Endogenous Goodwill     # Service-dependent demand feedback
└── Wrappers
    ├── IntegerActionWrapper     # Discrete order rounding
    └── EpisodeLoggerWrapper     # Trajectory recording
```

---

## Documentation

For detailed mathematical formulations and parameter references:

- **[MDP Formulation](docs/reference/mdp_formulation.md)** — State space, action space, transition dynamics, and reward function with equations
- **[Demand Engine](docs/reference/demand_engine.md)** — Composable non-stationary effects (trend, seasonal, shock) and endogenous goodwill dynamics
- **[External Datasets](docs/reference/external_datasets.md)** — Using real-world demand data (M5, Favorita, Rossmann) with recommended datasets and examples
- **[Network Topologies](docs/reference/network_topologies.md)** — Complete node/edge parameter tables, both built-in presets and YAML custom topologies
- **[Comparison with Prior Work](docs/comparison_with_prior_work.md)** — How this project relates to OR-Gym and OR-RL-Benchmarks

---

## Tutorial & Examples

**[Getting Started Tutorial](docs/guides/getting_started_tutorial.md)** — A comprehensive walkthrough covering:

| Section | What You'll Learn |
|---|---|
| The Problem | Why multi-echelon inventory management is hard |
| Network Exploration | Visualizing topology, understanding nodes & edges |
| Single Step Walkthrough | What happens inside `env.step()` |
| Constant Order Policy | Running a simple baseline and visualizing results |
| Lead Time Physics | Impulse response validation |
| Reward Breakdown | Decomposing profit into revenue, holding, and penalty |
| Demand Scenarios | Comparing stationary, trend, seasonal, shock, and combined |
| Bullwhip Effect | Observing order variance amplification across echelons |
| Goodwill Dynamics | Service-level feedback loops |
| Configuration Cookbook | Ready-to-use recipes for custom experiments |

**[Visual Dynamics Guide](docs/guides/visual_dynamics_guide_network.md)** — See exactly how decisions affect every node:

| Visualization | What It Shows |
|---|---|
| Inventory Heatmap | On-hand stock at every node across time |
| Actions vs Filled | Requested orders vs capacity-constrained fulfillment |
| Pipeline Heatmap | In-transit units per link over time |
| KPI Dashboard | Demand, sales, fill rate, backlog, per-step and cumulative profit |
| Node Profit Breakdown | Per-node contribution to system profit |


---

## Citing

If you use this environment in your research, please cite:

```bibtex
@misc{gym-invmgmt,
  author = {Barati, Reza},
  title = {gym-invmgmt: An Open Benchmarking Framework for Inventory Management Methods},
  year = {2026},
  howpublished = {\url{https://github.com/r2barati/gym-invmgmt}},
}
```

This environment builds on foundational work by [ORL Benchmarks](https://github.com/awslabs/or-rl-benchmarks) (Balaji et al., 2019), [OR-Gym](https://github.com/hubbs5/or-gym) (Hubbs et al., 2020), and [Perez et al. (2021)](https://www.mdpi.com/2227-9717/9/1/102). See [`CITATION.cff`](CITATION.cff) for full references.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
