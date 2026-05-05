# Comparison with Prior Work

This document describes how `gym-invmgmt` relates to and builds upon two prior open-source projects that established RL for operations research:

- **[OR-RL-Benchmarks](https://github.com/awslabs/or-rl-benchmarks)** (Balaji et al., 2019) — Bin Packing, Newsvendor, and Vehicle Routing
- **[OR-Gym](https://github.com/hubbs5/or-gym)** (Hubbs et al., 2020) — `InvManagement-v0/v1` and `NetworkManagement-v0/v1`

These projects form a lineage: OR-RL-Benchmarks (Balaji et al., 2019) introduced several RL environments for OR problems, and OR-Gym (Hubbs et al., 2020) built upon that work — adopting OR-RL-Benchmarks' Newsvendor, Bin Packing, and Vehicle Routing environments while adding new ones including the multi-echelon `NetworkManagement` environment. `gym-invmgmt` in turn builds directly on OR-Gym's `NetworkManagement`, extending it to support a wider range of research questions.

---

## Relationship to OR-Gym

OR-Gym's `NetworkManagement` environment is the direct ancestor of `gym-invmgmt`. The network topology, node/edge parameterization, and step sequence all originate from OR-Gym's design. The extensions in `gym-invmgmt` aim to make the environment more flexible for non-stationary demand research and more accessible for modern RL frameworks.

### Summary of Extensions

| Area | OR-Gym | gym-invmgmt |
|---|---|---|
| **API** | `gym.Env` (OpenAI Gym) | `gymnasium.Env` (Gymnasium ≥0.26) |
| **Demand modeling** | Single stationary distribution | Composable engine: trend, seasonal, shock, combined |
| **Goodwill dynamics** | Not included | Endogenous demand–service feedback loop |
| **External demand data** | User-specified arrays | Same, with guidance for M5, Favorita, etc. |
| **Custom topologies** | Modify Python source | YAML config files with validation |
| **State tracking** | Pandas DataFrames | NumPy arrays with index maps |
| **Observation space** | Demand + inventory + pipeline | Same, plus normalized time and sentiment |
| **Wrappers** | Not included | `IntegerActionWrapper`, `EpisodeLoggerWrapper` |
| **Visualization** | Basic node layout | Simple and detailed modes (with parameters) |
| **Documentation** | README + Jupyter notebook | MDP formulation, tutorials, parameter tables, YAML schema |
| **Tests** | Not included | 15 automated tests |

### Detailed Notes

#### Gymnasium API

OR-Gym was built on OpenAI Gym, which has since been deprecated in favor of the Gymnasium library maintained by the Farama Foundation. `gym-invmgmt` adopts the Gymnasium API (5-tuple return from `step()`, `reset(seed=...)` seeding) to remain compatible with actively maintained RL libraries such as Stable-Baselines3 ≥2.0 and Ray RLlib ≥2.0.

#### Composable Demand Engine

OR-Gym supports a single probability distribution for demand (e.g., Poisson with a fixed mean). This is appropriate for stationary environments but limits the study of agent robustness to non-stationary patterns. `gym-invmgmt` introduces a `DemandEngine` that supports composable effects — trend, seasonal, shock, and arbitrary combinations — enabling researchers to systematically study how agents handle different types of demand variability.

#### Endogenous Goodwill Dynamics

`gym-invmgmt` adds an optional goodwill feedback mechanism where sustained stockouts gradually reduce customer demand (sentiment decay), while consistent service slowly restores it (sentiment recovery). This creates an asymmetric feedback loop that tests whether agents can maintain long-term service quality, a dynamic not present in OR-Gym.

#### State Tracking

OR-Gym stores simulation matrices (`X`, `Y`, `R`, `S`, `D`, `U`, `P`) in Pandas DataFrames, which provide readable column labels but add overhead for per-step indexing. `gym-invmgmt` uses NumPy arrays with pre-computed index mappings (`node_map`, `reorder_map`, etc.), which generally reduces the simulation overhead. The tradeoff is somewhat less readable internal code, but the index maps are documented and accessible.

#### Custom YAML Topologies

In OR-Gym, creating a new network topology requires modifying the Python source code and understanding the internal graph construction. `gym-invmgmt` adds a YAML-based configuration system that parses network definitions, validates graph structure (DAG, connectivity, edge attributes), and auto-detects node roles. This lowers the barrier for researchers who want to experiment with different topologies without modifying code.

#### Design Decisions

Several deliberate changes were made relative to OR-Gym:

| Change | Rationale |
|---|---|
| Removed discount factor `alpha^t` from reward | In standard RL, discounting is typically handled by the algorithm's `gamma` parameter, not in the environment reward. Applying both can lead to confusing reward scales. |
| Continuous actions (integer rounding via wrapper) | Separates the rounding policy from environment physics, allowing researchers to choose when to enforce integer constraints. |
| Auto-detected echelon levels | Enables custom topologies — node roles are inferred from graph structure rather than manually specified. |
| Gymnasium `reset(seed=...)` seeding | Replaces the `seed_int` constructor parameter with modern Gymnasium seeding, compatible with vectorized environments. |

---

## Relationship to OR-RL-Benchmarks

OR-RL-Benchmarks (Balaji et al., 2019) was a pioneering project that introduced RL environments for several classic OR problems. Its environments — including the multi-period Newsvendor with lead times, Bin Packing, and Vehicle Routing — were subsequently adopted by OR-Gym (which credits Balaji et al. directly in its README for these three environments). In this sense, OR-RL-Benchmarks is the upstream ancestor of the entire lineage.

### The Newsvendor Environment

OR-RL-Benchmarks' Newsvendor (`News Vendor/src/news_vendor_environment.py`) is a multi-period, single-echelon inventory environment that includes several important design ideas:

- **Lead times**: configurable (default 5), with pipeline inventory tracked in the state
- **Randomized costs**: price, cost, holding cost, and lost-sale penalty randomized per episode — encouraging generalization
- **Action variants**: continuous, normalized `[0,1]`, and discrete versions of the same environment
- **Heuristic baselines**: `heuristic_baseline.py` and `multiperiod_vlt_heuristic.py` for benchmarking
- **PPO training**: example training scripts with Ray RLlib

Several of these ideas carry forward into the lineage. OR-Gym expanded the scope from single-echelon to multi-echelon network problems, and `gym-invmgmt` further extended that with composable demand, goodwill dynamics, and modern tooling.

### How gym-invmgmt Extends the Scope

The Newsvendor environment models a single-echelon ordering problem — an important and well-studied problem. The multi-echelon coordination problem addressed by `gym-invmgmt` builds on this foundation but adds network-level complexity:

| Aspect | OR-RL-Benchmarks Newsvendor | gym-invmgmt |
|---|---|---|
| Network structure | Single node | Multi-echelon DAG (configurable) |
| Echelons | 1 (retailer only) | Up to 4 (raw material → factory → distributor → retailer) |
| Shared suppliers | N/A | Factories may serve multiple downstream nodes |
| Production capacity | Not modeled | Per-factory capacity constraints |
| Pipeline holding cost | Not modeled | Per-edge cost parameter |
| Demand modeling | Stationary Poisson | Composable non-stationary effects |

The design philosophies are complementary: OR-RL-Benchmarks established the RL-for-OR benchmarking methodology across multiple problem types, OR-Gym expanded the inventory domain to multi-echelon networks, and `gym-invmgmt` focuses on depth within this domain — adding non-stationary demand, endogenous dynamics, and modern tooling.

---

## Benchmarking Fairness: Oracle vs. Observation-Parity Baselines

When comparing OR heuristics against RL agents, a subtle but important question arises: **do they receive the same information?**

### The Problem

Standard OR baselines (Newsvendor, (s,S)) are typically implemented by reading internal environment state directly:

```python
on_hand = env.X[period, node_idx]     # reads internal matrix
in_transit = env.Y[period, i]         # reads internal matrix
```

An RL agent, by contrast, only sees the observation vector returned by `env.step()`:

```python
obs, reward, _, _, _ = env.step(action)
# obs = [D[t-1], X[t], arrivals_from_R, features]
```

This creates an **information asymmetry**: the OR baseline acts as an oracle with perfect ground-truth access, while the RL agent must parse a flat vector.

### Our Approach: Both Modes

`gym-invmgmt` provides both modes in `examples/benchmark_agents.py`:

| Mode | Policy Classes | Info Source | Use Case |
|------|---------------|------------|----------|
| **Oracle** (†) | `NewsvendorPolicy`, `SSPolicy` | `env.X`, `env.Y` directly | Upper bound — "best a perfect-information policy can do" |
| **Obs-parity** (·obs) | `ObsNewsvendorPolicy`, `ObsSSPolicy` | `obs` vector via `ObsParser` | Fair comparison — same information budget as RL |

The `ObsParser` utility class pre-computes index slices at init time to parse the flat observation vector into per-node inventory and per-node pipeline totals in O(1) per step.

### Key Result

After fixing the observation bugs (lag-1 demand, R-based pipeline), the oracle and obs-parity results are **identical**:

```
  Newsvendor†          814     100.0%        121     5.13
  Newsvendor·obs       814     100.0%        121     5.13
```

This confirms that the observation vector is **informationally complete** — it contains exactly the same data as the internal matrices, just in a different format. The observation pipe introduces no information loss.

> **Note:** Both modes still use `env.demand_engine.get_current_mu()` for the demand distribution parameter. An RL agent must *learn* this from the demand history in the observation. A fully strict comparison would estimate μ from past demand, but that conflates policy quality with estimation quality.

### Why This Matters for Papers

When reporting RL-vs-OR comparisons:
- Use **oracle mode** to establish the theoretical upper bound
- Use **obs-parity mode** to demonstrate that any RL–OR gap is due to the *policy*, not the *information*
- If RL matches obs-parity but not oracle, the gap is in distribution knowledge (μ, σ), not state access

---

## Other Inspiring Works

Beyond the direct predecessors above, the following works have influenced or are relevant to the design and goals of this project.

### Foundational Operations Research

- **[Arrow, Harris & Marschak (1951)](https://www.jstor.org/stable/1907830)** — Formulated the single-period newsvendor problem and introduced the $(s, S)$ ordering policy, establishing the theoretical roots of inventory control. Published in *Econometrica*, Vol. 19, No. 3, pp. 250–272.
- **Clark & Scarf (1960)** — Derived optimal echelon base-stock policies for serial networks under stationary demand — a result that remains foundational for multi-echelon systems.
- **[Graves & Willems (2000)](https://mitsloan.mit.edu/shared/ods/documents?PublicationDocumentID=4372)** — Strategic safety stock placement in supply chain networks, extending the theory to general tree topologies. MIT Sloan Working Paper.

### RL for Inventory Management

- **[Oroojlooyjadid et al. (2022)](https://arxiv.org/abs/1708.05924)** — Applied deep Q-networks (DQN) to a beer-game-style serial supply chain, showing that RL can match or exceed simple base-stock heuristics. [GitHub](https://github.com/aoroojlooy/DQN-for-beer-game)
- **[Gijsbrechts et al. (2022)](https://arxiv.org/abs/1911.12959)** — Compared deep RL (A3C) against base-stock policies across lost-sales and multi-echelon settings, providing evidence for the competitiveness of RL. Published in *Manufacturing & Service Operations Management*.
- **[De Valva et al. (2023)](https://arxiv.org/abs/2209.07010)** — Analyzed theoretical conditions under which deep RL achieves near-optimal inventory performance.
- **[Meisheri et al. (2022)](https://arxiv.org/abs/2203.00490)** — Scalable multi-product inventory control using RL with lead-time constraints. Published in *Neural Computing and Applications*.
- **[Zhong (multi-echelon-drl)](https://github.com/qihuazhong/multi-echelon-drl)** — Heuristic-guided deep RL for multi-echelon inventory with both centralized and decentralized agent configurations, using YAML-based experiment setup.
- **[Liu et al. (MARL Multi-Echelon)](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management)** — Multi-agent deep RL for multi-echelon inventory, addressing cost reduction and bullwhip effect alleviation with serial and network topologies. [Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4262186)
- **[MABIM / ReplenishmentEnv](https://github.com/VictorYXL/ReplenishmentEnv)** — Multi-agent benchmark for inventory management with built-in OR baselines (base stock, $(s,S)$ policies) and MARL algorithms, supporting up to 200 SKUs.
- **[Dehossay (DRL_MMULS)](https://github.com/HenriDeh/DRL_MMULS)** — Deep RL for multi-item, multi-level uncapacitated lot-sizing. Written in Julia, exploring DRL in the lot-sizing variant of the inventory problem.

### Simulation-Based Optimization

- **[Anshul (multi-echelon-inventory-optimization)](https://github.com/anshul-musing/multi-echelon-inventory-optimization)** — SimPy-based discrete-event simulation for multi-echelon inventory, optimized with black-box methods (scipy, scikit-optimize, RBFOpt). Uses data-driven demand distributions bootstrapped from historical data. [Paper](https://arxiv.org/abs/1901.00090)

### Graph Neural Networks for Supply Chains

- **[Berto et al. (2024), RL4CO](https://github.com/ai4co/rl4co)** — GNN-augmented RL for combinatorial optimization problems, exploring scalable architectures for structured decision-making. [Paper](https://arxiv.org/abs/2306.17100)
- **[Kotecha & del Rio Chanona (2025)](https://github.com/nikikotecha/MultiAgentRL_InventoryControl)** — GNN-enhanced Multi-Agent PPO (MAPPO) for supply chain inventory control, with centralized learning and decentralized execution across networks of 6–24 agents. Published in *Computers & Chemical Engineering*. [Paper](https://www.sciencedirect.com/science/article/pii/S0098135425001152)

### Classical OR & Heuristic Tools

- **[Stockpyl](https://github.com/LarrySnyder/stockpyl)** (Snyder, 2023) — Python package for classical inventory optimization: EOQ, newsvendor, Wagner-Whitin, and multi-echelon optimization under both stochastic-service and guaranteed-service models. Companion to the textbook *Fundamentals of Supply Chain Theory*. [Docs](http://stockpyl.readthedocs.io/) · [INFORMS Tutorial](https://pubsonline.informs.org/doi/10.1287/educ.2023.0256)

### Imitation Learning

- **[Ross et al. (2011), DAgger](https://arxiv.org/abs/1011.0686)** — Iterative imitation learning that addresses distribution shift, applicable to supply chain settings where an OR-based oracle can serve as the expert.

---

> **Note:** This list is far from exhaustive. There are many more impactful works in inventory management that deserve recognition here. We apologize for not including them all and will continue to expand this section as the project evolves.
