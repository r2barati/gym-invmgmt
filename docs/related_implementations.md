# Related Implementations: Supply Chain RL Environments

A technical comparison of open-source supply chain RL environment implementations.
This document focuses on **how each codebase implements** the simulation mechanics —
state tracking, action handling, observation construction, and KPI computation — to
inform design decisions in `gym-invmgmt`.

For the broader academic context (papers, lineage, classical OR foundations), see
[comparison_with_prior_work.md](comparison_with_prior_work.md).

---

## Index

| # | Codebase | Authors | Language | Paper |
|---|---|---|---|---|
| 1 | [OR-Gym](#1-or-gym) | Hubbs et al. | Python/Gym | [Hubbs et al., 2020](https://arxiv.org/abs/2008.06319) |
| 2 | [gym-invmgmt](#2-gym-invmgmt-this-project) | — | Python/Gymnasium | — |
| 3 | [ReplenishmentEnv](#3-replenishmentenv) | Victor YXL (MSR) | Python/Gym | [Jiang et al., 2023](https://arxiv.org/abs/2312.01230) |
| 4 | [multi-echelon-drl](#4-multi-echelon-drl) | Zhong | Python/Gym | — |
| 5 | [MADRL Serial](#5-madrl-serial) | Liu et al. | Python | [Liu et al., 2022](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4262186) |
| 6 | [Neural_inventory_control](#6-neural_inventory_control) | Alvo et al. | Python/PyTorch | [Alvo et al., 2023](https://arxiv.org/abs/2306.11246) |
| 7 | [MultiAgentRL_InventoryControl](#7-multiagentrl_inventorycontrol) | Kotecha & del Rio Chanona | Python/RLlib | [Kotecha & del Rio Chanona, 2025](https://www.sciencedirect.com/science/article/pii/S0098135425001152) |
| 8 | [DRL_MMULS](#8-drl_mmuls) | Dehondt et al. | Julia | [Dehondt et al., 2024](https://arxiv.org/abs/2310.15868) |

---

## 1. OR-Gym

**Repository**: https://github.com/hubbs5/or-gym  
**File**: `or_gym/envs/supply_chain/inventory_management.py`  
**Direct ancestor of `gym-invmgmt`.**

### Architecture

Serial supply chain with configurable number of stages (`m`). Flat array representation
— no graph structure. Each stage has inventory, pipeline, and capacity.

### State Variables

| Variable | Shape | Type | Description |
|---|---|---|---|
| `I[n, :]` | `(T+1, m-1)` | Stock | On-hand inventory at each stage |
| `T[n, :]` | `(T+1, m-1)` | Stock | Pipeline inventory (running balance) |
| `R[n, :]` | `(T, m-1)` | Flow | Filled replenishment orders |
| `D[n]` | `(T,)` | Flow | Customer demand (scalar, retailer only) |
| `S[n, :]` | `(T, m)` | Flow | Units sold at each stage |
| `B[n, :]` | `(T, m)` | Stock | Backlog at each stage |
| `LS[n, :]` | `(T, m)` | Flow | Lost sales at each stage |
| `P[n]` | `(T,)` | Flow | Total discounted profit per period |
| `action_log[n]` | `(T, m-1)` | Flow | Agent's raw, unconstrained request |

### Action Handling

```python
R = np.maximum(action, 0).astype(int)
action_log[n] = R.copy()           # raw request stored
R = R + self.B[n-1, 1:]            # add backlogged orders
R[R >= c] = c[R >= c]              # capacity constraint
R[R >= Im1] = Im1[R >= Im1]        # inventory constraint
self.R[n, :] = R                   # filled amount stored
```

### Observation Construction

```python
state[:m] = I[t]                                      # on-hand inventory
state[-m*lt_max:] += action_log[t-lt_max:t].flatten()  # pipeline from action_log
```

> **Known issue**: Uses `action_log` (requested) not `R` (filled) for pipeline
> observation. Agent sees what it *wished for*, not what is actually in transit.

### KPI Computation

No `compute_kpis()` function. Profit computed inline per step:
```python
P = alpha**n * sum(p*S - (r*RR + k*U + h*II))
```

### Validation

14 `assert` statements checking array shapes, non-negativity, distribution parameters.

---

## 2. gym-invmgmt (This Project)

**Repository**: https://github.com/r2barati/gym-invmgmt  
**Files**: `gym_invmgmt/core_env.py`, `gym_invmgmt/utils.py`, `gym_invmgmt/network_topology.py`

### Architecture

Graph-based multi-echelon DAG via NetworkX. Supports arbitrary topologies (serial,
divergent, diamond, custom YAML). Separate `DemandEngine` for composable demand dynamics
(stationary, shock, seasonal, trend).

### State Variables

| Variable | Shape | Type | Property Alias | Description |
|---|---|---|---|---|
| `X[t, j]` | `(T+1, n_nodes)` | Stock | `.inventory` | On-hand inventory at node *j* |
| `Y[t, i]` | `(T+1, n_reorder)` | Stock | `.pipeline` | Pipeline (in-transit) on reorder link *i* |
| `R[t, i]` | `(T, n_reorder)` | Flow | `.orders_filled` | Filled replenishment on reorder link *i* |
| `S[t, k]` | `(T, n_network)` | Flow | `.shipments` | Material shipped on all edges |
| `D[t, i]` | `(T, n_retail)` | Flow | `.demand` | Customer demand on retail link *i* |
| `U[t, i]` | `(T, n_retail)` | Stock | `.standing_backlog` | Standing backlog on retail link *i* |
| `P[t, j]` | `(T, n_nodes)` | Flow | `.profit` | Profit at node *j* |
| `action_log[t, i]` | `(T, n_reorder)` | Flow | `.orders_requested` | Agent's raw unconstrained request |

### Action Handling

```python
action_log[t] = action_arr          # raw request stored
# Then per-link constraints applied:
R[t, i] = min(request, available_inventory, capacity * yield)
S[t, net_idx] = R[t, i]            # same value, different index key
```

### Observation Construction

```python
obs = [D[t-1],          # lag-1 realized demand (not future demand)
       X[t+1],          # post-delivery on-hand inventory
       arrivals_from_R, # arrival-indexed pipeline via R matrix
       time_features]   # normalized period, demand sentiment
```

### KPI Computation

```python
total_sold = sum(S[:T, retail_indices])      # direct from sales matrix
total_demand = sum(D[:T, :])
fill_rate = total_sold / total_demand
total_backlog = sum(U[T-1, :])               # final standing backlog
```

### Validation

Edge attributes (`L`, `p`, `g`, `b`, `demand_dist`, `dist_param`) and node attributes
(`h`, `I0`, `C`, `v`, `o`) validated at YAML load time. DAG and connectivity checks.

### Known Issues (Fixed)

The following issues were identified through cross-implementation comparison and have been
resolved:

1. **Benchmark agents: missing backlog in inventory position** — `benchmark_agents.py`
   computed `inv_pos = on_hand + in_transit`, omitting the standing backlog. The correct
   formula `inv_pos = on_hand + in_transit − backlog` was already used in `run_heuristic.py`
   and in the ancestor OR-Gym (`IP = cumsum(I + T − B)`). All four policy classes
   (`NewsvendorPolicy`, `SSPolicy`, `ObsNewsvendorPolicy`, `ObsSSPolicy`) have been corrected.

2. **Demand engine: fixed `mu` parameter assumption** — `demand_engine.py` called
   `dist.rvs(mu=mu)`, which works only for `scipy.stats.poisson`. Distributions like `norm`
   (which expects `loc=`) and `uniform` (which expects `loc=`, `scale=`) raised `TypeError`.
   The sampling call now dispatches the correct parameter names per distribution type, matching
   OR-Gym's `dist.rvs(**dist_param)` pattern.

---

## 3. ReplenishmentEnv

**Repository**: https://github.com/VictorYXL/ReplenishmentEnv  
**Files**: `ReplenishmentEnv/env/agent_states.py`, `ReplenishmentEnv/env/replenishment_env.py`

### Architecture

4D state tensor: `(warehouses × state_items × dates × SKUs)`. Config-driven via YAML/JSON.
Supports multi-warehouse, multi-SKU scenarios (up to 200 SKUs). The most granular lifecycle
tracking of all implementations.

### State Variables

| State Item | Type | Description |
|---|---|---|
| `replenish` | Flow | Agent's order request (after action conversion) |
| `sale` | Flow | Actual units sold: `min(in_stock, demand)` |
| `in_transit` | Stock | Units currently in the pipeline |
| `arrived` | Flow | Units that arrived this period from upstream |
| `accepted` | Flow | Units accepted after storage capacity check |
| `excess` | Flow | Units rejected due to storage overflow |
| `in_stock` | Stock | Current on-hand inventory |
| `demand` | Flow | Customer demand |
| `selling_price` | Static | Sale price to downstream |
| `procurement_cost` | Static | Buy-in cost from upstream |
| `volume` | Static | Storage volume per SKU |
| `basic_holding_cost` | Static | Per-unit holding cost |
| `backlog_ratio` | Param | Backlog penalty ratio |
| `overflow_cost_ratio` | Param | Overflow penalty ratio |
| `vlt` | Static | Vendor lead time |

### Key Distinction: Full Lifecycle Tracking

**Only implementation that explicitly tracks every lifecycle stage:**

```
replenish → in_transit → arrived → accepted → in_stock → sale
                                  ↘ excess (overflow)
```

No other codebase models storage capacity overflow or acceptance rates.

### Step Sequence

```
Replenish → Sell → Receive arrived skus → Update balance
```

### Observation Construction

Configurable via `snapshot()` method — any combination of current-period and lookback
state items can be included.

---

## 4. multi-echelon-drl

**Repository**: https://github.com/qihuazhong/multi-echelon-drl  
**Files**: `envs.py`, `supplynetwork.py`, `network_components.py`

### Architecture

Object-oriented: `Node`, `Arc`, `SupplyNetwork` classes. Beer-game-style serial chain
(retailer → wholesaler → distributor → manufacturer). Supports both single-agent and
multi-agent configurations.

### State Variables

State is accessed via `sn.get_state(agent)` returning a dict:

| Key | Type | Description |
|---|---|---|
| `on_hand` | Stock | Current inventory at node |
| `unfilled_demand` | Stock | Standing backlog |
| `latest_demand` | Flow | Most recent demand received |
| `unreceived_pipeline_0..3` | Flow | Per-slot arrival schedule (FIFO deque positions) |

### Pipeline Representation

Uses **FIFO deques** on each `Arc` — `shipments` and `sales_orders` are deque objects.
Each period, the head of the deque arrives, and new orders/shipments are appended.
This naturally provides an arrival-indexed pipeline observation.

### Step Sequence

```
1. Advance order slips
2. Place orders (customers → suppliers)
3. Advance shipments (suppliers → customers)
4. Fulfill orders
```

### Action Handling

Agent calls `sn.agent_action(period, quantity)`. Quantity is passed to `node.place_order()`,
which internally constrains by available inventory. **No separate logging of
raw request vs. filled amount** — only the filled amount is visible.

### Key Feature: Strongly-Typed Node Definitions

```python
retailer = Node(
    name="retailer",
    initial_inventory=10,
    holding_cost=1.0,
    backorder_cost=10,
    policy=bsp_48,          # Base-stock policy for non-agent nodes
    is_demand_source=True,
    demands=demand_generator,
)
```

Every parameter is a required, explicit constructor argument. No risk of missing attributes.

---

## 5. MADRL Serial

**Repository**: https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management  
**File**: `envs/serial.py`

### Architecture

Minimalist. Python lists as FIFO queues. Hardcoded constants at module level.
Multi-agent with `LEVEL_NUM` agents, each controlling one echelon.

### State Variables

| Variable | Type | Description |
|---|---|---|
| `inventory[i]` | Stock (scalar) | On-hand inventory at echelon *i* |
| `backlog[i]` | Stock (scalar) | Standing backlog at echelon *i* |
| `order[i]` | List (FIFO, len=L) | Pipeline queue — `order[i][0]` arrives next |
| `action_history[i]` | List | Agent's raw request history |
| `current_orders` | List | Most recent actions for all agents |

### Pipeline Representation

`order[i]` is a Python list of length `LEAD_TIME`. Each step:
```python
self.order[i].append(new_order)   # new order enters pipeline
self.order[i] = self.order[i][1:] # oldest order exits (arrives)
```

This is a straightforward FIFO queue. The pipeline observation IS the queue contents.

### Action Handling

At the last echelon (manufacturer), `order = action[i]` (unlimited supply).
At other echelons, the filled amount is `min(demand, upstream_inv + arriving)`.
Raw actions stored in `action_history`.

### Observation

```python
obs = [inventory, backlog, demand_or_downstream_action, order[0], ..., order[L-1]]
```

Per-agent observation includes the full pipeline queue as individual slots.

### Key Feature: Bullwhip Effect Measurement

Computes `std(action_history) / mean(action_history)` (coefficient of variation)
per echelon during evaluation — a direct measure of the bullwhip effect.

---

## 6. Neural_inventory_control

**Repository**: https://github.com/MatiasAlvo/Neural_inventory_control  
**File**: `environment.py`

### Architecture

PyTorch tensor-based. Batched operations for efficient training. Supports multi-item
(multi-SKU) with shared learning. Two modes: lost-sales (clip inventory ≥ 0) and
backlog (allow negative inventory).

### State Variables

| Variable | Type | Description |
|---|---|---|
| `inventory_on_hand` | Stock (tensor) | Current inventory per item |
| `pipeline` | Flow (tensor, `n_items × L`) | FIFO pipeline — `pipeline[:, 0]` arrives next |

### Pipeline Representation

Tensor-based FIFO shift:
```python
pipeline[:, :-1] = pipeline[:, 1:]   # shift left
pipeline[:, -1] = new_orders          # append new order
arrived = pipeline[:, 0]              # head arrives
```

### Action Handling

**No constraints** — agent's order quantity is always fully filled. There is no distinction
between requested and filled amounts. This models an unlimited external supplier.

### Observation

```python
obs = [inventory_on_hand, pipeline[:, 0], pipeline[:, 1], ..., pipeline[:, L-1]]
```

Full arrival schedule as individual tensor slices.

### Key Feature: Neural Network as Policy

The environment is designed to be used with a custom neural network architecture
(`DeepNN` class) that maps state directly to ordering decisions. This is not a standard
RL loop — it's a supervised/hybrid approach using newsvendor-style loss functions.

---

## 7. MultiAgentRL_InventoryControl

**Repository**: https://github.com/nikikotecha/MultiAgentRL_InventoryControl  
**File**: `env3rundiv.py`

### Architecture

Ray RLlib `MultiAgentEnv`. Supports divergent network topologies (up to 18 nodes).
Configurable via Python dict. Uses `(s, S)` action space — agent outputs reorder
point and order-up-to level, not raw quantities.

### State Variables

| Variable | Shape | Type | Description |
|---|---|---|---|
| `inv[t, node]` | `(T+1, n_nodes)` | Stock | On-hand inventory |
| `order_r[t, node]` | `(T, n_nodes)` | Flow | Computed order quantity (from s,S logic) |
| `order_u[t, node]` | `(T+1, n_nodes)` | Stock | Unfulfilled orders / pipeline |
| `ship[t, node]` | `(T, n_nodes)` | Flow | Units shipped (= sold) |
| `acquisition[t, node]` | `(T, n_nodes)` | Flow | Units received from upstream (explicitly tracked) |
| `demand[t, node]` | `(T+1, n_nodes)` | Flow | Demand at each node |
| `backlog[t, node]` | `(T+1, n_nodes)` | Stock | Standing backlog |
| `ship_to_list[t]` | List of dicts | Flow | Per-node-to-node shipping detail |
| `time_dependent_state` | `(T, n_nodes, max_delay)` | Mixed | Arrival schedule per delay slot |

### Key Feature: Explicit Arrival Tracking

Both `acquisition` (what arrived) and `time_dependent_state` (when it will arrive)
are explicitly tracked. `acquisition` is computed from delayed `ship_to_list` entries.

### Key Feature: Inventory Capacity

```python
inv[t+1] = min(max(inv[t] + acquisition[t] - ship[t], 0), inv_max)
```

Inventory is **capped** at `inv_max`. Excess is silently lost (unlike ReplenishmentEnv
which tracks overflow via `excess`).

### Key Feature: Divergent Multi-Agent with Fair Distribution

When a node ships to multiple downstream nodes and supply is insufficient, uses a
round-robin distribution loop with backlog priority. This is the most sophisticated
fair-allocation mechanism among all implementations.

---

## 8. DRL_MMULS

**Repository**: https://github.com/HenriDeh/DRL_MMULS  
**File**: `src/testbed/multi-item/load_environment.jl`

### Architecture

Julia-based. Declarative environment construction using the `InventoryModels.jl` package.
Environment defined as a **Bill of Materials (BOM)** tree with typed components:
`EndProduct`, `Assembly`, `Supplier`, `Market`, `Inventory`.

### Environment Definition

```julia
item = Item(
    Inventory(h, init_inv),                    # holding cost, initial inventory
    Assembly(K, 0.0, components...,            # setup cost, components
             leadtime = LeadTime(LT, 0.0)),    # stochastic lead time
    policy = sSPolicy(),                       # ordering policy
)
```

### Key Feature: Multi-Item with BOM Structure

Supports raw materials → sub-assemblies → end products with explicit requirements
matrix (`Rij`). Resource constraints (`RessourceConstraint`) enforce shared capacity.

### Key Feature: Stochastic Lead Times

`LeadTime(mean, cv)` — lead times follow a distribution, not fixed integers.

### State Tracking

Handled entirely by `InventoryModels.jl` internals (not visible in the loader file).
The loader file is a factory function, not the environment itself.

---

## Cross-Implementation Comparison Tables

### Material Lifecycle Tracking

| Stage | gym-invmgmt | OR-Gym | ReplenishmentEnv | multi-echelon-drl | MADRL Serial | Neural_inv_ctrl | Kotecha MARL | DRL_MMULS |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **1. Raw request** | `action_log` | `action_log` | `replenish` | ERROR | `action_history` | N/A¹ | `order_r` | ? |
| **2. Filled order** | `R` | `R` | (upstream `sale`) | (in Node) | `order[-1]` | N/A¹ | ≈ requested | ? |
| **3. Pipeline (stock)** | `Y` | `T` | `in_transit` | ERROR (deque) | ERROR (list) | ERROR (tensor) | `order_u` | ? |
| **4. Arrival schedule** | OK (from R) | ERROR (from action_log) | ERROR | OK (deque slots) | OK (list slots) | OK (tensor slots) | OK `time_dep_state` | ? |
| **5. Explicit arrivals** | ERROR (implicit) | ERROR (implicit) | `arrived` | ERROR | ERROR | ERROR | `acquisition` | ? |
| **6. Acceptance/overflow** | ERROR | ERROR | `accepted`/`excess` | ERROR | ERROR | ERROR | ~ (inv capped) | ? |
| **7. Retail sales** | `S` | `S` | `sale` | ERROR (implicit) | ERROR (implicit) | ERROR (implicit) | `ship` | ? |

¹ No constraints — request = filled.

### Architecture Comparison

| Feature | gym-invmgmt | OR-Gym | ReplenishmentEnv | multi-echelon-drl | MADRL | Neural_inv | Kotecha | DRL_MMULS |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **API** | Gymnasium | Gym | Gym | Gym | Custom | Custom | RLlib | Julia |
| **Topology** | DAG (YAML) | Serial | Config | Serial | Serial | Single | Divergent | BOM tree |
| **Multi-agent** | ERROR | ERROR | OK | OK | OK | ERROR | OK | ERROR |
| **Multi-SKU** | ERROR | ERROR | OK (200) | ERROR | ERROR | OK | ERROR | OK |
| **Demand engine** | Composable | Fixed dist | Data-driven | Fixed dist | Generator | Fixed dist | Configurable | Forecast |
| **Fill rate KPI** | OK | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |
| **Input validation** | Thorough | Asserts | Config | Minimal | None | None | Basic | None |
| **Custom topology** | OK (YAML) | ERROR | ERROR | ERROR | ERROR | ERROR | OK (dict) | ~ (BOM) |
| **Capacity constraints** | OK | OK | ERROR | ERROR | ERROR | ERROR | OK | OK |
| **Stochastic lead times** | ERROR | ERROR | ERROR | ERROR | OK (loss) | ERROR | OK (noise) | OK |

### Pipeline Representation

| Approach | Implementations | Description |
|---|---|---|
| **Running balance** | gym-invmgmt (`Y`), OR-Gym (`T`), ReplenishmentEnv (`in_transit`), Kotecha (`order_u`) | Single scalar per link = total in-transit. Updated via `balance = balance − arrived + ordered`. |
| **FIFO queue** | multi-echelon-drl (deque), MADRL (list), Neural_inv_ctrl (tensor), Kotecha (`time_dep_state`) | L-length vector per link. Position 0 = arriving next step. Shifted each period. |
| **Both** | Kotecha | Maintains `order_u` (balance) internally AND `time_dep_state` (schedule) for observation. |
| **R-indexed arrivals** | gym-invmgmt | Uses `R[t−L+k, i]` to construct arrival schedule from the filled-orders matrix. |

---

## Source Code Links

| Codebase | Primary Environment File |
|---|---|
| OR-Gym | [inventory_management.py](https://github.com/hubbs5/or-gym/blob/master/or_gym/envs/supply_chain/inventory_management.py) |
| gym-invmgmt | [core_env.py](../../gym_invmgmt/core_env.py) · [utils.py](../../gym_invmgmt/utils.py) · [network_topology.py](../../gym_invmgmt/network_topology.py) |
| ReplenishmentEnv | [agent_states.py](https://github.com/VictorYXL/ReplenishmentEnv/blob/main/ReplenishmentEnv/env/agent_states.py) · [replenishment_env.py](https://github.com/VictorYXL/ReplenishmentEnv/blob/main/ReplenishmentEnv/env/replenishment_env.py) |
| multi-echelon-drl | [envs.py](https://github.com/qihuazhong/multi-echelon-drl/blob/main/envs.py) · [supplynetwork.py](https://github.com/qihuazhong/multi-echelon-drl/blob/main/supplynetwork.py) |
| MADRL Serial | [serial.py](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/envs/serial.py) |
| Neural_inventory_control | [environment.py](https://github.com/MatiasAlvo/Neural_inventory_control/blob/main/environment.py) |
| Kotecha MARL | [env3rundiv.py](https://github.com/nikikotecha/MultiAgentRL_InventoryControl/blob/master/env3rundiv.py) |
| DRL_MMULS | [load_environment.jl](https://github.com/HenriDeh/DRL_MMULS/blob/master/src/testbed/multi-item/load_environment.jl) |
