# MDP Formulation

This document provides the formal Markov Decision Process (MDP) definition for the multi-echelon inventory management environment.

---

## Overview

The environment models a **multi-period, multi-echelon production–inventory system** for a single non-perishable product sold in discrete quantities. The agent (decision-maker) acts as a centralized planner who decides replenishment order quantities at every node in each time period.

The MDP is defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$.

---

## Notation

All variables used throughout this document are defined here. Each variable is traced to the code entity where it is set.

### State and Decision Variables

These are the matrices defined in `core_env.py` at reset. Rows are time periods; columns are nodes or edges.
Variables are classified as **stocks** (cumulative balances — read a single `[t]` row) or **flows** (per-period events — can be summed across time). Each variable also has a descriptive `@property` alias for external consumers.

| Symbol | Array Shape | Type | Property Alias | Description | Code Variable |
|--------|------------|------|----------------|-------------|---------------|
| $X_t^{(j)}$ | `(T+1, n_nodes)` | Stock | `.inventory` | On-hand inventory at node $j$, start of period $t$ | `self.X` |
| $Y_t^{(e)}$ | `(T+1, n_reorder)` | Stock | `.pipeline` | Pipeline (in-transit) inventory on reorder edge $e$ | `self.Y` |
| $U_t^{(e)}$ | `(T, n_retail)` | Stock | `.standing_backlog` | Unfulfilled (backlogged) demand on retail edge $e$ at end of period $t$ | `self.U` |
| $R_t^{(e)}$ | `(T, n_reorder)` | Flow | `.orders_filled` | **Filled** order quantity on reorder edge $e$ at period $t$ (may be less than agent's request) | `self.R` |
| $S_t^{(e)}$ | `(T, n_network)` | Flow | `.shipments` | Sales/shipment on edge $e$ at period $t$ | `self.S` |
| $D_t^{(e)}$ | `(T, n_retail)` | Flow | `.demand` | Realized customer demand on retail edge $e$ at period $t$ | `self.D` |
| $a_t^{(e)}$ | `(T, n_reorder)` | Flow | `.orders_requested` | Agent's raw unconstrained order request at period $t$ | `self.action_log` |
| $P_t^{(j)}$ | `(T, n_nodes)` | Flow | `.profit` | Profit contribution of node $j$ at period $t$ | `self.P` |

> **Stock vs. Flow**: Stock variables represent cumulative balances and should be read at a single time index (`X[t, j]`). Flow variables represent per-period events and can be summed across time for totals (`sum(D[:T, :])`). Never sum a stock variable across time.

### Node Parameters

Defined per node in `network_topology.py → _build_network_scenario()`. Each node has a **type** that determines which parameters apply.

| Symbol | Code Key | Applies To | Description |
|--------|----------|-----------|-------------|
| $I_0^{(j)}$ | `I0` | Factory, Distributor | Initial inventory at node $j$ when episode starts |
| $h_j$ | `h` | Factory, Distributor | Holding cost per unit per period at node $j$ |
| $C_j$ | `C` | Factory only | Maximum production capacity per period |
| $o_j$ | `o` | Factory only | Variable operating cost per unit produced |
| $v_j$ | `v` | Factory only | Production yield ($v \in (0,1]$; 1.0 = no loss) |

**Node types** (classified in `_compile_indices()`):

| Type | Rule | Role |
|------|------|------|
| Raw Material | No predecessors | Unlimited supplier (no inventory tracking) |
| Factory | Has `C` attribute | Produces goods; inventory + capacity constrained |
| Distributor | Has `I0` but no `C` | Warehouses; inventory constrained only |
| Retailer | Supplies a Market node | Sells to customers; subject to demand |
| Market | No successors | Demand sink (not a real node; represents customers) |

### Edge Parameters

Defined per edge in `_build_network_scenario()`. There are two types of edges: **reorder edges** (supply chain) and **retail edges** (customer-facing).

| Symbol | Code Key | Edge Type | Description |
|--------|----------|----------|-------------|
| $L_e$ | `L` | Reorder | Lead time in periods (order placed at $t$ arrives at $t + L$) |
| $p_e$ | `p` | Both | Unit price: selling price (downstream) or purchasing cost (upstream) |
| $g_e$ | `g` | Reorder | Pipeline holding cost per unit per period while in transit |
| $b_e$ | `b` | Retail | Backlog penalty per unfulfilled unit per period |

> **How to tell edge types apart:** Reorder edges have an `L` (lead time) attribute. Retail edges do not — they connect a retailer to a market node.

### Environment Parameters

Set in `CoreEnv.__init__()`.

| Symbol | Code Key | Default | Description |
|--------|----------|---------|-------------|
| $T$ | `num_periods` | 30 | Number of time periods (episode length) |
| $\alpha$ | `alpha` | 1.0 | Internal discount factor applied to profit at each step |
| $\mu$ | `base_mu` | 20 | Base Poisson demand mean (in `demand_config`) |

---

## State Space

The state at time $t$ is a flat vector concatenating four components:

$$s_t = [D_{t-1},\ X_t,\ \text{arrivals}_t,\ F_t]$$

| Component | Dimension | Source | Description |
|-----------|-----------|--------|-------------|
| Demand $D_{t-1}$ | $n_\text{retail}$ | `self.D[t-1, :]` | **Lag-1 realized demand** — the demand that was realized in the previous period. At $t=0$ this is all zeros (no demand realized yet). |
| Inventory $X_t$ | $n_\text{main}$ | `self.X[t, main_nodes]` | On-hand inventory at factories + distributors (post-delivery) |
| Arrivals | $\sum_e L_e$ | `self.R[t-L+k, e]` | **Arrival-indexed pipeline** from the $R$ matrix: for each reorder edge with lead time $L$, an $L$-element vector where `arrivals[0]` = units arriving this step (oldest order), `arrivals[k]` = units arriving in $k$ steps |
| Features $F_t$ | 2 | `demand_engine.get_observation(t)` | `[t / T, sentiment]` — normalized time and goodwill |

> **Key design decisions:**
> - **Lag-1 demand** prevents information leakage: the agent sees demand that *already happened*, not demand that will be sampled during the current step.
> - **R-based arrivals** use the filled order matrix (not `action_log`) to build the pipeline. This ensures the agent sees what is *actually in transit*, not what it *wished for*.
> - At reset ($t=0$), the demand component is all zeros because no demand has been realized yet.

**Total observation dimension:**

$$|\mathcal{S}| = n_\text{retail} + n_\text{main} + \sum_e L_e + 2$$

**Default network:** $|\mathcal{S}| = 1 + 7 + 60 + 2 = 70$, where:
- 1 retail edge (node 1 → market 0)
- 7 main nodes (3 distributors + 3 factories + 1 retailer)
- 60 pipeline slots (sum of all lead times: 5+3+8+10+9+11+12+0+1+2+0 = 61, minus zero-lead-time edges = 60)
- 2 features (time, goodwill)

---

## Action Space

$$a_t \in \mathbb{R}_{\geq 0}^{n_\text{reorder}}$$

Each action component represents the order quantity placed on reorder edge $e = (i, j)$ — how many units node $j$ requests from supplier $i$.

Actions are bounded:

$$0 \leq a_t^{(e)} \leq I_0^{\max} + C^{\max} \cdot T$$

where $I_0^{\max}$ is the maximum initial inventory across all nodes (400) and $C^{\max}$ is the maximum production capacity (90).

> **Note:** While the mathematical action space is continuous (to allow RL gradient flow), physical fulfillment can be forced to discrete units via `IntegerActionWrapper`. For deep RL training, normalize actions to $[-1, 1]$ with `gymnasium.wrappers.RescaleAction`.

---

## Transition Dynamics

At each time step $t$, events occur in the following sequence:

### Step 0: Order Placement

For each reorder edge $e = (i \rightarrow j)$, the filled order quantity depends on the supplier type:

**Raw material supplier** (unlimited):

$$R_t^{(e)} = a_t^{(e)}$$

**Distributor supplier** (inventory-constrained):

$$R_t^{(e)} = \min\bigl(a_t^{(e)},\; X_t^{(i)} - \text{allocated}_i\bigr)$$

where $\text{allocated}_i$ tracks how much of supplier $i$'s inventory has already been promised to other downstream nodes in this period (FCFS allocation).

**Factory supplier** (inventory + capacity constrained):

$$R_t^{(e)} = \min\bigl(a_t^{(e)},\; C_i - \text{used}_i,\; v_i \cdot (X_t^{(i)} - \text{allocated}_i)\bigr)$$

where:
- $C_i$ is the production capacity (max units the factory can produce this period)
- $\text{used}_i$ tracks capacity already consumed by other orders
- $v_i$ is the production yield — if $v_i < 1$, factory consumes $1/v_i$ units of raw material per unit produced

### Step 1: Deliveries and Inventory Update

For each managed node $j$ (factory or distributor):

$$X_{t+1}^{(j)} = X_t^{(j)} + \underbrace{\sum_i R_{t-L_{ij}}^{(i,j)}}_{\text{incoming deliveries}} - \underbrace{\frac{1}{v_j} \sum_k S_t^{(j,k)}}_{\text{outgoing shipments}}$$

- **Incoming deliveries:** Orders placed $L$ periods ago that arrive now
- **Outgoing shipments:** Units shipped downstream, yield-adjusted ($1/v_j$ raw material consumed per unit shipped)

Pipeline inventory updates:

$$Y_{t+1}^{(e)} = Y_t^{(e)} - R_{t-L_e}^{(e)} + R_t^{(e)}$$

### Step 2: Demand Realization

Customer demand is sampled at each retail edge (see [Demand Engine](demand_engine.md) for the full model).

Effective demand includes backlog from previous periods:

$$D_t^{\text{eff}} = D_t + U_{t-1} \quad \text{(when backlog mode is enabled, which is the default)}$$

Sales and unfulfilled quantities:

$$S_t = \min(D_t^{\text{eff}},\; X_{t+1}^{(j)}), \qquad U_t = D_t^{\text{eff}} - S_t$$

---

## Reward Function

The reward at each step is the **network-wide profit** — the sum over all managed nodes:

$$r_t = \sum_{j \in \text{main}} \alpha^t \bigl[\text{SR}_j - \text{PC}_j - \text{HC}_j - \text{OC}_j - \text{UP}_j\bigr]$$

Each component is computed per node $j$:

**Sales Revenue (SR):** Revenue from selling to downstream nodes.

$$\text{SR}_j = \sum_{k \in \text{succ}(j)} p_{jk} \cdot S_t^{(j,k)}$$

$p_{jk}$ is the **selling price** on the edge from $j$ to its downstream customer $k$.

**Purchasing Cost (PC):** Cost of ordering from upstream suppliers.

$$\text{PC}_j = \sum_{i \in \text{pred}(j)} p_{ij} \cdot R_t^{(i,j)}$$

$p_{ij}$ is the **purchasing price** — the same `p` attribute on the edge, but from $j$'s perspective it is a cost, not revenue.

**Holding Cost (HC):** Inventory carrying cost (on-hand + in-transit).

$$\text{HC}_j = h_j \cdot X_{t+1}^{(j)} + \sum_{i \in \text{pred}(j)} g_{ij} \cdot Y_{t+1}^{(i,j)}$$

$h_j$ is the node's holding cost; $g_{ij}$ is the edge's pipeline holding cost.

**Operating Cost (OC):** Factory variable production cost (only for factory nodes).

$$\text{OC}_j = \frac{o_j}{v_j} \cdot \sum_{k} S_t^{(j,k)}$$

$o_j$ is the operating cost per unit; dividing by yield $v_j$ accounts for the extra raw material consumed.

**Backlog Penalty (UP):** Penalty for unfulfilled demand (only for retailer nodes).

$$\text{UP}_j = \sum_{k \in \text{markets}} b_{jk} \cdot U_t^{(j,k)}$$

$b_{jk}$ is the backlog penalty per unfulfilled unit.

> **Note on `p` (price):** The edge attribute `p` appears in both SR and PC. From a downstream node's view, `p` is its purchasing cost; from the upstream node's view, `p` is its selling revenue. This creates a natural profit margin: a distributor buys at $p = 1.0$ from a factory and sells at $p = 1.5$ to a retailer, earning $0.50$ per unit margin.

---

## Default Network Values

### Base (Multi-Echelon Divergent) Network

**Nodes** — see [`_build_network_scenario()`](../gym_invmgmt/network_topology.py):

| Node | Type | $I_0$ | $h$ | $C$ | $o$ | $v$ |
|------|------|-------|------|------|------|------|
| 0 | Market | — | — | — | — | — |
| 1 | Retailer | 100 | 0.030 | — | — | — |
| 2 | Distributor | 110 | 0.020 | — | — | — |
| 3 | Distributor | 80 | 0.015 | — | — | — |
| 4 | Factory | 400 | 0.012 | 90 | 0.010 | 1.0 |
| 5 | Factory | 350 | 0.013 | 90 | 0.015 | 1.0 |
| 6 | Factory | 380 | 0.011 | 80 | 0.012 | 1.0 |
| 7 | Raw Material | — | — | — | — | — |
| 8 | Raw Material | — | — | — | — | — |

**Edges:**

| Edge | Type | $L$ | $p$ | $g$ | $b$ |
|------|------|-----|------|------|------|
| 1 → 0 | Retail | — | 2.000 | — | 0.10 |
| 2 → 1 | Reorder | 5 | 1.500 | 0.010 | — |
| 3 → 1 | Reorder | 3 | 1.600 | 0.015 | — |
| 4 → 2 | Reorder | 8 | 1.000 | 0.008 | — |
| 4 → 3 | Reorder | 10 | 0.800 | 0.006 | — |
| 5 → 2 | Reorder | 9 | 0.700 | 0.005 | — |
| 6 → 2 | Reorder | 11 | 0.750 | 0.007 | — |
| 6 → 3 | Reorder | 12 | 0.800 | 0.004 | — |
| 7 → 4 | Reorder | 0 | 0.150 | 0.000 | — |
| 7 → 5 | Reorder | 1 | 0.050 | 0.005 | — |
| 8 → 5 | Reorder | 2 | 0.070 | 0.002 | — |
| 8 → 6 | Reorder | 0 | 0.200 | 0.000 | — |

**Profit margin example:** Retailer 1 sells at $p = 2.00$ to the market and buys at $p = 1.50$ from distributor 2, earning $0.50$ gross margin per unit (before holding costs and penalties).

### Serial Network

| Node | Type | $I_0$ | $h$ | $C$ | $o$ | $v$ |
|------|------|-------|------|------|------|------|
| 0 | Market | — | — | — | — | — |
| 1 | Retailer | 100 | 0.030 | — | — | — |
| 2 | Distributor | 100 | 0.020 | — | — | — |
| 3 | Factory | 200 | 0.010 | 100 | 0.010 | 1.0 |
| 4 | Raw Material | — | — | — | — | — |

| Edge | Type | $L$ | $p$ | $g$ | $b$ |
|------|------|-----|------|------|------|
| 1 → 0 | Retail | — | 2.0 | — | 0.1 |
| 2 → 1 | Reorder | 4 | 1.5 | 0.010 | — |
| 3 → 2 | Reorder | 4 | 1.0 | 0.005 | — |
| 4 → 3 | Reorder | 0 | 0.5 | 0.000 | — |

---

## Episode Termination

- **Truncation**: Episode ends after $T$ steps (`num_periods`, default 30)
- **No early termination**: The environment never sets `terminated=True`

---

## References

- Hubbs, Perez, Sarwar, Li, Zarate, and Karunakaran. "OR-Gym: A Reinforcement Learning Library for Operations Research Problems." *arXiv:2008.06319*, 2020.
- Clark, A. J. and Scarf, H. "Optimal Policies for a Multi-Echelon Inventory Problem." *Management Science*, 6(4):475–490, 1960.
- Lee, H. L., Padmanabhan, V., and Whang, S. "The Bullwhip Effect in Supply Chains." *Sloan Management Review*, 38(3):93, 1997.
