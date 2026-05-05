# Demand Engine

The `DemandEngine` provides a flexible, composable demand model that supports both stationary and non-stationary demand patterns with optional endogenous feedback.

---

## Architecture

```
DemandEngine
├── Base Demand (μ₀) or External Series
├── Composable Effects (multiplicative)
│   ├── Trend     — Linear growth over time
│   ├── Seasonal  — Sinusoidal oscillations
│   └── Shock     — Step-function jump at time t*
├── External Demand Series — Real-world data injection (M5, Favorita, etc.)
└── Endogenous Goodwill — Service-dependent feedback loop
```

---

## Base Demand Model

At each time step $t$, customer demand is drawn from a **Poisson distribution**:

$$
D_t \sim \text{Poisson}(\mu_t)
$$

where $\mu_t$ is the effective mean demand, computed by composing effects on top of the base mean $\mu_0$:

$$
\mu_t = \mu_0 \cdot \prod_{e \in \text{active effects}} f_e(t) \cdot s_t
$$

### Noise Scaling

After sampling, the deviation from the mean is scaled by a configurable factor $\sigma$:

$$
D_t^{\text{final}} = \max\left(0,\; \mu_t + \sigma \cdot (D_t^{\text{raw}} - \mu_t)\right)
$$

| Value | Behavior |
|---|---|
| 0.0 | Deterministic (demand equals μ exactly) |
| 1.0 | Standard Poisson variance (default) |
| > 1.0 | Amplified variance |

> **Note:** When $\sigma \neq 1.0$, the resulting demand distribution is no longer strictly Poisson. Values $\sigma < 1.0$ produce under-dispersed demand (tighter around the mean), while $\sigma > 1.0$ produces over-dispersed demand (similar in effect to a Negative Binomial).

---

## Composable Non-Stationary Effects

Effects are applied **multiplicatively** and can be freely combined. Specifying `effects=['trend', 'seasonal']` applies both simultaneously.

### Trend

Linear growth over time:

$$
f_{\text{trend}}(t) = 1 + \beta \cdot t
$$

| Parameter | Default | Description |
|---|---|---|
| `trend_slope` (β) | 0.05 | Growth rate per period |

**Example**: With $\mu_0 = 20$ and $\beta = 0.05$, demand grows from 20 at $t=0$ to $20 \times (1 + 0.05 \times 29) = 49$ at $t=29$.

### Seasonal

Sinusoidal oscillation around the baseline:

$$
f_{\text{seasonal}}(t) = 1 + A \cdot \sin(\omega \cdot t)
$$

| Parameter | Default | Description |
|---|---|---|
| `seasonal_amp` (A) | 0.5 | Amplitude (0.5 = ±50% fluctuation) |
| `seasonal_freq` (ω) | 2π/30 | Angular frequency (~monthly cycle) |

**Example**: Demand oscillates between $\mu_0 \cdot 0.5$ and $\mu_0 \cdot 1.5$ over a 30-period cycle.

### Shock

A permanent step-function increase at time $t^*$:

$$
f_{\text{shock}}(t) = \begin{cases}
1 & \text{if } t < t^* \\
m & \text{if } t \geq t^*
\end{cases}
$$

| Parameter | Default | Description |
|---|---|---|
| `shock_time` (t*) | 15 | Period when shock activates |
| `shock_mag` (m) | 2.0 | Multiplicative factor (2.0 = demand doubles) |

### Combined Examples

```python
# Trend + seasonal + shock (all three simultaneously)
demand_config = {
    'effects': ['trend', 'seasonal', 'shock'],
    'base_mu': 20,
    'trend_slope': 0.03,
    'seasonal_amp': 0.4,
    'shock_time': 20,
    'shock_mag': 1.5,
}
# Example evaluation at t=25 (after shock_time=20, so all three effects are active):
# μ = 20 × (1 + 0.03×25) × (1 + 0.4·sin(2π/30 × 25)) × 1.5 ≈ 34.3
```

---

## Endogenous Goodwill Dynamics

When `use_goodwill=True`, the environment simulates a **service-level feedback loop** where unfulfilled demand erodes customer willingness to buy, and consistent fulfillment gradually restores it.

### Update Rule

The goodwill sentiment $s_t$ is updated after each period:

$$
s_{t+1} = \begin{cases}
\min\left(s_{\max},\; s_t \cdot \gamma_{\text{grow}}\right) & \text{if } U_t = 0 \text{ (fully satisfied)} \\
\max\left(s_{\min},\; s_t \cdot \gamma_{\text{decay}}\right) & \text{if } U_t > 0 \text{ (stockout occurred)}
\end{cases}
$$

| Parameter | Default | Description |
|---|---|---|
| `gw_growth` (γ_grow) | 1.01 | Growth factor per satisfied period (+1%) |
| `gw_decay` (γ_decay) | 0.90 | Decay factor per stockout period (-10%) |
| s_max | 2.0 | Sentiment ceiling (max 2× demand) |
| s_min | 0.2 | Sentiment floor (min 0.2× demand) |
| s₀ | 1.0 | Initial sentiment (neutral) |

### Dynamics

The asymmetry is deliberate:
- **Decay is fast** (10% per stockout) — customers leave quickly after poor service
- **Recovery is slow** (1% per period) — trust rebuilds gradually

This creates a **ratchet effect**: a few consecutive stockouts at $t=5{-}8$ can reduce demand to $0.66\mu_0$, which then takes ~40 periods of perfect service to fully recover.

### Interaction with Non-Stationary Effects

Goodwill is applied **after** all other effects:

$$
\mu_t = \underbrace{\mu_0 \cdot \prod_e f_e(t)}_{\text{non-stationary base}} \cdot \underbrace{s_t}_{\text{goodwill}}
$$

This means goodwill amplifies or dampens already-modified demand. For example, a demand shock with eroded goodwill: $\mu_t = 20 \times 2.0 \times 0.5 = 20$ — the shock is fully offset by poor reputation.

---

## External Demand Series

The `external_series` parameter allows injecting **real-world demand data** directly into the DemandEngine. When provided, `get_current_mu(t)` uses `external_series[t]` as the base mean instead of the constant `base_mu`.

This enables benchmarking with datasets like M5 (Walmart), Favorita (grocery), Rossmann (retail), or any custom demand data.

### How It Works

$$
\mu_t = \text{externalSeries}[t] \cdot \prod_e f_e(t) \cdot s_t
$$

Where `external_series[t]` replaces the constant `base_mu` (μ₀) as the base demand at each time step.

Composable effects and goodwill are still applied on top of the external series. If the time index exceeds the series length, the last value is used (clamping).

### Usage

```python
import numpy as np

# Real demand data (e.g., from CSV)
real_demand = np.array([15, 22, 18, 30, 12, 25, 19, 28, ...])  # one value per period

env = gym.make("GymInvMgmt/MultiEchelon-v0",
    demand_config={
        'type': 'stationary',
        'base_mu': float(np.mean(real_demand)),  # default value for agents that need it
        'external_series': real_demand,
    },
    num_periods=len(real_demand),
)
```

The `external_series` parameter is fully backward-compatible:
- If not provided (or `None`), the DemandEngine behaves exactly as before using `base_mu`.
- Existing code and configurations continue to work without any changes.

> **See also:** [External Datasets Guide](external_datasets.md) — recommended real-world datasets (M5, Favorita, Rossmann) with download links and usage examples.

---

## Configuration Reference

```python
demand_config = {
    # --- Base ---
    'base_mu': 20,              # Base Poisson mean
    'noise_scale': 1.0,         # Variance scaling (0=deterministic)

    # --- External Demand Series ---
    'external_series': None,    # np.ndarray of per-period demand (replaces base_mu)

    # --- Non-Stationary Effects ---
    # Option A: Single effect shorthand
    'type': 'stationary',       # 'stationary' | 'trend' | 'seasonal' | 'shock'

    # Option B: Composable effects (preferred)
    'effects': ['trend', 'seasonal'],  # Any combination

    # Effect parameters
    'trend_slope': 0.05,        # β: growth rate
    'seasonal_amp': 0.5,        # A: oscillation amplitude
    'seasonal_freq': 0.2094,    # ω: angular frequency (2π/30)
    'shock_time': 15,           # t*: shock activation period
    'shock_mag': 2.0,           # m: shock multiplier

    # --- Goodwill ---
    'use_goodwill': False,      # Enable feedback loop
    'gw_growth': 1.01,          # Growth rate per satisfied period
    'gw_decay': 0.90,           # Decay rate per stockout
}
```
