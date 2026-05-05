# External Datasets Guide

Use real-world demand data with `gym-invmgmt` to benchmark agents against realistic patterns. The `DemandEngine.external_series` parameter accepts any NumPy array of per-period demand values.

---

## Recommended Datasets

### Retail & Grocery

| Dataset | Scale | Source | Key Features |
|---------|-------|--------|-------------|
| **M5 Forecasting** | 30K items × 10 stores × 1941 days | [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) | Walmart daily sales, prices, events, SNAP |
| **Favorita Grocery** | 4K items × 54 stores × 1684 days | [Kaggle](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data) | Store metadata (city, type, cluster), oil prices, holidays |
| **Rossmann Stores** | 1115 stores × 942 days | [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data) | Store type, assortment, competition distance, promos |
| **UCI Online Retail** | 541K transactions | [UCI ML](https://archive.ics.uci.edu/ml/datasets/Online+Retail) | Invoice-level data, customer IDs, unit prices |

### Time Series Benchmarks

| Dataset | Scale | Source | Key Features |
|---------|-------|--------|-------------|
| **Airline Passengers** | 144 months | [GitHub](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv) | Classic trend + 12-month seasonality |
| **Shampoo Sales** | 36 months | [GitHub](https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv) | Strong upward trend |
| **Daily Female Births** | 365 days | [GitHub](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv) | Stationary demand (Poisson-like) |
| **Australian Tourism** | Quarterly, multi-region | [Rob Hyndman](https://robjhyndman.com/data/ausquest.csv) | Multi-region hierarchical demand |

### Supply Chain Specific

| Dataset | Scale | Source | Key Features |
|---------|-------|--------|-------------|
| **Store Sales (Kaggle)** | 1782 days × 54 stores × 33 families | [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) | Promotions, oil prices, holidays |
| **Instacart Basket** | 3.4M orders × 50K products | [Kaggle](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data) | Order frequency, reorder patterns |

---

## Quick Start

### Download & Run (3 lines)

```bash
# Download airline passengers (small, public, no Kaggle account needed)
curl -sL "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv" \
  -o data/airline_passengers.csv

# Parse + benchmark (auto-scales to env range, detects trend & seasonality)
PYTHONPATH=. python scripts/eval/benchmark_real_dataset.py \
  --data data/airline_passengers.csv \
  --format csv \
  --demand-column Passengers \
  --tier 2
```

### M5 Dataset

```bash
# 1. Download from Kaggle (requires kaggle CLI)
kaggle competitions download -c m5-forecasting-accuracy -p data/m5/
unzip data/m5/*.zip -d data/m5/

# 2. Benchmark a specific item at a specific store
PYTHONPATH=. python scripts/eval/benchmark_real_dataset.py \
  --data data/m5/ \
  --format m5 \
  --item FOODS_3_090 \
  --store CA_1 \
  --tier 2

# 3. Benchmark an entire department across all stores
PYTHONPATH=. python scripts/eval/benchmark_real_dataset.py \
  --data data/m5/ \
  --format m5 \
  --dept FOODS_3 \
  --tier 2
```

### Favorita Dataset

```bash
kaggle competitions download -c favorita-grocery-sales-forecasting -p data/favorita/
unzip data/favorita/*.zip -d data/favorita/

PYTHONPATH=. python scripts/eval/benchmark_real_dataset.py \
  --data data/favorita/ \
  --format favorita \
  --family GROCERY_I \
  --tier 2
```

### Generic CSV

```bash
# Any CSV with a demand column
PYTHONPATH=. python scripts/eval/benchmark_real_dataset.py \
  --data your_sales_data.csv \
  --format csv \
  --demand-column units_sold \
  --price-column unit_price \
  --tier 2
```

---

## What Gets Extracted

The parser automatically extracts everything possible from each dataset:

### Tier 1 — Direct Extraction (always applied)

| What | Extracted From | Maps To |
|------|---------------|---------|
| Demand series | Sales/units columns | `external_series` |
| Selling price | Price columns (M5: `sell_prices.csv`) | Edge `p` parameter |
| Store count | Store IDs | Number of retail nodes |
| Store grouping | State/region metadata | Distributor regions |

### Tier 2 — Statistical Inference (opt-in via `--tier 2`)

| What | Method | Maps To |
|------|--------|---------|
| Demand distribution | Var/Mean dispersion ratio | `noise_scale` |
| Trend | Linear regression | `trend_slope` |
| Seasonality | FFT peak detection | `seasonal_amp`, `seasonal_freq` |
| Demand shocks | Z-score changepoint | `shock_time`, `shock_mag` |
| Initial inventory | μ×L + 2σ√L safety stock | `I0` estimate |

### Auto-Scaling

Datasets with high demand (μ > 40) are automatically scaled to the environment's calibrated range (μ ≈ 20) while **preserving demand shape** (relative variation, trends, seasonality). Disable with `--no-scale`.

---

## Using in Python (without CLI)

```python
from scripts.eval.dataset_to_env import parse_dataset
from src.envs.builder import make_supply_chain_env

# Parse dataset
config = parse_dataset("data/m5/", format="m5", tier=2,
                       item_id="FOODS_3_090", store_id="CA_1")

# Show what was extracted
config.print_summary()

# Get env kwargs
env_kwargs = config.to_env_kwargs(scenario='base', num_periods=30)

# Create environment with real-world demand
env = make_supply_chain_env(agent_type='or', **env_kwargs)
obs, info = env.reset(seed=42)
```

Or directly with `DemandEngine.external_series`:

```python
import numpy as np
import gymnasium as gym
import gym_invmgmt

demand = np.loadtxt("my_sales.csv", delimiter=",", skiprows=1, usecols=1)

env = gym.make("GymInvMgmt/MultiEchelon-v0",
    demand_config={
        'type': 'stationary',
        'base_mu': float(np.mean(demand)),
        'external_series': demand,
    },
    num_periods=len(demand),
)
```

---

## Verified Datasets

These datasets have been tested end-to-end with the parser and benchmark:

| Dataset | Rows | Inferred Effects | Inference Correct? |
|---------|------|-----------------|-------------------|
| Airline Passengers | 144 | Trend (+137%), 12-month seasonality | OK Textbook match |
| Shampoo Sales | 36 | Trend (+139%), Shock at t=30 | OK Known pattern |
| Daily Births | 365 | None (stationary) | OK Correct — births are stable |
| M5 Synthetic | 200 | Overdispersed (Var/Mean=3.48) | OK Expected for retail |
