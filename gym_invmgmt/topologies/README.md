# Network Topology Definitions

YAML configuration files for defining custom supply-chain network topologies.

## Usage

```python
from gym_invmgmt import make_custom_env

env = make_custom_env('gym_invmgmt/topologies/diamond.yaml', num_periods=30)
obs, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## Available Topologies

| File | Topology | Nodes | Actions | Description |
|------|---------|:-----:|:-------:|-------------|
| [`serial.yaml`](serial.yaml) | RM → F → D → R → M | 5 | 3 | Equivalent to `scenario='serial'` |
| [`divergent.yaml`](divergent.yaml) | 2 RM → 3 F → 2 D → R → M | 9 | 11 | Equivalent to `scenario='network'` |
| [`diamond.yaml`](diamond.yaml) | RM → 2 F → R → M | 5 | 4 | Diamond with parallel factories |
| [`assembly.yaml`](assembly.yaml) | 2 RM → 2 Comp → Assembly → R → M | 7 | 5 | Convergent (assembly) supply chain |
| [`distribution_tree.yaml`](distribution_tree.yaml) | RM → F → Hub → 3 R → 3 M | 9 | 5 | Distribution tree with 3 retailers |
| [`w_network.yaml`](w_network.yaml) | 2 RM → 2 F → DC → 2 R → 2 M | 9 | 6 | Shared DC with dual sourcing |

## Creating Your Own

Copy any file above as a starting point. Node roles are auto-detected from graph structure:

- **Market**: no successors (demand sink)
- **Raw Material**: no predecessors (unlimited supply)
- **Factory**: has `C` (capacity) attribute
- **Distributor**: has `I0` but not `C`
- **Retailer**: directly supplies a market

See the [full YAML schema reference](../../docs/network_topologies.md#defining-custom-topologies) for all supported parameters and validation rules.
