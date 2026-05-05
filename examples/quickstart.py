"""
quickstart.py — Minimal example of running the gym-invmgmt environment.

Usage:
    python examples/quickstart.py
"""
import gymnasium as gym
import gym_invmgmt

# ── Create environment ────────────────────────────────────────────────
env = gym.make("GymInvMgmt/MultiEchelon-v0")
obs, info = env.reset(seed=42)

print(f"Environment: {env.spec.id}")
print(f"  Observation shape: {env.observation_space.shape}")
print(f"  Action shape:      {env.action_space.shape}")
print(f"  Num periods:       {env.unwrapped.num_periods}")
print()

# ── Run one episode with random actions ───────────────────────────────
total_reward = 0.0
done = False
step = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    step += 1

print(f"Episode finished in {step} steps")
print(f"  Total reward:    {total_reward:,.2f}")
print(f"  Total inventory: {info['total_inventory']:.0f}")
print(f"  Total backlog:   {info['total_backlog']:.0f}")

# ── Compute KPIs ──────────────────────────────────────────────────────
from gym_invmgmt.utils import compute_kpis

kpis = compute_kpis(env)
print(f"\nKPIs:")
print(f"  Fill rate:       {kpis['fill_rate']:.2%}")
print(f"  Inventory turns: {kpis['inventory_turns']:.2f}")
print(f"  Avg inventory:   {kpis['avg_inventory']:.0f}")
