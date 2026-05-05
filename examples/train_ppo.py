"""
train_ppo.py — Train a PPO agent on the inventory management environment.

Requires: pip install stable-baselines3

Optional:
    pip install wandb          — Weights & Biases experiment tracking

Pipeline:
  CoreEnv → RescaleAction([-1,1]) → DummyVecEnv → VecNormalize → PPO

Usage:
    python examples/train_ppo.py                          # TensorBoard + CSV (default)
    python examples/train_ppo.py --wandb                  # + Weights & Biases
    python examples/train_ppo.py --wandb --run-name exp1  # custom W&B run name

Output:
    models/ppo_invmgmt.zip          — trained model
    models/vec_normalize.pkl         — observation normalization stats
    models/tb_logs/                  — TensorBoard event files
    models/csv_logs/                 — CSV training metrics
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RescaleAction

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from gym_invmgmt import CoreEnv
from gym_invmgmt.utils import run_episode, compute_kpis


# ── Configuration ─────────────────────────────────────────────────────

ENV_CONFIG = dict(
    scenario='network',
    demand_config={
        'type': 'stationary',
        'base_mu': 20,
        'use_goodwill': False,
    },
    num_periods=30,
    backlog=True,
)

TOTAL_TIMESTEPS = 200_000       # increase for better results
MODEL_DIR       = 'models'
MODEL_PATH      = f'{MODEL_DIR}/ppo_invmgmt'
STATS_PATH      = f'{MODEL_DIR}/vec_normalize.pkl'


# ── Environment factory ──────────────────────────────────────────────

def make_env(seed: int = 0):
    """Create a wrapped training environment."""
    def _init():
        env = CoreEnv(**ENV_CONFIG)
        # Rescale actions from [-1, 1] → [0, max_order]
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ── Simple logging callback ──────────────────────────────────────────

class ProgressCallback(BaseCallback):
    """Print mean episode reward every N timesteps."""

    def __init__(self, log_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            infos = self.locals.get('infos', [{}])
            ep_rewards = [
                info['episode']['r']
                for info in infos if 'episode' in info
            ]
            if ep_rewards:
                print(f"  [{self.num_timesteps:>7,} steps]  "
                      f"mean_reward = {np.mean(ep_rewards):,.0f}")
        return True


# ── Logger setup ─────────────────────────────────────────────────────

def setup_logger(model, use_wandb: bool = False, run_name: str | None = None):
    """Configure SB3 logging backends.

    Always enables: TensorBoard + CSV + stdout.
    Optionally enables: Weights & Biases (--wandb flag).
    """
    log_dir = f'{MODEL_DIR}/csv_logs'
    os.makedirs(log_dir, exist_ok=True)

    # Base formats: TensorBoard is handled via tensorboard_log param;
    # here we add CSV and stdout
    output_formats = ["csv", "stdout", "tensorboard"]

    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="gym-invmgmt",
                name=run_name or "ppo_invmgmt",
                config={
                    "algorithm": "PPO",
                    "total_timesteps": TOTAL_TIMESTEPS,
                    "env_config": ENV_CONFIG,
                    "policy": "MlpPolicy",
                    "net_arch": [256, 256],
                    "learning_rate": 3e-4,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                },
                sync_tensorboard=True,  # auto-sync TB logs to W&B
            )
            print("OK Weights & Biases logging enabled")
        except ImportError:
            print("WARNING wandb not installed — skipping W&B logging")
            print("  Install with: pip install wandb")

    new_logger = configure(log_dir, output_formats)
    model.set_logger(new_logger)


# ── Train ─────────────────────────────────────────────────────────────

def train(use_wandb: bool = False, run_name: str | None = None):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Vectorized env with observation & reward normalization
    vec_env = DummyVecEnv([make_env(seed=0)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )

    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=f'{MODEL_DIR}/tb_logs',
        verbose=1,
    )

    # Set up logging backends (CSV, TensorBoard, optionally W&B)
    setup_logger(model, use_wandb=use_wandb, run_name=run_name)

    print("=" * 60)
    print(f"Training PPO for {TOTAL_TIMESTEPS:,} timesteps")
    print(f"  Scenario:  {ENV_CONFIG['scenario']}")
    print(f"  Demand:    {ENV_CONFIG['demand_config']['type']}")
    print(f"  Obs shape: {vec_env.observation_space.shape}")
    print(f"  Act shape: {vec_env.action_space.shape}")
    print(f"  Logging:   TensorBoard + CSV" + (" + W&B" if use_wandb else ""))
    print("=" * 60)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=ProgressCallback())

    # Save model + normalization stats
    model.save(MODEL_PATH)
    with open(STATS_PATH, 'wb') as f:
        pickle.dump(vec_env, f)

    print(f"\nOK Model saved → {MODEL_PATH}.zip")
    print(f"OK Stats saved → {STATS_PATH}")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
            print("OK W&B run finished")
        except Exception:
            pass

    vec_env.close()
    return model


# ── Evaluate ──────────────────────────────────────────────────────────

def evaluate():
    """Quick evaluation of the trained model vs random policy."""
    print("\n" + "=" * 60)
    print("Evaluation (5 episodes each)")
    print("=" * 60)

    # Random baseline
    random_profits = []
    for seed in range(5):
        env = CoreEnv(**ENV_CONFIG)
        result = run_episode(env, seed=seed)
        random_profits.append(result['total_reward'])

    print(f"\n  Random policy:  {np.mean(random_profits):>10,.0f} ± {np.std(random_profits):,.0f}")

    # Trained PPO (simplified eval without VecNormalize for brevity)
    if os.path.exists(f"{MODEL_PATH}.zip"):
        model = PPO.load(MODEL_PATH)
        ppo_profits = []
        for seed in range(5):
            env = CoreEnv(**ENV_CONFIG)
            env = RescaleAction(env, min_action=-1.0, max_action=1.0)
            obs, _ = env.reset(seed=seed)
            done, total = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = env.step(action)
                total += reward
                done = term or trunc
            ppo_profits.append(total)
        print(f"  Trained PPO:    {np.mean(ppo_profits):>10,.0f} ± {np.std(ppo_profits):,.0f}")
    else:
        print("  (no trained model found — run train() first)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on gym-invmgmt")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--run-name", type=str, default=None,
                        help="W&B run name (default: ppo_invmgmt)")
    args = parser.parse_args()

    train(use_wandb=args.wandb, run_name=args.run_name)
    evaluate()
