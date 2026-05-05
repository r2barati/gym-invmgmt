"""
Domain randomization wrapper for generalist inventory-control training.

The canonical generalist training scripts historically used an inline wrapper.
This public wrapper mirrors that behavior while exposing explicit knobs for
effect probabilities, goodwill/backlog options, noise scale, and demand-shape
parameters. It mutates the underlying ``DemandEngine`` directly so parameter
names cannot silently drift from ``DemandEngine``'s implementation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import gymnasium as gym
import numpy as np


class DomainRandomizationWrapper(gym.Wrapper):
    """Randomize demand, goodwill, backlog, and noise on each reset.

    By default, each demand effect in ``("trend", "seasonal", "shock")`` is
    independently included with probability ``effect_prob``. This matches the
    composable-effect training protocol used by the generalist agents. Passing
    ``demand_types`` is supported for backward compatibility; values other than
    ``"stationary"`` are interpreted as the candidate effect set.
    """

    DEFAULT_EFFECT_OPTIONS = ("trend", "seasonal", "shock")
    DEFAULT_GOODWILL_OPTIONS = (False, True)
    DEFAULT_BACKLOG_OPTIONS = (False, True)

    def __init__(
        self,
        env: gym.Env,
        demand_types: Optional[Sequence[str]] = None,
        goodwill_options: Optional[Sequence[bool]] = None,
        backlog_options: Optional[Sequence[bool]] = None,
        effect_options: Optional[Sequence[str]] = None,
        effect_prob: float = 0.5,
        noise_scale_range: tuple[float, float] = (0.5, 2.0),
        external_series_prob: float = 0.15,
        external_series_length: Optional[int] = None,
        base_mu: float = 20.0,
        trend_slope: Optional[float] = None,
        seasonal_amp: Optional[float] = None,
        seasonal_freq: Optional[float] = None,
        shock_time: Optional[int] = None,
        shock_mag: Optional[float] = None,
        randomize_backlog: bool = True,
    ) -> None:
        super().__init__(env)

        if effect_options is None and demand_types is not None:
            effect_options = [d for d in demand_types if d != "stationary"]

        if effect_options is None:
            effect_options = self.DEFAULT_EFFECT_OPTIONS

        self.effect_options = tuple(effect_options)
        self.goodwill_options = tuple(goodwill_options or self.DEFAULT_GOODWILL_OPTIONS)
        self.backlog_options = tuple(backlog_options or self.DEFAULT_BACKLOG_OPTIONS)
        self.effect_prob = float(effect_prob)
        self.noise_scale_range = noise_scale_range
        self.external_series_prob = float(external_series_prob)
        self.external_series_length = external_series_length
        self.base_mu = float(base_mu)
        self.trend_slope = trend_slope
        self.seasonal_amp = seasonal_amp
        self.seasonal_freq = seasonal_freq
        self.shock_time = shock_time
        self.shock_mag = shock_mag
        self.randomize_backlog = randomize_backlog

    def _rng_for_reset(self, kwargs):
        base_env = self.env.unwrapped
        seed = kwargs.get("seed")
        if seed is not None:
            return np.random.default_rng(seed)
        return getattr(base_env, "np_random", np.random.default_rng())

    def _sample_external_series(self, rng: np.random.Generator) -> Optional[np.ndarray]:
        if self.external_series_prob <= 0 or rng.random() >= self.external_series_prob:
            return None

        base_env = self.env.unwrapped
        n = int(self.external_series_length or getattr(base_env, "num_periods", 30))
        t = np.arange(n)
        base = self.base_mu + 0.02 * t * rng.uniform(0, 2)
        seasonal = 1 + 0.3 * np.sin(2 * np.pi * t / 7)
        demand = rng.poisson(np.maximum(base * seasonal, 0.5))
        spikes = rng.random(n) < 0.05
        demand[spikes] = (demand[spikes] * 3).astype(int)
        return demand.astype(float)

    def _sample_effects(self, rng: np.random.Generator) -> list[str]:
        if self.effect_prob <= 0:
            return []
        if self.effect_prob >= 1:
            return list(self.effect_options)
        return [effect for effect in self.effect_options if rng.random() < self.effect_prob]

    def reset(self, **kwargs):
        rng = self._rng_for_reset(kwargs)
        base_env = self.env.unwrapped
        demand_engine = base_env.demand_engine

        effects = self._sample_effects(rng)
        use_goodwill = bool(rng.choice(self.goodwill_options))
        noise_low, noise_high = self.noise_scale_range
        noise_scale = float(rng.uniform(noise_low, noise_high))
        external_series = self._sample_external_series(rng)

        demand_engine.effects = effects
        demand_engine.type = "combined_chaos" if effects else "stationary"
        demand_engine.base_mu = self.base_mu
        demand_engine.use_goodwill = use_goodwill
        demand_engine.noise_scale = noise_scale
        demand_engine.sentiment = 1.0
        demand_engine.external_series = external_series

        if self.trend_slope is not None:
            demand_engine.trend_slope = float(self.trend_slope)
        if self.seasonal_amp is not None:
            demand_engine.seasonal_amp = float(self.seasonal_amp)
        if self.seasonal_freq is not None:
            demand_engine.seasonal_freq = float(self.seasonal_freq)
        if self.shock_time is not None:
            demand_engine.shock_time = int(self.shock_time)
        if self.shock_mag is not None:
            demand_engine.shock_mag = float(self.shock_mag)

        if self.randomize_backlog:
            base_env.backlog = bool(rng.choice(self.backlog_options))

        return self.env.reset(**kwargs)
