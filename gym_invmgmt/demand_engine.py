import numpy as np
from scipy.stats import norm as _norm
from scipy.stats import poisson as _poisson
from scipy.stats import uniform as _uniform

VALID_EFFECTS = {"trend", "seasonal", "shock"}
TYPE_EFFECT_ALIASES = {
    "stationary": [],
    "trend": ["trend"],
    "seasonal": ["seasonal"],
    "shock": ["shock"],
    "trend+seasonal": ["trend", "seasonal"],
    "trend_seasonal": ["trend", "seasonal"],
    "trend-seasonal": ["trend", "seasonal"],
    "seasonal+shock": ["seasonal", "shock"],
    "seasonal_shock": ["seasonal", "shock"],
    "seasonal-shock": ["seasonal", "shock"],
    "trend+shock": ["trend", "shock"],
    "trend_shock": ["trend", "shock"],
    "trend-shock": ["trend", "shock"],
    "trend+seasonal+shock": ["trend", "seasonal", "shock"],
    "trend_seasonal_shock": ["trend", "seasonal", "shock"],
    "trend-seasonal-shock": ["trend", "seasonal", "shock"],
    "combined_chaos": ["trend", "seasonal", "shock"],
}


def _normalize_effects(config):
    """Return validated demand effects from compatibility type aliases or effects=[...]."""
    if "effects" in config:
        effects = list(config["effects"])
    else:
        demand_type = config.get("type", "stationary")
        if demand_type not in TYPE_EFFECT_ALIASES:
            raise ValueError(
                f"Unknown demand type {demand_type!r}. "
                f"Supported types: {sorted(TYPE_EFFECT_ALIASES)}; "
                "or pass effects=['trend', 'seasonal', 'shock']."
            )
        effects = TYPE_EFFECT_ALIASES[demand_type]

    unknown = [effect for effect in effects if effect not in VALID_EFFECTS]
    if unknown:
        raise ValueError(f"Unknown demand effect(s): {unknown!r}. " f"Supported effects: {sorted(VALID_EFFECTS)}.")
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(effects))


class DemandEngine:
    """
    Handles dynamic demand generation including:
    1. Composable Non-Stationary Effects (Trend, Seasonal, Shock — can be combined)
    2. Endogenous Goodwill (Service-dependent demand)
    3. Noise Scaling (controls demand variance independently from the mean)
    """

    def __init__(self, config=None):
        self.config = config if config else {}
        self.base_mu = self.config.get("base_mu", 20)

        # --- External Demand Series (e.g. M5 competition data) ---
        # When provided, get_current_mu(t) uses external_series[t] as the
        # base mu instead of self.base_mu.  Composable effects and goodwill
        # are still applied on top.
        self.external_series = self.config.get("external_series", None)
        if self.external_series is not None:
            import warnings as _warnings

            import numpy as _np

            # Convert to numpy array with helpful error
            try:
                self.external_series = _np.asarray(self.external_series, dtype=float)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"external_series must be convertible to a numeric array. "
                    f"Got type {type(self.config['external_series']).__name__}: {e}"
                ) from e

            # Must be 1-D
            if self.external_series.ndim != 1:
                raise ValueError(f"external_series must be a 1-D array, " f"got shape {self.external_series.shape}")

            # Must not be empty
            if len(self.external_series) == 0:
                raise ValueError("external_series must not be empty.")

            # Check for NaN / Inf
            if _np.any(~_np.isfinite(self.external_series)):
                bad_count = int(_np.sum(~_np.isfinite(self.external_series)))
                raise ValueError(
                    f"external_series contains {bad_count} non-finite values "
                    f"(NaN or Inf). Please clean the data before passing it."
                )

            # Warn if suspiciously short
            if len(self.external_series) < 10:
                _warnings.warn(
                    f"external_series has only {len(self.external_series)} values. "
                    f"Values will be clamped for time steps beyond this length.",
                    stacklevel=2,
                )

        # --- Composable Effects ---
        # Accepts either effects=['trend', 'seasonal', 'shock'] (any combination)
        # or type aliases such as type='trend+seasonal' / 'combined_chaos'.
        self.effects = _normalize_effects(self.config)

        # self.type mirrors the requested type label for convenience.
        self.type = self.config.get("type", "stationary")

        self.use_goodwill = self.config.get("use_goodwill", False)

        # --- Noise Scaling ---
        # 1.0 = default (unchanged), 0.0 = deterministic, >1.0 = amplified variance
        self.noise_scale = self.config.get("noise_scale", 1.0)

        # --- Effect Parameters (Defaults match literature/Julia code) ---
        self.trend_slope = self.config.get("trend_slope", 0.05)
        self.seasonal_amp = self.config.get("seasonal_amp", 0.5)  # 50% fluctuation
        self.seasonal_freq = self.config.get("seasonal_freq", 2 * np.pi / 30)  # ~Monthly cycle
        self.shock_time = self.config.get("shock_time", 15)
        self.shock_mag = self.config.get("shock_mag", 2.0)

        # --- Goodwill Parameters ---
        self.sentiment = 1.0
        self.gw_growth = self.config.get("gw_growth", 1.01)  # 1% growth if satisfied
        self.gw_decay = self.config.get("gw_decay", 0.90)  # 10% drop if lost sales
        self.gw_cap = float(self.config.get("gw_cap", 2.0))  # Max demand multiplier
        self.gw_floor = float(self.config.get("gw_floor", 0.2))  # Min demand multiplier
        if self.gw_floor <= 0 or self.gw_cap <= 0 or self.gw_floor > self.gw_cap:
            raise ValueError(
                "Goodwill bounds must satisfy 0 < gw_floor <= gw_cap. "
                f"Got gw_floor={self.gw_floor}, gw_cap={self.gw_cap}."
            )

    def reset(self, np_random=None):
        self.sentiment = 1.0
        if np_random is not None:
            self.np_random = np_random
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()
        return self.get_observation(0)

    def sample(self, t, dist, dist_params=None):
        """
        Sample demand for time step t using the engine's own PRNG.

        Args:
            t: Current time step.
            dist: Distribution class (e.g. scipy.stats.poisson).
            dist_params: Distribution parameters from edge config (e.g. {'mu': 20}).
                         When provided, the engine's current mu (from composable
                         effects) replaces the location parameter ('mu' or 'loc').
                         When None, parameters are auto-dispatched based on dist type.

        Returns:
            Sampled demand value (float, non-negative).
        """
        mu = self.get_current_mu(t)

        if hasattr(dist, "rvs"):
            if dist_params is not None:
                # Use edge-provided params, replacing the location with engine's mu.
                params = dict(dist_params)
                if "mu" in params:
                    params["mu"] = mu
                elif "loc" in params:
                    params["loc"] = mu
                elif isinstance(dist, type(_poisson)):
                    params["mu"] = mu
                else:
                    params["loc"] = mu
            else:
                # Auto-dispatch based on distribution type.
                # - poisson uses 'mu'
                # - norm uses 'loc' (with scale=sqrt(mu) for Poisson-like variance)
                # - uniform uses 'loc' and 'scale' (centered on mu)
                if isinstance(dist, type(_poisson)):
                    params = {"mu": mu}
                elif isinstance(dist, type(_norm)):
                    params = {"loc": mu, "scale": np.sqrt(max(mu, 0.1))}
                elif isinstance(dist, type(_uniform)):
                    # Uniform centered on mu with width 2*mu (range [0, 2*mu])
                    params = {"loc": 0, "scale": 2 * mu}
                else:
                    # Fallback: try 'mu', then 'loc'
                    try:
                        return max(0.0, mu + self.noise_scale * (dist.rvs(mu=mu, random_state=self.np_random) - mu))
                    except TypeError:
                        params = {"loc": mu}
            raw = dist.rvs(**params, random_state=self.np_random)
        elif callable(dist):
            # Custom callable: try passing mu, fall back to zero-arg
            try:
                raw = dist(mu)
            except TypeError:
                raw = dist()

        return max(0.0, mu + self.noise_scale * (raw - mu))

    def update_goodwill(self, unfulfilled_qty):
        """Updates reputation based on previous step's lost sales."""
        if not self.use_goodwill:
            return

        if unfulfilled_qty > 0:
            self.sentiment = max(self.gw_floor, self.sentiment * self.gw_decay)
        else:
            self.sentiment = min(self.gw_cap, self.sentiment * self.gw_growth)

    def get_current_mu(self, t):
        """Calculates base mu for time t (before noise).

        Effects are composable — multiple effects are applied multiplicatively.
        For example, effects=['seasonal', 'shock'] applies both a seasonal pattern
        AND a demand shock on top of each other.

        If ``external_series`` was provided at construction, its value at index
        ``t`` is used as the base mu instead of ``self.base_mu``.
        """
        # Use external series if available, otherwise fall back to constant base_mu
        using_external_series = self.external_series is not None
        if using_external_series:
            t_clamped = min(int(t), len(self.external_series) - 1)
            mu = float(self.external_series[t_clamped])
        else:
            mu = self.base_mu

        # 1. Apply Composable Non-Stationary Effects (if/if/if instead of if/elif)
        if "trend" in self.effects:
            mu *= 1 + self.trend_slope * t

        if "seasonal" in self.effects:
            mu *= 1 + self.seasonal_amp * np.sin(self.seasonal_freq * t)

        if "shock" in self.effects:
            if t >= self.shock_time:
                mu *= self.shock_mag

        # 2. Apply Endogenous Goodwill
        if self.use_goodwill:
            mu *= self.sentiment

        # External empirical traces, such as M5 unit sales, can legitimately
        # contain zero-demand days. Synthetic parametric demand keeps the small
        # positive floor used by the benchmark configuration.
        return max(0.0 if using_external_series else 0.1, mu)

    def get_observation(self, t):
        """
        Returns raw features: [time_step, sentiment_multiplier].
        Normalization or encoding (e.g. sin/cos) can be applied externally.
        """
        # Return shape: (2,)
        return np.array([t, self.sentiment], dtype=np.float64)
