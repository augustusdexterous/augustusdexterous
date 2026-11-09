"""Synthetic trial-return matrices for examples and tests (no real market data)."""

from __future__ import annotations

import numpy as np


def simulate_trials(
    n_obs: int = 1000,
    n_trials: int = 50,
    true_sharpe_annual: float = 0.0,
    n_skilled: int = 0,
    vol: float = 0.01,
    correlation: float = 0.0,
    periods_per_year: float = 252.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Simulate a (n_obs, n_trials) matrix of daily-like strategy returns.

    All trials are pure noise except the first ``n_skilled`` columns, which get a
    drift giving ``true_sharpe_annual``. ``correlation`` adds a common factor.
    """
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((n_obs, 1))
    idio = rng.standard_normal((n_obs, n_trials))
    z = np.sqrt(correlation) * common + np.sqrt(1 - correlation) * idio
    r = vol * z
    if n_skilled:
        mu = true_sharpe_annual / np.sqrt(periods_per_year) * vol
        r[:, :n_skilled] += mu
    return r
