"""Sharpe-ratio statistics: PSR, expected max Sharpe, Deflated Sharpe Ratio.

All Sharpe ratios are *per period* (not annualized) unless stated otherwise.
References:
    Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ._utils import as_returns_matrix, as_returns_vector

EULER_GAMMA = 0.5772156649015329


def sharpe_ratio(returns, risk_free: float = 0.0, ddof: int = 1):
    """Per-period Sharpe ratio. 1-D input -> float, 2-D (T, N) -> array of N."""
    arr = np.asarray(returns.to_numpy() if hasattr(returns, "to_numpy") else returns, dtype=float)
    excess = arr - risk_free
    mu = excess.mean(axis=0)
    sd = excess.std(axis=0, ddof=ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), 0.0)
    return float(sr) if np.ndim(sr) == 0 else sr


def annualize_sharpe(sr, periods_per_year: float = 252.0):
    """Scale a per-period Sharpe ratio to annual units (sqrt-time rule)."""
    return np.asarray(sr) * np.sqrt(periods_per_year)


def _sr_std_error(sr: float, n_obs: int, skew: float, kurtosis: float) -> float:
    """Standard error of the SR estimator under non-normal returns.

    ``kurtosis`` is the *raw* (Pearson) kurtosis, equal to 3 for a normal.
    """
    var = (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2) / (n_obs - 1.0)
    return float(np.sqrt(max(var, 1e-300)))


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """PSR: probability that the true SR exceeds ``sr_benchmark`` (per period)."""
    r = as_returns_vector(returns, min_rows=3)
    sr = sharpe_ratio(r)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    se = _sr_std_error(sr, r.size, skew, kurt)
    return float(stats.norm.cdf((sr - sr_benchmark) / se))


def expected_max_sharpe(n_trials: int, sr_variance: float, sr_mean: float = 0.0) -> float:
    """Expected maximum of ``n_trials`` i.i.d. Sharpe estimates under the null.

    Uses the False Strategy Theorem approximation:
        E[max] ~ mean + sqrt(V) * ((1-g) Z^-1(1-1/N) + g Z^-1(1-1/(N e)))
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sr_variance < 0:
        raise ValueError("sr_variance must be non-negative")
    if n_trials == 1:
        return float(sr_mean)
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_mean + np.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


@dataclass(frozen=True)
class DeflatedSharpeResult:
    dsr: float               # probability the selected SR is real after deflation
    sharpe: float            # observed per-period SR of the selected strategy
    sr_benchmark: float      # expected max SR under the null (per period)
    n_trials: float          # (effective) number of independent trials
    n_obs: int
    skew: float
    kurtosis: float          # raw kurtosis (normal = 3)

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.dsr > 1.0 - alpha


def deflated_sharpe_ratio(
    returns,
    n_trials: float | None = None,
    sr_variance: float | None = None,
    selected: int | None = None,
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio of the best (or ``selected``) trial.

    Parameters
    ----------
    returns : array (T, N) or (T,)
        Trial return matrix. With a 2-D input, the SR variance across trials and
        the number of trials are estimated from it unless given explicitly.
        With a 1-D input, ``n_trials`` and ``sr_variance`` are required.
    n_trials : number of independent trials (override, e.g. an effective count
        from clustering of correlated trials).
    sr_variance : cross-sectional variance of per-period trial SRs (override).
    selected : column index to evaluate; default = column with the highest SR.
    """
    arr = as_returns_matrix(returns, min_rows=3)
    srs = sharpe_ratio(arr)
    srs = np.atleast_1d(srs)
    if arr.shape[1] == 1 and (n_trials is None or sr_variance is None):
        raise ValueError("for a single series supply both n_trials and sr_variance")
    n = float(arr.shape[1] if n_trials is None else n_trials)
    v = float(np.var(srs, ddof=1) if sr_variance is None else sr_variance)
    idx = int(np.argmax(srs)) if selected is None else int(selected)
    r = arr[:, idx]
    sr = float(srs[idx])
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    sr0 = expected_max_sharpe(max(int(round(n)), 1), v)
    se = _sr_std_error(sr, r.size, skew, kurt)
    dsr = float(stats.norm.cdf((sr - sr0) / se))
    return DeflatedSharpeResult(dsr, sr, sr0, n, int(r.size), skew, kurt)


def min_track_record_length(returns, sr_benchmark: float = 0.0, alpha: float = 0.05) -> float:
    """Minimum number of observations for SR > ``sr_benchmark`` at confidence 1-alpha."""
    r = as_returns_vector(returns, min_rows=3)
    sr = sharpe_ratio(r)
    if sr <= sr_benchmark:
        return float("inf")
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    z = stats.norm.ppf(1 - alpha)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (z / (sr - sr_benchmark)) ** 2)
