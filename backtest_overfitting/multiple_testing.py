"""Multiple-testing corrections for many strategy trials.

Provides Bonferroni, Holm (FWER) and Benjamini-Hochberg / Benjamini-Yekutieli
(FDR) adjustments, plus Sharpe-based p-values and a Sharpe "haircut".
Reference: Harvey & Liu (2015), "Backtesting".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ._utils import as_returns_matrix
from .sharpe import _sr_std_error, sharpe_ratio

METHODS = ("bonferroni", "holm", "bh", "by")


def _validate_p(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("pvalues is empty")
    if np.any((p < 0) | (p > 1)) or not np.all(np.isfinite(p)):
        raise ValueError("pvalues must be finite and within [0, 1]")
    return p


def bonferroni(pvalues) -> np.ndarray:
    p = _validate_p(pvalues)
    return np.minimum(p * p.size, 1.0)


def holm(pvalues) -> np.ndarray:
    p = _validate_p(pvalues)
    m = p.size
    order = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def benjamini_hochberg(pvalues, dependent: bool = False) -> np.ndarray:
    """BH step-up FDR adjustment; ``dependent=True`` gives Benjamini-Yekutieli."""
    p = _validate_p(pvalues)
    m = p.size
    order = np.argsort(p)[::-1]
    ranks = np.arange(m, 0, -1)
    c = np.sum(1.0 / np.arange(1, m + 1)) if dependent else 1.0
    adj = np.minimum.accumulate(p[order] * m * c / ranks)
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def adjust_pvalues(pvalues, method: str = "holm") -> np.ndarray:
    method = method.lower()
    if method == "bonferroni":
        return bonferroni(pvalues)
    if method == "holm":
        return holm(pvalues)
    if method == "bh":
        return benjamini_hochberg(pvalues)
    if method == "by":
        return benjamini_hochberg(pvalues, dependent=True)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def sharpe_pvalues(returns, sr_benchmark: float = 0.0) -> np.ndarray:
    """One-sided p-values for H0: SR <= sr_benchmark, one per trial column.

    Uses the non-normal SR standard error (skew/kurtosis aware), i.e. 1 - PSR.
    """
    arr = as_returns_matrix(returns, min_rows=3)
    srs = np.atleast_1d(sharpe_ratio(arr))
    skews = stats.skew(arr, axis=0, bias=False)
    kurts = stats.kurtosis(arr, axis=0, fisher=False, bias=False)
    t = arr.shape[0]
    z = np.array([(sr - sr_benchmark) / _sr_std_error(sr, t, s, k)
                  for sr, s, k in zip(srs, skews, kurts)])
    return stats.norm.sf(z)


@dataclass(frozen=True)
class MultipleTestingResult:
    method: str
    alpha: float
    sharpe: np.ndarray
    pvalues: np.ndarray
    adjusted_pvalues: np.ndarray
    reject: np.ndarray

    @property
    def n_significant(self) -> int:
        return int(self.reject.sum())


def multiple_testing(returns, method: str = "holm", alpha: float = 0.05,
                     sr_benchmark: float = 0.0) -> MultipleTestingResult:
    """Test every trial's Sharpe against ``sr_benchmark`` with a correction."""
    arr = as_returns_matrix(returns, min_rows=3)
    p = sharpe_pvalues(arr, sr_benchmark)
    adj = adjust_pvalues(p, method)
    return MultipleTestingResult(method, alpha, np.atleast_1d(sharpe_ratio(arr)),
                                 p, adj, adj <= alpha)


def haircut_sharpe(sr: float, n_obs: int, n_trials: int, method: str = "bonferroni") -> float:
    """Harvey-Liu Sharpe haircut: SR implied by the multiple-testing-adjusted p-value.

    ``sr`` is per period; returns the haircut per-period SR (>= 0) under a
    normal-returns t-statistic approximation t = SR * sqrt(T). Method is
    "bonferroni" or "sidak".
    """
    if n_obs < 2 or n_trials < 1:
        raise ValueError("n_obs must be >= 2 and n_trials >= 1")
    p_single = stats.norm.sf(sr * np.sqrt(n_obs))
    if method == "bonferroni":
        p_adj = min(p_single * n_trials, 1.0)
    elif method == "sidak":
        p_adj = 1.0 - (1.0 - p_single) ** n_trials
    else:
        raise ValueError("method must be 'bonferroni' or 'sidak'")
    if p_adj >= 0.5:
        return 0.0
    return float(stats.norm.isf(p_adj) / np.sqrt(n_obs))
