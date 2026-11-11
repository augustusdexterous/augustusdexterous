"""Multiple-testing corrections for a set of strategy trials.

P-values test H0: true Sharpe <= ``sr_benchmark`` for each trial, using the
probabilistic Sharpe ratio (non-normal standard error) from :mod:`metrics`.
Adjustments: Bonferroni and Holm (familywise error), Benjamini-Hochberg and
Benjamini-Yekutieli (false discovery rate).

References: Harvey & Liu (2015), "Backtesting", Journal of Portfolio Management.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._matrix import (
    as_returns_matrix,
    column_moments,
    column_sharpe_per_period,
    flat_mask,
)
from .metrics import probabilistic_sharpe_ratio

__all__ = [
    "METHODS",
    "MultipleTestingResult",
    "adjust_pvalues",
    "benjamini_hochberg",
    "bonferroni",
    "haircut_sharpe",
    "holm",
    "multiple_testing",
    "sharpe_pvalues",
]

METHODS = ("bonferroni", "holm", "bh", "by")


def _validate_p(pvalues: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("pvalues is empty")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be finite and within [0, 1]")
    return p


def bonferroni(pvalues: ArrayLike) -> NDArray[np.float64]:
    """Bonferroni familywise adjustment: ``min(m * p, 1)``."""
    p = _validate_p(pvalues)
    return np.minimum(p * p.size, 1.0)


def holm(pvalues: ArrayLike) -> NDArray[np.float64]:
    """Holm step-down familywise adjustment (uniformly more powerful than Bonferroni)."""
    p = _validate_p(pvalues)
    m = p.size
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def benjamini_hochberg(
    pvalues: ArrayLike, dependent: bool = False
) -> NDArray[np.float64]:
    """Benjamini-Hochberg FDR adjustment; ``dependent=True`` gives Benjamini-Yekutieli."""
    p = _validate_p(pvalues)
    m = p.size
    order = np.argsort(p)[::-1]
    ranks = np.arange(m, 0, -1)
    c = float(np.sum(1.0 / np.arange(1, m + 1))) if dependent else 1.0
    adjusted = np.minimum.accumulate(p[order] * m * c / ranks)
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def adjust_pvalues(pvalues: ArrayLike, method: str = "holm") -> NDArray[np.float64]:
    """Adjust p-values with one of ``METHODS``: bonferroni, holm, bh, by."""
    key = method.lower()
    if key == "bonferroni":
        return bonferroni(pvalues)
    if key == "holm":
        return holm(pvalues)
    if key == "bh":
        return benjamini_hochberg(pvalues)
    if key == "by":
        return benjamini_hochberg(pvalues, dependent=True)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def sharpe_pvalues(
    returns: ArrayLike, sr_benchmark: float = 0.0
) -> NDArray[np.float64]:
    """One-sided p-value per trial for H0: per-period SR <= ``sr_benchmark``.

    Equals ``1 - PSR(sr_benchmark)``. Zero-volatility trials cannot reject, so they get p = 1.
    """
    matrix = as_returns_matrix(returns, min_rows=3)
    sr = column_sharpe_per_period(matrix)
    skew, kurt = column_moments(matrix)
    flat = flat_mask(matrix)
    pvals = np.ones(matrix.shape[1])
    for j in range(matrix.shape[1]):
        if flat[j]:
            continue
        psr = probabilistic_sharpe_ratio(
            sr[j], matrix.shape[0], skew[j], kurt[j], sr_benchmark
        )
        pvals[j] = 1.0 - psr
    return pvals


@dataclass(frozen=True)
class MultipleTestingResult:
    method: str
    alpha: float
    sharpe: NDArray[np.float64]  # per-period Sharpe of each trial
    pvalues: NDArray[np.float64]
    adjusted_pvalues: NDArray[np.float64]
    reject: NDArray[np.bool_]

    @property
    def n_significant(self) -> int:
        return int(self.reject.sum())


def multiple_testing(
    returns: ArrayLike,
    method: str = "holm",
    alpha: float = 0.05,
    sr_benchmark: float = 0.0,
) -> MultipleTestingResult:
    """Test every trial's Sharpe against ``sr_benchmark`` with a multiplicity correction."""
    matrix = as_returns_matrix(returns, min_rows=3)
    p = sharpe_pvalues(matrix, sr_benchmark)
    adjusted = adjust_pvalues(p, method)
    return MultipleTestingResult(
        method=method,
        alpha=alpha,
        sharpe=column_sharpe_per_period(matrix),
        pvalues=p,
        adjusted_pvalues=adjusted,
        reject=adjusted <= alpha,
    )


def haircut_sharpe(
    sharpe: float, n_obs: int, n_trials: int, method: str = "bonferroni"
) -> float:
    """Harvey-Liu Sharpe haircut: the per-period SR whose single-test p-value equals the
    adjusted p-value of ``sharpe``.

    Uses the normal approximation t = SR * sqrt(T). ``method`` is "bonferroni" (p * N) or
    "sidak" (1 - (1 - p)^N). Adjusted p-values are capped at 1; values below 1e-300 are
    floored there so the inverse stays finite.
    """
    if n_obs < 2 or n_trials < 1:
        raise ValueError("n_obs must be >= 2 and n_trials >= 1")
    if not np.isfinite(sharpe):
        raise ValueError("sharpe must be finite")
    log_p = float(stats.norm.logsf(sharpe * np.sqrt(n_obs)))
    p_single = float(np.exp(log_p))
    if method == "bonferroni":
        p_adj = min(p_single * n_trials, 1.0)
    elif method == "sidak":
        # -expm1(N * log1p(-p)) keeps precision when p is tiny.
        p_adj = (
            min(-np.expm1(n_trials * np.log1p(-p_single)), 1.0) if p_single < 1 else 1.0
        )
    else:
        raise ValueError("method must be 'bonferroni' or 'sidak'")
    if p_adj >= 0.5:
        return 0.0
    p_adj = max(p_adj, 1e-300)
    return float(stats.norm.isf(p_adj) / np.sqrt(n_obs))
