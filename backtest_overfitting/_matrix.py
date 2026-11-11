"""Shared helpers for trial-return matrices of shape (T, N): rows are periods, columns are trials."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

# Same relative tolerance as metrics._ZERO_VOL_RTOL: a column whose sample std is
# float noise relative to its largest entry is treated as having zero volatility.
_ZERO_VOL_RTOL = 1e-12


def as_returns_matrix(
    returns: ArrayLike, min_rows: int = 3, min_cols: int = 1
) -> NDArray[np.float64]:
    """Validate and return a finite float64 array of shape (T, N).

    A 1-D input is treated as a single trial. Objects with ``to_numpy()`` (e.g. pandas
    DataFrames) are accepted.
    """
    if hasattr(returns, "to_numpy"):
        returns = returns.to_numpy()
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"returns must be 1-D or 2-D (T x N), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contain NaN or infinite values")
    t, n = arr.shape
    if t < min_rows:
        raise ValueError(f"need at least {min_rows} periods, got {t}")
    if n < min_cols:
        raise ValueError(f"need at least {min_cols} trials, got {n}")
    return arr


def validate_periods_per_year(periods_per_year: float) -> float:
    """Require a finite, strictly positive number of periods per year."""
    value = float(periods_per_year)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(
            f"periods_per_year must be a finite positive number, got {periods_per_year}"
        )
    return value


def flat_mask(matrix: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Columns with zero volatility (constant, including all zeros)."""
    std = matrix.std(axis=0, ddof=1)
    scale = np.max(np.abs(matrix), axis=0)
    return std <= _ZERO_VOL_RTOL * scale


def column_sharpe_per_period(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Non-annualized Sharpe ratio of every column.

    All-zero columns (no trading) score 0.0. A constant non-zero column has an
    undefined (infinite) Sharpe ratio, so it is rejected instead of being ranked.
    """
    mean = matrix.mean(axis=0)
    flat = flat_mask(matrix)
    bad = np.flatnonzero(flat & (mean != 0))
    if bad.size:
        raise ValueError(
            f"trial {int(bad[0])} has constant non-zero returns; its Sharpe ratio is undefined. "
            "Remove it or check how the return series was built."
        )
    std = matrix.std(axis=0, ddof=1)
    safe_std = np.where(flat, 1.0, std)
    return np.where(flat, 0.0, mean / safe_std)


def column_moments(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Moment-based skewness and raw kurtosis (normal = 3) of every column.

    Zero-volatility columns get the normal-distribution values (skew 0, kurtosis 3)
    because their moments are undefined.
    """
    flat = flat_mask(matrix)
    with np.errstate(all="ignore"):
        skew = stats.skew(matrix, axis=0, bias=True)
        kurt = stats.kurtosis(matrix, axis=0, fisher=False, bias=True)
    skew = np.where(flat, 0.0, np.asarray(skew, dtype=np.float64))
    kurt = np.where(flat, 3.0, np.asarray(kurt, dtype=np.float64))
    return skew, kurt
