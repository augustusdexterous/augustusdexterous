"""Shared helpers for trial-return matrices of shape (T, N): rows are periods, columns are trials."""

from __future__ import annotations

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


def column_sharpe_per_period(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Non-annualized Sharpe ratio of every column. Zero-volatility columns get 0.0."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    scale = np.max(np.abs(matrix), axis=0)
    flat = std <= _ZERO_VOL_RTOL * scale
    safe_std = np.where(flat, 1.0, std)
    return np.where(flat, 0.0, mean / safe_std)


def column_moments(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Moment-based skewness and raw kurtosis (normal = 3) of every column."""
    skew = stats.skew(matrix, axis=0, bias=True)
    kurt = stats.kurtosis(matrix, axis=0, fisher=False, bias=True)
    return np.asarray(skew, dtype=np.float64), np.asarray(kurt, dtype=np.float64)
