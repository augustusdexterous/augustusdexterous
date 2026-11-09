"""Input validation helpers shared across the package."""

from __future__ import annotations

import numpy as np


def as_returns_matrix(returns, min_rows: int = 2, min_cols: int = 1) -> np.ndarray:
    """Coerce input to a finite 2-D float array of shape (T, N).

    Rows are time periods, columns are parameter trials (strategy variants).
    A 1-D input is treated as a single trial. Objects with a ``.to_numpy()``
    method (e.g. pandas DataFrames) are accepted.
    """
    if hasattr(returns, "to_numpy"):
        returns = returns.to_numpy()
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"returns must be 1-D or 2-D, got ndim={arr.ndim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contain NaN or infinite values")
    t, n = arr.shape
    if t < min_rows:
        raise ValueError(f"need at least {min_rows} time periods, got {t}")
    if n < min_cols:
        raise ValueError(f"need at least {min_cols} trials, got {n}")
    return arr


def as_returns_vector(returns, min_rows: int = 2) -> np.ndarray:
    """Coerce input to a finite 1-D float array."""
    if hasattr(returns, "to_numpy"):
        returns = returns.to_numpy()
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError("expected a 1-D return series")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contain NaN or infinite values")
    if arr.size < min_rows:
        raise ValueError(f"need at least {min_rows} observations, got {arr.size}")
    return arr
