r"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation.

Reference: Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of
Backtest Overfitting", Journal of Computational Finance 20(4), 39-69.

Algorithm (CSCV)
----------------
Given a ``T x N`` matrix of per-period returns (one column per trial):

1. Split the rows into ``S`` contiguous blocks (``S`` even).
2. For every combination of ``S/2`` blocks used as in-sample (IS) data, the
   remaining blocks form the out-of-sample (OOS) data. There are
   ``C(S, S/2)`` such splits.
3. Select the trial with the best IS Sharpe ratio, ``n* = argmax SR_IS``.
4. Compute its OOS relative rank ``omega = rank_OOS(n*) / (N + 1)`` (ranks are
   1-based and ascending, so rank N is best) and its logit
   ``lambda = ln(omega / (1 - omega))``.
5. ``PBO = P(lambda <= 0)``, the fraction of splits in which the IS-selected
   trial ranks at or below the OOS median.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from itertools import combinations
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

from .metrics import _ZERO_VOL_RTOL

__all__ = ["PBOResult", "probability_of_backtest_overfitting"]


class PBOResult(TypedDict):
    """Return value of :func:`probability_of_backtest_overfitting`.

    Attributes
    ----------
    pbo:
        Probability of backtest overfitting, in [0, 1].
    logits:
        Logit of the OOS relative rank for each of the ``C(S, S/2)`` splits.
    rank_oos:
        1-based ascending OOS rank of the IS-selected trial for each split,
        in ``[1, N]`` (fractional values only when OOS Sharpe ratios tie).
    """

    pbo: float
    logits: NDArray[np.float64]
    rank_oos: NDArray[np.float64]


def _validate_inputs(trials: ArrayLike, n_splits: int) -> NDArray[np.float64]:
    """Check the trial matrix and split count, returning a float64 array."""
    splits = operator.index(n_splits)
    if splits < 2 or splits % 2 != 0:
        raise ValueError(f"n_splits must be an even integer >= 2, got {n_splits}")
    matrix = np.asarray(trials, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"trials must be 2-D (T x N), got shape {matrix.shape}")
    n_rows, n_cols = matrix.shape
    if n_cols < 2:
        raise ValueError("trials must contain at least 2 columns (trials)")
    if n_rows < splits:
        raise ValueError(f"need at least n_splits={splits} rows, got {n_rows}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("trials contains NaN or infinite values")
    # Every IS and OOS subset must hold at least 2 rows for a sample std.
    if (n_rows // splits) * (splits // 2) < 2:
        raise ValueError(
            "each in-sample and out-of-sample subset needs at least 2 rows"
        )
    return matrix


def _column_sharpe(data: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per-column, non-annualized Sharpe ratio :math:`\bar{r} / s_r` (``s`` with ``ddof=1``).

    A column is zero-volatility when its sample std is at most ``_ZERO_VOL_RTOL``
    times its largest absolute entry, the same tolerance used by
    :func:`metrics.sharpe_ratio`. Such columns are ranked +inf (positive mean),
    -inf (negative mean) or 0 (zero mean), so that constant winners are not
    discarded. The test is relative because the sample std of a constant column is
    float rounding noise (around 1e-18 for returns near 1e-3), not exactly zero, and
    an exact ``std > 0`` test would give such columns ratios near 1e15 whose order
    depends on rounding.
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1)
    scale = np.max(np.abs(data), axis=0)
    flat = np.where(mean > 0, np.inf, np.where(mean < 0, -np.inf, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mean / std
    return np.where(std <= _ZERO_VOL_RTOL * scale, flat, ratio)


def probability_of_backtest_overfitting(
    trials: ArrayLike,
    n_splits: int = 16,
) -> PBOResult:
    r"""Estimate the probability of backtest overfitting with CSCV.

    For each of the :math:`\binom{S}{S/2}` in-sample/out-of-sample partitions,
    the trial with the highest IS Sharpe ratio is chosen and its OOS relative
    rank is converted to a logit:

    .. math::

        \omega_c = \frac{\operatorname{rank}_{OOS}(n^*_c)}{N+1}, \qquad
        \lambda_c = \ln\frac{\omega_c}{1-\omega_c}, \qquad
        PBO = \frac{1}{\binom{S}{S/2}} \sum_c \mathbf{1}[\lambda_c \le 0]

    Parameters
    ----------
    trials:
        Array of shape ``(T, N)``: ``T`` periods of returns for ``N`` trials
        (strategies or parameter configurations). Use a ``(T, N)`` float array.
    n_splits:
        Even number ``S`` of contiguous row blocks. Larger values give more
        partitions (``C(16, 8) = 12870``) at higher cost.

    Returns
    -------
    PBOResult
        Dictionary with keys ``"pbo"``, ``"logits"`` and ``"rank_oos"``.

    Raises
    ------
    ValueError
        If ``trials`` is not 2-D with at least 2 columns, contains non-finite
        values, has fewer rows than ``n_splits``, or ``n_splits`` is not an even
        integer >= 2. Also raised when the smallest in-sample or out-of-sample
        subset would hold fewer than 2 rows, because each Sharpe ratio needs a
        sample standard deviation.
    TypeError
        If ``n_splits`` is not an integer.

    Notes
    -----
    A zero-volatility column with a positive mean ranks +inf, so several constant
    positive trials tie at the top. A tie at the out-of-sample median gives logit 0,
    which counts towards PBO under the ``<= 0`` rule.

    References
    ----------
    Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2017).
    The Probability of Backtest Overfitting. Journal of Computational Finance,
    20(4), 39-69.
    """
    matrix = _validate_inputs(trials, n_splits)
    _, n_cols = matrix.shape
    splits = int(n_splits)
    half = splits // 2

    # np.array_split keeps block sizes within one row of each other.
    blocks: Sequence[NDArray[np.float64]] = np.array_split(matrix, splits, axis=0)

    logits: list[float] = []
    ranks: list[float] = []
    for is_ids in combinations(range(splits), half):
        is_set = set(is_ids)
        oos_ids = [b for b in range(splits) if b not in is_set]
        is_data = np.concatenate([blocks[b] for b in is_ids], axis=0)
        oos_data = np.concatenate([blocks[b] for b in oos_ids], axis=0)

        best = int(np.argmax(_column_sharpe(is_data)))
        oos_ranks = rankdata(_column_sharpe(oos_data), method="average")
        rank = float(oos_ranks[best])

        omega = rank / (n_cols + 1)
        logits.append(float(np.log(omega / (1.0 - omega))))
        ranks.append(rank)

    logit_arr = np.asarray(logits, dtype=np.float64)
    rank_arr = np.asarray(ranks, dtype=np.float64)
    return PBOResult(
        pbo=float(np.mean(logit_arr <= 0.0)),
        logits=logit_arr,
        rank_oos=rank_arr,
    )
