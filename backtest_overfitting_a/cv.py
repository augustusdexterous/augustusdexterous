r"""Time-aware cross-validation splitters for financial data.

Purged k-fold with embargo follows Lopez de Prado (2018), "Advances in Financial
Machine Learning", Chapter 7. Each sample is an event whose label spans the
closed interval ``[t0[i], t1[i]]``, so neighbouring samples can share
information. Training samples whose intervals overlap the test span are
purged, and an embargo of ``embargo_pct * n_samples`` samples is dropped after
each test fold.

Walk-forward splitting produces rolling train/test windows in which the test
window always lies strictly after its training window by sample index. It does
not purge on label spans; see :func:`walk_forward_split`.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["purged_kfold_split", "walk_forward_split"]

_Split = tuple[NDArray[np.intp], NDArray[np.intp]]

# Relative tolerance for float noise in ``embargo_pct * n``, e.g. 0.07 * 100 = 7.000000000000001.
_EMBARGO_RTOL: float = 1e-12


def _validate_intervals(
    t0: ArrayLike, t1: ArrayLike
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Validate event start/end arrays.

    Checks that both arrays are 1-D, non-empty, of equal length and finite, that
    ``t1 >= t0`` elementwise, and that ``t0`` is non-decreasing. Finiteness must
    be checked explicitly because NaN makes every comparison False, which would
    silently disable purging. Sortedness compares neighbours directly rather than
    through ``np.diff``, which wraps around for unsigned integer dtypes.
    """
    starts = np.asarray(t0)
    ends = np.asarray(t1)
    if starts.ndim != 1 or ends.ndim != 1:
        raise ValueError("t0 and t1 must be one-dimensional")
    if starts.shape != ends.shape:
        raise ValueError(
            f"t0 and t1 must have the same length, got {starts.size} and {ends.size}"
        )
    if starts.size == 0:
        raise ValueError("t0 and t1 must not be empty")
    if not (np.all(np.isfinite(starts)) and np.all(np.isfinite(ends))):
        raise ValueError("t0 and t1 must contain only finite values")
    if np.any(ends < starts):
        raise ValueError("every event must satisfy t1 >= t0")
    if np.any(starts[1:] < starts[:-1]):
        raise ValueError("t0 must be sorted in non-decreasing order")
    return starts, ends


def _embargo_size(embargo_pct: float, n: int) -> int:
    r"""Number of samples to embargo, :math:`h = \lceil \text{embargo\_pct} \cdot N \rceil`.

    A product within ``_EMBARGO_RTOL`` (relative) of a positive integer is snapped
    to that integer, so float noise such as ``0.07 * 100 = 7.000000000000001`` does
    not add a sample. Any other positive product rounds up, however small it is.
    """
    product = embargo_pct * n
    nearest = round(product)
    if nearest > 0 and abs(product - nearest) <= _EMBARGO_RTOL * nearest:
        return int(nearest)
    return math.ceil(product)


def _purged_kfold_gen(
    starts: NDArray[Any],
    ends: NDArray[Any],
    n_splits: int,
    embargo: int,
) -> Iterator[_Split]:
    n = starts.size
    indices = np.arange(n, dtype=np.intp)
    for test_idx in np.array_split(indices, n_splits):
        span_start = starts[test_idx].min()
        span_end = ends[test_idx].max()

        # Purge: drop any sample whose interval intersects the test span.
        overlaps = (starts <= span_end) & (ends >= span_start)
        excluded = overlaps.copy()
        excluded[test_idx] = True

        # Embargo: drop the samples that immediately follow the test fold.
        last = int(test_idx[-1])
        excluded[last + 1 : last + 1 + embargo] = True

        train_idx = indices[~excluded]
        yield train_idx, test_idx


def purged_kfold_split(
    t0: ArrayLike,
    t1: ArrayLike,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> Iterator[_Split]:
    r"""Purged k-fold cross-validation with an embargo.

    The samples are divided into ``n_splits`` contiguous test folds. For fold
    :math:`k` with test span :math:`[\min_{i\in F_k} t_{0,i},\ \max_{i\in F_k} t_{1,i}]`:

    * **Purge:** a training sample ``j`` is removed if
      :math:`t_{0,j} \le \max_{i\in F_k} t_{1,i}` and
      :math:`t_{1,j} \ge \min_{i\in F_k} t_{0,i}`, i.e. its label interval
      overlaps the test span.
    * **Embargo:** the next :math:`h = \lceil \text{embargo\_pct} \cdot N \rceil`
      samples after the test fold are removed from training.

    Parameters
    ----------
    t0:
        Sorted, non-decreasing event start indices (or any sortable numeric
        times), one per sample.
    t1:
        Event end indices, ``t1[i] >= t0[i]``, same length as ``t0``.
    n_splits:
        Number of folds: an integer between 2 and the number of samples.
    embargo_pct:
        Fraction of samples to embargo after each test fold, in ``[0, 1)``.

    Returns
    -------
    Iterator[tuple[ndarray, ndarray]]
        Yields ``(train_idx, test_idx)`` integer index arrays for each fold.
        Validation happens eagerly, before the first fold is produced.

    Raises
    ------
    ValueError
        On invalid interval arrays (including non-finite times or unsorted
        ``t0``), ``n_splits`` out of range, or ``embargo_pct`` outside ``[0, 1)``.
    TypeError
        If ``n_splits`` is not an integer.

    References
    ----------
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley,
    Chapter 7 (Cross-Validation in Finance).
    """
    starts, ends = _validate_intervals(t0, t1)
    n = starts.size
    folds = operator.index(n_splits)
    if not 2 <= folds <= n:
        raise ValueError(f"n_splits must be in [2, {n}], got {n_splits}")
    if not (math.isfinite(embargo_pct) and 0.0 <= embargo_pct < 1.0):
        raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct}")
    return _purged_kfold_gen(starts, ends, folds, _embargo_size(embargo_pct, n))


def _walk_forward_gen(
    n_samples: int,
    train_size: int,
    test_size: int,
    step: int,
) -> Iterator[_Split]:
    last_start = n_samples - train_size - test_size
    for start in range(0, last_start + 1, step):
        train_idx = np.arange(start, start + train_size, dtype=np.intp)
        test_idx = np.arange(
            start + train_size, start + train_size + test_size, dtype=np.intp
        )
        yield train_idx, test_idx


def walk_forward_split(
    n_samples: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> Iterator[_Split]:
    """Rolling walk-forward train/test windows.

    Window ``k`` trains on ``[s_k, s_k + train_size)`` and tests on
    ``[s_k + train_size, s_k + train_size + test_size)``, where
    ``s_k = k * step``. The test window always starts immediately after the
    training window, so the two never overlap and training data never follows
    test data.

    Parameters
    ----------
    n_samples:
        Total number of samples.
    train_size:
        Number of samples in each training window.
    test_size:
        Number of samples in each test window.
    step:
        Offset between consecutive window starts. Defaults to ``test_size``,
        which gives non-overlapping test windows.

    Returns
    -------
    Iterator[tuple[ndarray, ndarray]]
        Yields ``(train_idx, test_idx)`` integer index arrays. Validation happens
        eagerly, so an invalid size raises here rather than during iteration.
        Once validation passes the first window always fits, so at least one
        window is yielded.

    Raises
    ------
    ValueError
        If any size is smaller than 1, ``step`` is smaller than 1, or
        ``train_size + test_size > n_samples``.
    TypeError
        If ``n_samples``, a size or ``step`` is not an integer.

    Notes
    -----
    Windows are defined by sample position only. The test window starts after
    the training window by index, but no purging is done on label spans. If a
    training sample's label interval ``[t0, t1]`` reaches into the test window,
    it overlaps test labels, and this function cannot detect that because it
    receives no event times. When labels span more than one sample, use
    :func:`purged_kfold_split` for label-aware splits, or drop the overlapping
    training samples yourself.

    Examples
    --------
    >>> [ (tr.tolist(), te.tolist()) for tr, te in walk_forward_split(10, 4, 2) ][:2]
    [([0, 1, 2, 3], [4, 5]), ([2, 3, 4, 5], [6, 7])]
    """
    n = operator.index(n_samples)
    train = operator.index(train_size)
    test = operator.index(test_size)
    stride = test if step is None else operator.index(step)
    if train < 1 or test < 1:
        raise ValueError("train_size and test_size must be at least 1")
    if stride < 1:
        raise ValueError("step must be at least 1")
    if train + test > n:
        raise ValueError(
            f"train_size + test_size ({train + test}) exceeds n_samples ({n})"
        )
    return _walk_forward_gen(n, train, test, stride)
