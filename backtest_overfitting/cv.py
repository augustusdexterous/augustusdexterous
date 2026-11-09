"""Time-series cross-validation: walk-forward and purged k-fold.

Splitters yield ``(train_idx, test_idx)`` integer arrays over row positions.
``cv_select`` runs any splitter over a trial-return matrix: pick the best trial
on each train fold, record its performance on the matching test fold.
Reference: Lopez de Prado (2018), "Advances in Financial Machine Learning", ch. 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from ._utils import as_returns_matrix
from .sharpe import sharpe_ratio

Split = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class WalkForward:
    """Walk-forward splitter.

    train_size : rows in the first (or every, if rolling) training window.
    test_size : rows in each test window.
    step : rows to advance between folds (default = test_size).
    expanding : True -> anchored/expanding train window; False -> rolling.
    gap : rows skipped between train end and test start (embargo).
    """

    train_size: int
    test_size: int
    step: int | None = None
    expanding: bool = False
    gap: int = 0

    def split(self, n_samples: int) -> Iterator[Split]:
        if self.train_size < 1 or self.test_size < 1 or self.gap < 0:
            raise ValueError("train_size and test_size must be >= 1, gap >= 0")
        step = self.step or self.test_size
        if step < 1:
            raise ValueError("step must be >= 1")
        start = 0
        train_end = self.train_size
        while train_end + self.gap + self.test_size <= n_samples:
            test_start = train_end + self.gap
            train_start = 0 if self.expanding else start
            yield (np.arange(train_start, train_end),
                   np.arange(test_start, test_start + self.test_size))
            start += step
            train_end += step

    def n_splits(self, n_samples: int) -> int:
        return sum(1 for _ in self.split(n_samples))


@dataclass(frozen=True)
class PurgedKFold:
    """K-fold over contiguous blocks with purging and embargo.

    label_horizon : each observation i is labelled using rows i..i+h (h = label
        horizon). Training rows whose label window overlaps the test block are
        purged. Alternatively pass ``t1`` to ``split`` with explicit label end
        indices per row.
    embargo : fraction of n_samples (float < 1) or row count (int >= 1) removed
        from training right after each test block.
    """

    n_splits: int = 5
    label_horizon: int = 0
    embargo: float | int = 0.0

    def _embargo_rows(self, n: int) -> int:
        if isinstance(self.embargo, float) and self.embargo < 1:
            return int(np.ceil(self.embargo * n))
        return int(self.embargo)

    def split(self, n_samples: int, t1: np.ndarray | None = None) -> Iterator[Split]:
        if self.n_splits < 2 or self.n_splits > n_samples:
            raise ValueError("n_splits must be in [2, n_samples]")
        idx = np.arange(n_samples)
        if t1 is None:
            t1 = np.minimum(idx + self.label_horizon, n_samples - 1)
        t1 = np.asarray(t1, dtype=int)
        if t1.shape != (n_samples,) or np.any(t1 < idx):
            raise ValueError("t1 must have one end index >= start index per row")
        emb = self._embargo_rows(n_samples)
        for test in np.array_split(idx, self.n_splits):
            t0, tn = test[0], test[-1]
            test_label_end = t1[test].max()
            # Purge: training label window [i, t1[i]] overlaps [t0, test_label_end].
            overlap = (idx <= test_label_end) & (t1 >= t0)
            embargoed = (idx > tn) & (idx <= test_label_end + emb)
            train_mask = ~overlap & ~embargoed
            yield idx[train_mask], test


@dataclass(frozen=True)
class CVSelectionResult:
    oos_returns: np.ndarray        # concatenated test-fold returns of selected trials
    oos_index: np.ndarray          # row positions of oos_returns
    selected: np.ndarray           # selected trial per fold
    is_scores: np.ndarray          # train metric of the selected trial per fold
    oos_scores: np.ndarray         # test metric of the selected trial per fold
    oos_sharpe: float              # Sharpe of the concatenated OOS series
    efficiency: float              # mean(oos_scores) / mean(is_scores)
    n_folds: int


def cv_select(
    returns,
    splitter,
    metric: Callable[[np.ndarray], np.ndarray] | None = None,
) -> CVSelectionResult:
    """Out-of-sample evaluation of "pick the best trial" under ``splitter``.

    A large gap between IS and OOS scores (low ``efficiency``) indicates the
    parameter search is fitting noise.
    """
    arr = as_returns_matrix(returns)
    metric = metric or (lambda m: np.atleast_1d(sharpe_ratio(m)))
    oos_r, oos_i, sel, is_s, oos_s = [], [], [], [], []
    for train, test in splitter.split(arr.shape[0]):
        if train.size < 2 or test.size < 2:
            continue
        m_is = np.asarray(metric(arr[train]), dtype=float)
        best = int(np.argmax(m_is))
        m_oos = np.asarray(metric(arr[test]), dtype=float)
        sel.append(best)
        is_s.append(m_is[best])
        oos_s.append(m_oos[best])
        oos_r.append(arr[test, best])
        oos_i.append(test)
    if not sel:
        raise ValueError("splitter produced no usable folds")
    oos_returns = np.concatenate(oos_r)
    is_s, oos_s = np.asarray(is_s), np.asarray(oos_s)
    mean_is = is_s.mean()
    eff = float(oos_s.mean() / mean_is) if mean_is != 0 else float("nan")
    return CVSelectionResult(
        oos_returns=oos_returns,
        oos_index=np.concatenate(oos_i),
        selected=np.asarray(sel, dtype=int),
        is_scores=is_s,
        oos_scores=oos_s,
        oos_sharpe=float(sharpe_ratio(oos_returns)),
        efficiency=eff,
        n_folds=len(sel),
    )
