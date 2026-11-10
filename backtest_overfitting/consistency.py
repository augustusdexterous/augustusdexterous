"""Chronological consistency check: does the in-sample ranking survive a later period?

Unlike CSCV (which mixes blocks from across the sample), this uses one
time-ordered split: select on rows ``[0, split)``, judge on rows ``[split, T)``.
It catches regime changes that block-based methods can miss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import rankdata

from ._matrix import as_returns_matrix, column_sharpe_per_period

__all__ = ["ChronologicalCheck", "chronological_check"]


@dataclass(frozen=True)
class ChronologicalCheck:
    split: int
    n_trials: int
    is_best: int  # trial with the highest in-sample Sharpe
    is_sharpe_best: float  # its in-sample Sharpe (annualized)
    oos_sharpe_best: float  # its out-of-sample Sharpe (annualized)
    oos_percentile_best: float  # share of trials with lower OOS Sharpe, in [0, 100]
    rank_correlation: float  # Spearman corr. of IS vs OOS Sharpe across trials
    top_k: int
    oos_sharpe_top_k: float  # mean OOS Sharpe of the top-k in-sample trials
    oos_sharpe_all: float  # mean OOS Sharpe over all trials

    def summary(self) -> str:
        rows = [
            ("split (row index)", f"{self.split} | {self.n_trials} trials"),
            ("IS-best trial", f"#{self.is_best}  IS SR={self.is_sharpe_best:.2f}"),
            (
                "IS-best OOS SR",
                f"{self.oos_sharpe_best:.2f}  (OOS percentile {self.oos_percentile_best:.0f}%)",
            ),
            ("IS vs OOS rank corr.", f"{self.rank_correlation:.2f}"),
            (f"OOS SR of IS top-{self.top_k}", f"{self.oos_sharpe_top_k:.2f}"),
            ("OOS SR of all trials", f"{self.oos_sharpe_all:.2f}"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k:<{width}} : {v}" for k, v in rows)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def chronological_check(
    returns: ArrayLike,
    split: int,
    periods_per_year: float = 252.0,
    top_k: int = 10,
) -> ChronologicalCheck:
    """Select on rows ``[0, split)``, evaluate on rows ``[split, T)``.

    ``returns`` is (T, N) with rows in time order and one column per trial. A rank
    correlation near zero, or an in-sample best trial near the out-of-sample median,
    means the selection did not carry over to the later period.
    """
    matrix = as_returns_matrix(returns, min_rows=4, min_cols=2)
    t, n = matrix.shape
    if not 2 <= split <= t - 2:
        raise ValueError(f"split must be in [2, {t - 2}], got {split}")
    if not 1 <= top_k <= n:
        raise ValueError(f"top_k must be in [1, {n}], got {top_k}")
    ann = np.sqrt(float(periods_per_year))
    sr_is = column_sharpe_per_period(matrix[:split]) * ann
    sr_oos = column_sharpe_per_period(matrix[split:]) * ann
    best = int(np.argmax(sr_is))
    top = np.argsort(sr_is)[-top_k:]
    return ChronologicalCheck(
        split=int(split),
        n_trials=n,
        is_best=best,
        is_sharpe_best=float(sr_is[best]),
        oos_sharpe_best=float(sr_oos[best]),
        oos_percentile_best=float(np.mean(sr_oos < sr_oos[best]) * 100),
        rank_correlation=_spearman(sr_is, sr_oos),
        top_k=int(top_k),
        oos_sharpe_top_k=float(sr_oos[top].mean()),
        oos_sharpe_all=float(sr_oos.mean()),
    )
