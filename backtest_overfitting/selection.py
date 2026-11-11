"""Out-of-sample evaluation of "pick the best trial on the past" with walk-forward windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._matrix import (
    as_returns_matrix,
    column_sharpe_per_period,
    validate_periods_per_year,
)
from .cv import walk_forward_split

__all__ = ["SelectionResult", "walk_forward_selection"]


@dataclass(frozen=True)
class SelectionResult:
    n_folds: int
    selected: np.ndarray  # trial chosen on each training window
    is_sharpe: np.ndarray  # its per-period Sharpe on the training window
    oos_sharpe: np.ndarray  # its per-period Sharpe on the following test window
    oos_returns: np.ndarray  # concatenated test-window returns of the selected trials
    oos_sharpe_annual: float  # annualized Sharpe of oos_returns
    efficiency: float  # mean OOS Sharpe / mean IS Sharpe (nan if IS mean is 0)

    def summary(self) -> str:
        return (
            f"folds={self.n_folds}  OOS SR (ann.)={self.oos_sharpe_annual:.2f}  "
            f"efficiency={self.efficiency:.2f}"
        )


def walk_forward_selection(
    returns: ArrayLike,
    train_size: int,
    test_size: int,
    step: int | None = None,
    expanding: bool = False,
    periods_per_year: float = 252.0,
) -> SelectionResult:
    """Pick the best trial on each training window and score it on the next test window.

    ``expanding=True`` keeps all data from the start as training data; otherwise the
    training window rolls forward with fixed length. An efficiency well below 1 means
    the in-sample winners do not keep their edge.
    """
    periods_per_year = validate_periods_per_year(periods_per_year)
    matrix = as_returns_matrix(returns, min_rows=4, min_cols=2)
    t = matrix.shape[0]
    selected, is_s, oos_s, oos_r = [], [], [], []
    for train, test in walk_forward_split(t, train_size, test_size, step):
        if expanding:
            train = np.arange(0, train[-1] + 1, dtype=np.intp)
        train_sr = column_sharpe_per_period(matrix[train])
        best = int(np.argmax(train_sr))
        selected.append(best)
        is_s.append(train_sr[best])
        oos_s.append(column_sharpe_per_period(matrix[test])[best])
        oos_r.append(matrix[test, best])
    if not selected:
        raise ValueError("no walk-forward windows fit the data")
    oos_returns = np.concatenate(oos_r)
    is_arr, oos_arr = np.asarray(is_s), np.asarray(oos_s)
    mean_is = float(is_arr.mean())
    eff = float(oos_arr.mean() / mean_is) if mean_is != 0 else float("nan")
    return SelectionResult(
        n_folds=len(selected),
        selected=np.asarray(selected, dtype=np.intp),
        is_sharpe=is_arr * np.sqrt(float(periods_per_year)),
        oos_sharpe=oos_arr * np.sqrt(float(periods_per_year)),
        oos_returns=oos_returns,
        oos_sharpe_annual=float(
            column_sharpe_per_period(oos_returns[:, None])[0]
            * np.sqrt(periods_per_year)
        ),
        efficiency=eff,
    )
