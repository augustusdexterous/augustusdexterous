"""One-call overfitting report for a trial-return matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._matrix import (
    as_returns_matrix,
    column_moments,
    column_sharpe_per_period,
    validate_periods_per_year,
)
from .consistency import ChronologicalCheck, chronological_check
from .metrics import deflated_sharpe_ratio, expected_max_sharpe
from .multiple_testing import MultipleTestingResult, multiple_testing
from .pbo import PBOResult, probability_of_backtest_overfitting

__all__ = ["OverfittingReport", "analyze"]


@dataclass(frozen=True)
class OverfittingReport:
    n_obs: int
    n_trials: int
    periods_per_year: float
    n_splits: int
    best_trial: int
    best_sharpe: float  # annualized, full sample
    sr_benchmark: float  # expected max Sharpe under the null, annualized
    deflated_sharpe: float  # DSR of the best trial, in [0, 1]
    pbo: PBOResult
    multiple_testing: MultipleTestingResult
    chronological: ChronologicalCheck

    def summary(self) -> str:
        mt = self.multiple_testing
        c = self.chronological
        rows = [
            ("observations x trials", f"{self.n_obs} x {self.n_trials}"),
            (
                "best trial (full sample)",
                f"#{self.best_trial}  SR={self.best_sharpe:.2f} (ann.)",
            ),
            ("expected max SR under null", f"{self.sr_benchmark:.2f} (ann.)"),
            ("Deflated Sharpe Ratio", f"{self.deflated_sharpe:.3f}"),
            (
                f"PBO (CSCV, S={self.n_splits})",
                f"{self.pbo['pbo']:.3f}  ({len(self.pbo['logits'])} partitions)",
            ),
            (
                f"significant ({mt.method}, a={mt.alpha})",
                f"{mt.n_significant}/{self.n_trials}",
            ),
            (
                f"chronological split {c.split}",
                (
                    f"rank rho={c.rank_correlation:+.2f}, "
                    f"IS-best OOS pct={c.oos_percentile_best:.0f}%, "
                    f"OOS SR best={c.oos_sharpe_best:.2f} vs all={c.oos_sharpe_all:.2f}"
                ),
            ),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k:<{width}} : {v}" for k, v in rows)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


def analyze(
    returns: ArrayLike,
    periods_per_year: float = 252.0,
    n_splits: int = 10,
    mt_method: str = "holm",
    alpha: float = 0.05,
    split: int | None = None,
) -> OverfittingReport:
    """Run DSR, PBO, multiple testing and a chronological check in one call.

    ``returns`` is (T, N): rows are periods in time order, columns are trials.
    ``split`` is the row index for the chronological check (default: the midpoint).
    """
    periods_per_year = validate_periods_per_year(periods_per_year)
    matrix = as_returns_matrix(returns, min_rows=2 * n_splits, min_cols=2)
    t, n = matrix.shape
    sr_period = column_sharpe_per_period(matrix)
    skew, kurt = column_moments(matrix)
    best = int(np.argmax(sr_period))
    var_trials = float(np.var(sr_period, ddof=1))
    dsr = deflated_sharpe_ratio(
        sr_period[best], t, float(skew[best]), float(kurt[best]), n, var_trials
    )
    ann = np.sqrt(float(periods_per_year))
    return OverfittingReport(
        n_obs=t,
        n_trials=n,
        periods_per_year=float(periods_per_year),
        n_splits=int(n_splits),
        best_trial=best,
        best_sharpe=float(sr_period[best] * ann),
        sr_benchmark=float(expected_max_sharpe(n, var_trials) * ann),
        deflated_sharpe=float(dsr),
        pbo=probability_of_backtest_overfitting(matrix, n_splits=n_splits),
        multiple_testing=multiple_testing(matrix, method=mt_method, alpha=alpha),
        chronological=chronological_check(
            matrix,
            split=split if split is not None else t // 2,
            periods_per_year=periods_per_year,
            top_k=min(10, n),
        ),
    )
