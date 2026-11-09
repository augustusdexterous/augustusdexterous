"""One-call overfitting report for a trial-return matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._utils import as_returns_matrix
from .cv import CVSelectionResult, PurgedKFold, WalkForward, cv_select
from .multiple_testing import MultipleTestingResult, multiple_testing
from .pbo import PBOResult, pbo_cscv
from .sharpe import DeflatedSharpeResult, annualize_sharpe, deflated_sharpe_ratio, sharpe_ratio


@dataclass(frozen=True)
class OverfittingReport:
    n_obs: int
    n_trials: int
    best_trial: int
    best_sharpe_annual: float
    dsr: DeflatedSharpeResult
    pbo: PBOResult
    multiple_testing: MultipleTestingResult
    walk_forward: CVSelectionResult
    purged_kfold: CVSelectionResult
    periods_per_year: float

    def summary(self) -> str:
        a = np.sqrt(self.periods_per_year)
        mt = self.multiple_testing
        rows = [
            ("observations x trials", f"{self.n_obs} x {self.n_trials}"),
            ("best trial (IS)", f"#{self.best_trial}  SR={self.best_sharpe_annual:.2f} (ann.)"),
            ("expected max SR under null", f"{self.dsr.sr_benchmark * a:.2f} (ann.)"),
            ("Deflated Sharpe Ratio", f"{self.dsr.dsr:.3f}"),
            (f"PBO (CSCV, S={self.pbo.n_splits})", f"{self.pbo.pbo:.3f}"),
            ("P(OOS loss | IS-best)", f"{self.pbo.prob_oos_loss:.3f}"),
            (f"significant ({mt.method}, a={mt.alpha})", f"{mt.n_significant}/{self.n_trials}"),
            ("walk-forward OOS SR", f"{self.walk_forward.oos_sharpe * a:.2f} (ann.), "
                                    f"efficiency={self.walk_forward.efficiency:.2f}"),
            ("purged k-fold OOS SR", f"{self.purged_kfold.oos_sharpe * a:.2f} (ann.), "
                                     f"efficiency={self.purged_kfold.efficiency:.2f}"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k:<{width}} : {v}" for k, v in rows)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


def analyze(
    returns,
    periods_per_year: float = 252.0,
    n_splits: int = 10,
    n_trials: float | None = None,
    mt_method: str = "holm",
    alpha: float = 0.05,
    wf_train: int | None = None,
    wf_test: int | None = None,
    kfold: int = 5,
    label_horizon: int = 0,
    embargo: float | int = 0.01,
) -> OverfittingReport:
    """Run DSR, PBO, multiple testing, walk-forward and purged k-fold at once.

    ``returns`` is a (T, N) matrix: rows = periods, columns = parameter trials.
    ``n_trials`` overrides the trial count used by the DSR (e.g. if more
    configurations were tried than are in the matrix).
    """
    arr = as_returns_matrix(returns, min_rows=2 * n_splits, min_cols=2)
    t, n = arr.shape
    srs = np.atleast_1d(sharpe_ratio(arr))
    best = int(np.argmax(srs))
    wf_train = wf_train or max(t // 2, 2)
    wf_test = wf_test or max((t - wf_train) // 5, 2)
    return OverfittingReport(
        n_obs=t,
        n_trials=n,
        best_trial=best,
        best_sharpe_annual=float(annualize_sharpe(srs[best], periods_per_year)),
        dsr=deflated_sharpe_ratio(arr, n_trials=n_trials),
        pbo=pbo_cscv(arr, n_splits=n_splits),
        multiple_testing=multiple_testing(arr, method=mt_method, alpha=alpha),
        walk_forward=cv_select(arr, WalkForward(wf_train, wf_test)),
        purged_kfold=cv_select(arr, PurgedKFold(kfold, label_horizon, embargo)),
        periods_per_year=periods_per_year,
    )
