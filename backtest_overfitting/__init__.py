"""Detect backtest overfitting in strategy searches.

Input convention: a return matrix of shape ``(T, N)``, rows are periods in time order
and columns are the trials (parameter configurations) you searched over.

Submodules
----------
metrics          Sharpe ratio, probabilistic and deflated Sharpe ratios.
pbo              Probability of backtest overfitting (CSCV).
multiple_testing Familywise and FDR corrections across trials, Sharpe haircut.
consistency      Chronological check: does the best in-sample trial hold up later?
selection        Walk-forward evaluation of "pick the best on the past".
cv               Purged k-fold (with embargo) and walk-forward index splitters.
report           ``analyze()``: all of the above in one call.
data             Synthetic return matrices for experiments and tests.
"""

from .consistency import ChronologicalCheck, chronological_check
from .cv import purged_kfold_split, walk_forward_split
from .data import simulate_trials
from .metrics import (
    EULER_MASCHERONI,
    deannualize_sharpe,
    deflated_sharpe_from_returns,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from .multiple_testing import (
    MultipleTestingResult,
    adjust_pvalues,
    benjamini_hochberg,
    bonferroni,
    haircut_sharpe,
    holm,
    multiple_testing,
    sharpe_pvalues,
)
from .pbo import PBOResult, probability_of_backtest_overfitting
from .report import OverfittingReport, analyze
from .selection import SelectionResult, walk_forward_selection

__version__ = "0.2.0"

__all__ = [
    "EULER_MASCHERONI",
    "ChronologicalCheck",
    "MultipleTestingResult",
    "OverfittingReport",
    "PBOResult",
    "SelectionResult",
    "__version__",
    "adjust_pvalues",
    "analyze",
    "benjamini_hochberg",
    "bonferroni",
    "chronological_check",
    "deannualize_sharpe",
    "deflated_sharpe_from_returns",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "haircut_sharpe",
    "holm",
    "multiple_testing",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "purged_kfold_split",
    "sharpe_pvalues",
    "sharpe_ratio",
    "simulate_trials",
    "walk_forward_selection",
    "walk_forward_split",
]
