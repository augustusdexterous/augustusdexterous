"""Detect backtest overfitting: deflated Sharpe, PBO (CSCV) and purged splitters.

Submodules
----------
metrics
    Sharpe ratio, expected maximum Sharpe, probabilistic and deflated Sharpe ratios.
pbo
    Probability of backtest overfitting via combinatorially symmetric CV.
cv
    Purged k-fold cross-validation with embargo, and walk-forward splits.
"""

from .cv import purged_kfold_split, walk_forward_split
from .metrics import (
    EULER_MASCHERONI,
    deannualize_sharpe,
    deflated_sharpe_from_returns,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from .pbo import PBOResult, probability_of_backtest_overfitting

__version__ = "0.1.0"

__all__ = [
    "EULER_MASCHERONI",
    "PBOResult",
    "__version__",
    "deannualize_sharpe",
    "deflated_sharpe_from_returns",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "purged_kfold_split",
    "sharpe_ratio",
    "walk_forward_split",
]
