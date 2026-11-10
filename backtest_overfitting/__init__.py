"""Backtest overfitting toolkit.

Input convention: a returns matrix of shape (T, N), rows = time periods,
columns = parameter trials (one column per configuration tested).
"""

from .consistency import ChronologicalCheck, chronological_check
from .cv import CVSelectionResult, PurgedKFold, WalkForward, cv_select
from .data import simulate_trials
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
from .pbo import PBOResult, pbo_cscv
from .report import OverfittingReport, analyze
from .sharpe import (
    DeflatedSharpeResult,
    annualize_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)

__version__ = "0.1.0"

__all__ = [
    "ChronologicalCheck", "CVSelectionResult", "DeflatedSharpeResult", "MultipleTestingResult",
    "OverfittingReport", "PBOResult", "PurgedKFold", "WalkForward",
    "adjust_pvalues", "analyze", "annualize_sharpe", "benjamini_hochberg",
    "bonferroni", "chronological_check", "cv_select", "deflated_sharpe_ratio", "expected_max_sharpe",
    "haircut_sharpe", "holm", "min_track_record_length", "multiple_testing",
    "pbo_cscv", "probabilistic_sharpe_ratio", "sharpe_pvalues", "sharpe_ratio",
    "simulate_trials",
]
