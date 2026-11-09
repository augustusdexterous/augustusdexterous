"""Probability of Backtest Overfitting via CSCV.

Reference: Bailey, Borwein, Lopez de Prado & Zhu (2016),
"The Probability of Backtest Overfitting".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Callable

import numpy as np
from scipy import stats

from ._utils import as_returns_matrix
from .sharpe import sharpe_ratio

Metric = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PBOResult:
    pbo: float                      # P(logit <= 0): best IS trial is below OOS median
    logits: np.ndarray              # one logit per combination
    is_performance: np.ndarray      # IS metric of the IS-best trial, per combination
    oos_performance: np.ndarray     # OOS metric of that same trial, per combination
    oos_rank: np.ndarray            # relative OOS rank in (0, 1)
    selected: np.ndarray            # IS-best trial index per combination
    n_splits: int
    n_combinations: int
    degradation_slope: float        # OLS slope of OOS on IS performance
    prob_oos_loss: float            # P(OOS metric of selected trial < 0)
    extra: dict = field(default_factory=dict)


def pbo_cscv(
    returns,
    n_splits: int = 16,
    metric: Metric | None = None,
    max_combinations: int | None = None,
    random_state: int | None = None,
) -> PBOResult:
    """Combinatorially Symmetric Cross-Validation.

    Parameters
    ----------
    returns : array (T, N) - rows are periods, columns are parameter trials.
    n_splits : even number S of contiguous row blocks. C(S, S/2) combinations.
    metric : function mapping a (t, N) sub-matrix to N scores (default: Sharpe).
    max_combinations : if set and smaller than C(S, S/2), sample that many
        combinations at random (reproducible via ``random_state``).
    """
    arr = as_returns_matrix(returns, min_cols=2)
    t, n = arr.shape
    if n_splits < 2 or n_splits % 2:
        raise ValueError("n_splits must be an even integer >= 2")
    if t < 2 * n_splits:
        raise ValueError(f"need at least {2 * n_splits} rows for n_splits={n_splits}")
    metric = metric or (lambda m: np.atleast_1d(sharpe_ratio(m)))

    blocks = np.array_split(np.arange(t), n_splits)
    all_combos = combinations(range(n_splits), n_splits // 2)
    total = comb(n_splits, n_splits // 2)
    if max_combinations is not None and max_combinations < total:
        rng = np.random.default_rng(random_state)
        combos_list = list(all_combos)
        pick = rng.choice(total, size=max_combinations, replace=False)
        combos = [combos_list[i] for i in sorted(pick)]
    else:
        combos = list(all_combos)

    logits, is_perf, oos_perf, ranks, sel = [], [], [], [], []
    for combo in combos:
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        m_is = np.asarray(metric(arr[is_idx]), dtype=float)
        m_oos = np.asarray(metric(arr[oos_idx]), dtype=float)
        best = int(np.argmax(m_is))
        # Relative rank of the IS-best trial within the OOS distribution, in (0, 1).
        rank = stats.rankdata(m_oos)[best]
        omega = rank / (n + 1.0)
        logits.append(np.log(omega / (1.0 - omega)))
        is_perf.append(m_is[best])
        oos_perf.append(m_oos[best])
        ranks.append(omega)
        sel.append(best)

    logits = np.asarray(logits)
    is_perf = np.asarray(is_perf)
    oos_perf = np.asarray(oos_perf)
    if np.ptp(is_perf) > 0:
        slope = float(np.polyfit(is_perf, oos_perf, 1)[0])
    else:
        slope = float("nan")
    return PBOResult(
        pbo=float(np.mean(logits <= 0)),
        logits=logits,
        is_performance=is_perf,
        oos_performance=oos_perf,
        oos_rank=np.asarray(ranks),
        selected=np.asarray(sel, dtype=int),
        n_splits=n_splits,
        n_combinations=len(combos),
        degradation_slope=slope,
        prob_oos_loss=float(np.mean(oos_perf < 0)),
    )
