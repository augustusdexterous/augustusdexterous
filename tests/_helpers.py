import numpy as np

import backtest_overfitting as bo


def per_period_srs(matrix):
    return np.array(
        [
            bo.sharpe_ratio(matrix[:, j], periods_per_year=1)
            for j in range(matrix.shape[1])
        ]
    )


def dsr_of_column(matrix, j):
    """DSR of trial j, deflated by all columns of the matrix (the full search)."""
    var = float(np.var(per_period_srs(matrix), ddof=1))
    return bo.deflated_sharpe_from_returns(
        matrix[:, j], n_trials=matrix.shape[1], var_sharpe_trials=var
    )
