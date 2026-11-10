"""Monte Carlo checks that error rates and power are what the theory promises."""

from _helpers import dsr_of_column

import backtest_overfitting as bo


def test_holm_familywise_error_is_controlled_under_null():
    reps = 300
    hits = sum(
        bo.multiple_testing(bo.simulate_trials(500, 50, seed=s), "holm").n_significant
        > 0
        for s in range(reps)
    )
    assert hits / reps <= 0.10  # nominal 0.05; allows for Monte Carlo noise


def test_dsr_false_positive_rate_under_null():
    reps = 300
    hits = sum(
        dsr_of_column(bo.simulate_trials(500, 50, seed=s), 0) > 0.95
        for s in range(reps)
    )
    assert hits / reps <= 0.05


def test_dsr_detects_strong_skill_with_enough_data():
    reps = 100
    hits = sum(
        dsr_of_column(
            bo.simulate_trials(1500, 50, n_skilled=1, true_sharpe_annual=3.0, seed=s), 0
        )
        > 0.95
        for s in range(reps)
    )
    assert hits / reps >= 0.9
