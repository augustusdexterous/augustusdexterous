import pytest

import backtest_overfitting as bo


def test_noise_does_not_transfer():
    r = bo.simulate_trials(n_obs=2000, n_trials=100, seed=7)
    c = bo.chronological_check(r, split=1000)
    assert abs(c.rank_correlation) < 0.3
    assert 0 <= c.oos_percentile_best <= 100


def test_persistent_skill_transfers():
    r = bo.simulate_trials(
        n_obs=2000, n_trials=20, n_skilled=1, true_sharpe_annual=3.0, seed=2
    )
    c = bo.chronological_check(r, split=1000)
    assert c.is_best == 0
    assert c.oos_percentile_best >= 90
    assert c.oos_sharpe_best > c.oos_sharpe_all


def test_validation():
    r = bo.simulate_trials(n_obs=100, n_trials=5, seed=0)
    with pytest.raises(ValueError):
        bo.chronological_check(r, split=1)
    with pytest.raises(ValueError):
        bo.chronological_check(r, split=50, top_k=6)
