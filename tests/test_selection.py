import numpy as np
import pytest

import backtest_overfitting as bo


def test_noise_selection_decays_out_of_sample():
    r = bo.simulate_trials(n_obs=2000, n_trials=50, seed=3)
    res = bo.walk_forward_selection(r, train_size=500, test_size=100)
    assert res.n_folds == 15
    assert res.oos_returns.size == 1500
    assert res.efficiency < 0.5


def test_skilled_trial_is_selected_every_fold():
    r = bo.simulate_trials(
        n_obs=2000, n_trials=20, n_skilled=1, true_sharpe_annual=3.0, seed=2
    )
    res = bo.walk_forward_selection(r, train_size=500, test_size=250, expanding=True)
    assert np.all(res.selected == 0)
    assert res.oos_sharpe_annual > 1.0


def test_validation():
    r = bo.simulate_trials(n_obs=100, n_trials=5, seed=0)
    with pytest.raises(ValueError):
        bo.walk_forward_selection(r, train_size=200, test_size=10)
