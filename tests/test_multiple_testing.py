import numpy as np
import pytest

import backtest_overfitting as bo


def test_adjustments_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(bo.bonferroni(p), [0.04, 0.16, 0.12, 0.02])
    np.testing.assert_allclose(bo.holm(p), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(bo.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    assert np.all(bo.adjust_pvalues(p, "by") >= bo.benjamini_hochberg(p))


def test_adjust_rejects_unknown_method_and_bad_pvalues():
    with pytest.raises(ValueError):
        bo.adjust_pvalues([0.1], "nope")
    with pytest.raises(ValueError):
        bo.holm([1.5])
    with pytest.raises(ValueError):
        bo.bonferroni([])


def test_skilled_trial_survives_and_noise_does_not():
    noise = bo.simulate_trials(n_obs=1000, n_trials=100, seed=1)
    skilled = bo.simulate_trials(
        n_obs=1500, n_trials=20, n_skilled=1, true_sharpe_annual=3.0, seed=2
    )
    assert bo.multiple_testing(noise, "holm").n_significant == 0
    res = bo.multiple_testing(skilled, "bh")
    assert res.reject[0]
    assert res.n_significant <= 2


def test_zero_volatility_trial_gets_pvalue_one():
    r = bo.simulate_trials(n_obs=300, n_trials=3, seed=4)
    r[:, 0] = 0.0
    p = bo.sharpe_pvalues(r)
    assert p[0] == 1.0
    assert np.all((p >= 0) & (p <= 1))


def test_haircut_sharpe():
    sr = 2.0 / np.sqrt(252)
    h = bo.haircut_sharpe(sr, n_obs=1000, n_trials=100)
    assert 0 <= h < sr
    assert bo.haircut_sharpe(sr, 1000, 1) == pytest.approx(sr)
