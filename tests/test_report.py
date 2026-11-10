import numpy as np
import pytest
from _helpers import dsr_of_column

import backtest_overfitting as bo


@pytest.fixture(scope="module")
def noise():
    return bo.simulate_trials(n_obs=1000, n_trials=100, seed=1)


@pytest.fixture(scope="module")
def skilled():
    return bo.simulate_trials(
        n_obs=1500, n_trials=20, n_skilled=1, true_sharpe_annual=3.0, seed=2
    )


def test_report_on_noise_flags_overfitting(noise):
    rep = bo.analyze(noise)
    assert rep.n_trials == 100
    assert rep.deflated_sharpe < 0.95
    assert rep.pbo["pbo"] > 0.3
    assert rep.multiple_testing.n_significant == 0
    assert rep.best_sharpe > 0 and rep.sr_benchmark > 0


def test_report_on_skill_is_clean(skilled):
    rep = bo.analyze(skilled)
    assert rep.best_trial == 0
    assert rep.deflated_sharpe > 0.95
    assert rep.pbo["pbo"] < 0.1
    assert "Deflated Sharpe Ratio" in rep.summary()


def test_dsr_matches_metrics_path(skilled):
    rep = bo.analyze(skilled)
    assert rep.deflated_sharpe == pytest.approx(
        dsr_of_column(skilled, rep.best_trial), rel=1e-9
    )


def test_report_validation():
    with pytest.raises(ValueError):
        bo.analyze(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        bo.analyze(np.ones((50, 1)))
