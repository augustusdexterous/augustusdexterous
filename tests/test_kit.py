import numpy as np
import pytest
from scipy import stats

import backtest_overfitting as bo


@pytest.fixture(scope="module")
def noise():
    return bo.simulate_trials(n_obs=1000, n_trials=100, seed=1)


@pytest.fixture(scope="module")
def skilled():
    # one genuinely skilled trial (SR 3 ann.) among 20 noise trials
    return bo.simulate_trials(n_obs=1500, n_trials=20, n_skilled=1,
                              true_sharpe_annual=3.0, seed=2)


# ---------- Sharpe / PSR / DSR ----------

def test_sharpe_matches_manual():
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.01])
    assert bo.sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1))
    m = np.column_stack([r, 2 * r])
    np.testing.assert_allclose(bo.sharpe_ratio(m), [bo.sharpe_ratio(r)] * 2)


def test_psr_normal_case():
    rng = np.random.default_rng(0)
    r = 0.01 * rng.standard_normal(2000) + 0.001
    psr = bo.probabilistic_sharpe_ratio(r)
    sr = bo.sharpe_ratio(r)
    approx = stats.norm.cdf(sr * np.sqrt(len(r) - 1) / np.sqrt(1 + sr**2 / 2))
    assert psr == pytest.approx(approx, abs=0.02)


def test_expected_max_sharpe_monotone():
    vals = [bo.expected_max_sharpe(n, 0.01) for n in (1, 10, 100, 1000)]
    assert vals[0] == 0.0
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_dsr_rejects_pure_noise(noise):
    res = bo.deflated_sharpe_ratio(noise)
    assert res.sharpe > 0  # best of 100 noise trials looks positive...
    assert res.dsr < 0.95  # ...but is not significant after deflation
    assert not res.is_significant()


def test_dsr_accepts_real_skill(skilled):
    res = bo.deflated_sharpe_ratio(skilled)
    assert res.dsr > 0.95


def test_dsr_single_series_requires_inputs():
    r = np.random.default_rng(0).standard_normal(300)
    with pytest.raises(ValueError):
        bo.deflated_sharpe_ratio(r)
    res = bo.deflated_sharpe_ratio(r, n_trials=50, sr_variance=0.003)
    assert 0 <= res.dsr <= 1


def test_min_track_record_length():
    rng = np.random.default_rng(3)
    r = 0.01 * rng.standard_normal(1000) + 0.002
    assert 1 < bo.min_track_record_length(r) < 1000
    assert bo.min_track_record_length(-np.abs(r)) == float("inf")


# ---------- PBO / CSCV ----------

def test_pbo_high_on_noise(noise):
    res = bo.pbo_cscv(noise, n_splits=10)
    assert res.n_combinations == 252
    assert res.pbo > 0.3
    assert res.logits.shape == (252,)


def test_pbo_low_with_skill(skilled):
    res = bo.pbo_cscv(skilled, n_splits=10)
    assert res.pbo < 0.1
    assert np.bincount(res.selected).argmax() == 0


def test_pbo_sampling_and_validation(noise):
    res = bo.pbo_cscv(noise, n_splits=16, max_combinations=200, random_state=0)
    assert res.n_combinations == 200
    with pytest.raises(ValueError):
        bo.pbo_cscv(noise, n_splits=5)
    with pytest.raises(ValueError):
        bo.pbo_cscv(noise[:, :1])


def test_pbo_custom_metric(noise):
    res = bo.pbo_cscv(noise, n_splits=8, metric=lambda m: m.mean(axis=0))
    assert 0 <= res.pbo <= 1


# ---------- multiple testing ----------

def test_adjustments_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(bo.bonferroni(p), [0.04, 0.16, 0.12, 0.02])
    np.testing.assert_allclose(bo.holm(p), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(bo.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    by = bo.adjust_pvalues(p, "by")
    assert np.all(by >= bo.benjamini_hochberg(p))
    with pytest.raises(ValueError):
        bo.adjust_pvalues(p, "nope")


def test_multiple_testing_on_noise_and_skill(noise, skilled):
    assert bo.multiple_testing(noise, "holm").n_significant == 0
    res = bo.multiple_testing(skilled, "bh")
    assert res.reject[0]
    assert res.n_significant <= 2


def test_haircut_sharpe():
    sr = 2.0 / np.sqrt(252)
    h = bo.haircut_sharpe(sr, n_obs=1000, n_trials=100)
    assert 0 <= h < sr
    assert bo.haircut_sharpe(sr, 1000, 1) == pytest.approx(sr)


# ---------- CV splitters ----------

def test_walk_forward_rolling_and_expanding():
    folds = list(bo.WalkForward(100, 20).split(200))
    assert len(folds) == 5
    for tr, te in folds:
        assert tr.size == 100 and te.size == 20 and tr.max() < te.min()
    folds = list(bo.WalkForward(100, 20, expanding=True, gap=5).split(200))
    assert all(tr[0] == 0 for tr, _ in folds)
    assert all(te.min() - tr.max() == 6 for tr, te in folds)


def test_purged_kfold_purge_and_embargo():
    n, h = 100, 5
    for tr, te in bo.PurgedKFold(5, label_horizon=h, embargo=3).split(n):
        assert np.intersect1d(tr, te).size == 0
        t0, tn = te.min(), te.max()
        # no training label window [i, i+h] overlaps the test block
        assert not np.any((tr <= tn + h) & (np.minimum(tr + h, n - 1) >= t0))
        # embargo after test block (plus its label horizon)
        assert not np.any((tr > tn) & (tr <= min(tn + h, n - 1) + 3))
    # without purging, folds partition the data
    folds = list(bo.PurgedKFold(4).split(40))
    assert sum(te.size for _, te in folds) == 40
    assert all(tr.size + te.size == 40 for tr, te in folds)


def test_purged_kfold_explicit_t1():
    t1 = np.minimum(np.arange(50) + 2, 49)
    a = list(bo.PurgedKFold(5).split(50, t1=t1))
    b = list(bo.PurgedKFold(5, label_horizon=2).split(50))
    for (tra, tea), (trb, teb) in zip(a, b):
        np.testing.assert_array_equal(tra, trb)
        np.testing.assert_array_equal(tea, teb)


def test_cv_select_noise_vs_skill(noise, skilled):
    wf = bo.cv_select(noise, bo.WalkForward(500, 100))
    assert wf.n_folds == 5
    assert wf.oos_returns.size == 500
    assert wf.efficiency < 0.5          # IS-best noise decays OOS
    pk = bo.cv_select(skilled, bo.PurgedKFold(5, embargo=0.01))
    assert np.all(pk.selected == 0)
    assert pk.oos_sharpe > 0


# ---------- report / input handling ----------

def test_analyze_report(noise, skilled):
    rep = bo.analyze(noise)
    assert rep.n_trials == 100 and rep.pbo.pbo > 0.3 and rep.dsr.dsr < 0.95
    text = rep.summary()
    assert "Deflated Sharpe Ratio" in text and "PBO" in text
    rep2 = bo.analyze(skilled)
    assert rep2.best_trial == 0 and rep2.dsr.dsr > 0.95 and rep2.pbo.pbo < 0.1


def test_input_validation():
    bad = np.ones((50, 3))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        bo.pbo_cscv(bad, n_splits=4)
    with pytest.raises(ValueError):
        bo.analyze(np.zeros((5, 3)))


def test_accepts_dataframe_like(noise):
    class Frame:
        def __init__(self, a):
            self.a = a

        def to_numpy(self):
            return self.a

    assert bo.pbo_cscv(Frame(noise), n_splits=4).pbo == bo.pbo_cscv(noise, n_splits=4).pbo


# ---------- chronological consistency ----------

def test_chronological_noise_does_not_transfer():
    r = bo.simulate_trials(n_obs=2000, n_trials=100, seed=7)
    c = bo.chronological_check(r, split=1000)
    assert abs(c.rank_correlation) < 0.3
    assert 0 <= c.oos_percentile_best <= 100


def test_chronological_persistent_skill_transfers():
    r = bo.simulate_trials(n_obs=2000, n_trials=20, n_skilled=1,
                           true_sharpe_annual=3.0, seed=2)
    c = bo.chronological_check(r, split=1000)
    assert c.is_best == 0
    assert c.oos_percentile_best >= 90
    assert c.oos_sharpe_best > c.oos_sharpe_all


def test_chronological_validation():
    r = bo.simulate_trials(n_obs=100, n_trials=5, seed=0)
    with pytest.raises(ValueError):
        bo.chronological_check(r, split=1)
    with pytest.raises(ValueError):
        bo.chronological_check(r, split=50, top_k=6)


# ---------- calibration (Monte Carlo) ----------

def test_holm_familywise_error_is_controlled_under_null():
    reps = 300
    hits = sum(
        bo.multiple_testing(bo.simulate_trials(500, 50, seed=s), "holm").n_significant > 0
        for s in range(reps)
    )
    # nominal 0.05; allow ~3.5 standard errors of Monte Carlo noise
    assert hits / reps <= 0.10


def test_dsr_false_positive_rate_under_null():
    reps = 300
    hits = sum(
        bo.deflated_sharpe_ratio(bo.simulate_trials(500, 50, seed=s)).dsr > 0.95
        for s in range(reps)
    )
    assert hits / reps <= 0.05


def test_dsr_detects_strong_skill_with_enough_data():
    reps = 100
    hits = sum(
        bo.deflated_sharpe_ratio(
            bo.simulate_trials(1500, 50, n_skilled=1, true_sharpe_annual=3.0, seed=s),
            selected=0,
        ).dsr > 0.95
        for s in range(reps)
    )
    assert hits / reps >= 0.9
