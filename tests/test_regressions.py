"""Regression tests for defects found in review. Each one failed before the fix."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import backtest_overfitting as bo
from backtest_overfitting.cli import load_trials, main
from backtest_overfitting.cv import purged_kfold_split

REPO = Path(__file__).resolve().parent.parent


def _write(path, rows, header):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ---- annualization (review: reported Sharpe values were not checked) ----


def test_report_annualization_matches_independent_formulas():
    r = bo.simulate_trials(
        n_obs=800, n_trials=30, n_skilled=1, true_sharpe_annual=2.0, seed=11
    )
    rep = bo.analyze(r, periods_per_year=252)
    per_period = np.array(
        [bo.sharpe_ratio(r[:, j], periods_per_year=1) for j in range(30)]
    )
    assert rep.best_trial == int(np.argmax(per_period))
    assert rep.best_sharpe == pytest.approx(
        bo.sharpe_ratio(r[:, rep.best_trial], periods_per_year=252)
    )
    var = float(np.var(per_period, ddof=1))
    assert rep.sr_benchmark == pytest.approx(
        bo.expected_max_sharpe(30, var) * np.sqrt(252)
    )


def test_periods_per_year_must_be_positive():
    r = bo.simulate_trials(n_obs=200, n_trials=5, seed=0)
    for bad in (0, -252, float("nan")):
        with pytest.raises(ValueError):
            bo.analyze(r, periods_per_year=bad)
        with pytest.raises(ValueError):
            bo.chronological_check(r, split=100, periods_per_year=bad)
        with pytest.raises(ValueError):
            bo.walk_forward_selection(
                r, train_size=50, test_size=20, periods_per_year=bad
            )


def test_two_trials_are_enough_for_analyze():
    r = bo.simulate_trials(n_obs=300, n_trials=2, seed=3)
    rep = bo.analyze(r)
    assert rep.n_trials == 2


# ---- zero-volatility handling ----


def test_all_zero_best_column_does_not_crash_and_gets_p_one():
    r = bo.simulate_trials(n_obs=300, n_trials=4, seed=5)
    r[:, 0] = 0.0
    r[:, 1:] = -np.abs(
        r[:, 1:]
    )  # every real trial is negative, so the flat trial is best
    rep = bo.analyze(r)
    assert rep.best_trial == 0
    assert np.isfinite(rep.deflated_sharpe)
    assert bo.sharpe_pvalues(r)[0] == 1.0


def test_constant_nonzero_trial_is_rejected_not_ranked():
    r = bo.simulate_trials(n_obs=300, n_trials=4, seed=6)
    r[:, 2] = 0.001
    with pytest.raises(ValueError, match="constant non-zero"):
        bo.analyze(r)


# ---- multiple testing ----


def test_benjamini_yekutieli_exact_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    c = sum(1 / i for i in range(1, 5))
    np.testing.assert_allclose(
        bo.benjamini_hochberg(p, dependent=True),
        [0.01 * 4 * c / 2, 0.04 * 4 * c / 4, 0.03 * 4 * c / 3, 0.005 * 4 * c / 1],
    )


def test_haircut_exact_values_and_sidak_branch():
    sr, n, n_trials = 0.3, 100, 10
    p = float(stats.norm.sf(sr * np.sqrt(n)))
    expected_bonf = stats.norm.isf(p * n_trials) / np.sqrt(n)
    expected_sidak = stats.norm.isf(1 - (1 - p) ** n_trials) / np.sqrt(n)
    hb = bo.haircut_sharpe(sr, n, n_trials, "bonferroni")
    hs = bo.haircut_sharpe(sr, n, n_trials, "sidak")
    assert hb == pytest.approx(expected_bonf, rel=1e-9)
    assert hs == pytest.approx(expected_sidak, rel=1e-9)
    assert (
        hs >= hb
    )  # Sidak's adjusted p-value is never larger, so its haircut is never smaller


def test_haircut_stays_finite_for_very_strong_signals():
    for method in ("bonferroni", "sidak"):
        h = bo.haircut_sharpe(5.0, 100, 10, method)
        assert np.isfinite(h) and h > 0


# ---- CLI input rules ----


def test_yyyymmdd_date_column_is_skipped_with_a_note(tmp_path, capsys):
    p = tmp_path / "t.csv"
    rng = np.random.default_rng(0)
    rows = [
        [f"2020{m:02d}{d:02d}"] + [f"{x:.6f}" for x in rng.normal(0, 0.01, 3)]
        for m in (1, 2)
        for d in range(1, 26)
    ]
    _write(p, rows, ["date", "a", "b", "c"])
    names, matrix, skipped = load_trials(p)
    assert names == ["a", "b", "c"]
    assert matrix.shape == (50, 3)
    assert any(s.startswith("date") for s in skipped)


def test_price_column_is_skipped(tmp_path):
    p = tmp_path / "t.csv"
    _write(
        p,
        [
            ["100.5", "0.01", "-0.02"],
            ["101.2", "0.00", "0.03"],
            ["99.8", "0.02", "0.01"],
        ],
        ["close", "a", "b"],
    )
    names, _, skipped = load_trials(p)
    assert names == ["a", "b"]
    assert skipped and "close" in skipped[0]


def test_blank_cell_is_an_error_not_a_silent_drop(tmp_path, capsys):
    p = tmp_path / "t.csv"
    _write(p, [["0.01", "0.02"], ["", "0.03"], ["0.00", "0.01"]], ["a", "b"])
    assert main([str(p)]) == 2
    assert "missing value at data row 2" in capsys.readouterr().err


def test_short_row_is_an_error(tmp_path, capsys):
    p = tmp_path / "t.csv"
    _write(p, [["0.01", "0.02"], ["0.00"], ["0.01", "0.01"]], ["a", "b"])
    assert main([str(p)]) == 2
    assert "missing value" in capsys.readouterr().err


def test_columns_option_selects_and_validates(tmp_path, capsys):
    p = tmp_path / "t.csv"
    _write(
        p,
        [["0.01", "0.02", "0.03"], ["0.00", "0.03", "0.01"], ["0.02", "-0.01", "0.00"]],
        ["a", "b", "c"],
    )
    names, matrix, _ = load_trials(p, columns=["a", "c"])
    assert names == ["a", "c"] and matrix.shape == (3, 2)
    assert main([str(p), "--columns", "a,zzz"]) == 2
    assert "not found" in capsys.readouterr().err


def test_demo_csv_runs_through_module_entry_point():
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "backtest_overfitting",
            str(REPO / "examples" / "demo_returns.csv"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "best trial: trial_01" in out.stdout
    assert "skipped column: date" in out.stderr


# ---- splitter fold counts ----


def test_purged_kfold_yields_exactly_n_splits_folds():
    t0 = np.arange(100)
    folds = list(purged_kfold_split(t0, t0 + 3, n_splits=5, embargo_pct=0.05))
    assert len(folds) == 5
    assert sum(te.size for _, te in folds) == 100


# ---- metrics raises ----


def test_deflated_sharpe_from_returns_input_errors():
    good = np.random.default_rng(0).normal(0.001, 0.01, 100)
    with pytest.raises(ValueError):
        bo.deflated_sharpe_from_returns(good[:1], n_trials=5, var_sharpe_trials=0.01)
    with pytest.raises(ValueError):
        bo.deflated_sharpe_from_returns(
            np.r_[good[:-1], np.nan], n_trials=5, var_sharpe_trials=0.01
        )
    with pytest.raises(ValueError):
        bo.deflated_sharpe_from_returns(
            np.full(50, 0.01), n_trials=5, var_sharpe_trials=0.01
        )
    with pytest.raises(ValueError):
        bo.deflated_sharpe_from_returns(good, n_trials=0, var_sharpe_trials=0.01)
    with pytest.raises(ValueError):
        bo.deflated_sharpe_from_returns(good, n_trials=5, var_sharpe_trials=-1.0)
    with pytest.raises(ValueError):
        bo.deflated_sharpe_from_returns(
            good, n_trials=5, var_sharpe_trials=0.01, rf=float("inf")
        )
