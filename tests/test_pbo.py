"""Tests for backtest_overfitting.pbo (fixed seeds throughout)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtest_overfitting.pbo import _column_sharpe, probability_of_backtest_overfitting


def test_result_structure_and_shapes() -> None:
    rng = np.random.default_rng(7)
    trials = rng.normal(0.0, 0.01, size=(128, 6))
    result = probability_of_backtest_overfitting(trials, n_splits=8)

    assert set(result) == {"pbo", "logits", "rank_oos"}
    assert result["logits"].shape == (math.comb(8, 4),)
    assert result["rank_oos"].shape == result["logits"].shape
    assert np.all((result["rank_oos"] >= 1) & (result["rank_oos"] <= 6))
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["pbo"] == pytest.approx(float(np.mean(result["logits"] <= 0.0)))


def test_pure_noise_matrix_has_high_pbo() -> None:
    rng = np.random.default_rng(2024)
    trials = rng.normal(0.0, 0.01, size=(512, 30))
    result = probability_of_backtest_overfitting(trials, n_splits=16)
    assert result["pbo"] > 0.3


def test_true_drift_column_gives_low_pbo() -> None:
    rng = np.random.default_rng(2024)
    trials = rng.normal(0.0, 0.01, size=(512, 30))
    trials[:, 0] += 0.003  # column 0 carries a genuine positive drift
    result = probability_of_backtest_overfitting(trials, n_splits=16)
    assert result["pbo"] < 0.2
    # The genuine winner should rank at or near the top out of sample.
    assert np.median(result["rank_oos"]) == 30


def test_zero_volatility_winner_is_not_discarded() -> None:
    rng = np.random.default_rng(3)
    trials = rng.normal(0.0, 0.01, size=(64, 5))
    trials[:, 0] = 0.001  # constant positive return: always the IS best
    result = probability_of_backtest_overfitting(trials, n_splits=8)
    assert np.all(result["rank_oos"] == 5)
    assert result["pbo"] == 0.0


def test_is_deterministic_for_same_input() -> None:
    rng = np.random.default_rng(99)
    trials = rng.normal(0.0, 0.01, size=(96, 4))
    first = probability_of_backtest_overfitting(trials, n_splits=8)
    second = probability_of_backtest_overfitting(trials, n_splits=8)
    assert first["pbo"] == second["pbo"]
    np.testing.assert_array_equal(first["logits"], second["logits"])


@pytest.mark.parametrize("n_splits", [3, 0, -2])
def test_rejects_odd_or_non_positive_n_splits(n_splits: int) -> None:
    trials = np.random.default_rng(0).normal(size=(40, 3))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(trials, n_splits=n_splits)


def test_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        probability_of_backtest_overfitting(np.zeros(40), n_splits=4)


def test_rejects_single_trial_column() -> None:
    with pytest.raises(ValueError, match="2 columns"):
        probability_of_backtest_overfitting(np.zeros((40, 1)), n_splits=4)


def test_rejects_too_few_rows() -> None:
    with pytest.raises(ValueError, match="rows"):
        probability_of_backtest_overfitting(np.zeros((3, 4)), n_splits=4)


def test_rejects_non_finite_values() -> None:
    trials = np.random.default_rng(0).normal(size=(40, 3))
    trials[5, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        probability_of_backtest_overfitting(trials, n_splits=4)


def test_rejects_subsets_with_fewer_than_two_rows() -> None:
    # 3 rows >= n_splits=2, but the 1-row blocks leave subsets with a single row.
    trials = np.random.default_rng(0).normal(size=(3, 4))
    with pytest.raises(ValueError, match="subset needs at least 2 rows"):
        probability_of_backtest_overfitting(trials, n_splits=2)


def test_rejects_non_integer_n_splits() -> None:
    trials = np.random.default_rng(0).normal(size=(40, 3))
    with pytest.raises(TypeError):
        probability_of_backtest_overfitting(trials, n_splits=4.0)


# --- zero-volatility columns ------------------------------------------------


@pytest.mark.parametrize("rows", [8, 12, 33, 64, 1000])
@pytest.mark.parametrize("value", [0.0015, 0.001, 0.003, 0.059])
def test_constant_positive_column_ranks_plus_inf_for_any_length(
    rows: int, value: float
) -> None:
    # The exact std > 0 test gave ratios near 1e15 for some lengths and layouts.
    data = np.full((rows, 3), value)
    np.testing.assert_array_equal(_column_sharpe(data), np.full(3, np.inf))


def test_constant_negative_and_zero_columns_rank_minus_inf_and_zero() -> None:
    data = np.column_stack([np.full(40, -0.002), np.zeros(40), np.full(40, 0.002)])
    np.testing.assert_array_equal(_column_sharpe(data), [-np.inf, 0.0, np.inf])


def test_varying_column_keeps_its_finite_sharpe() -> None:
    data = np.random.default_rng(8).normal(0.001, 0.01, size=(50, 1))
    ratio = _column_sharpe(data)
    assert ratio[0] == pytest.approx(data.mean() / data.std(ddof=1), rel=1e-12)


def test_constant_trials_give_same_pbo_across_row_counts() -> None:
    # Float noise used to flip the PBO between 1.0 (8 rows) and 0.0 (12+ rows).
    results: set[float] = set()
    for rows in (8, 12, 16, 24, 40):
        trials = np.column_stack([np.full(rows, 0.0015), np.full(rows, 0.001)])
        results.add(probability_of_backtest_overfitting(trials, n_splits=4)["pbo"])
    assert len(results) == 1
