"""Tests for backtest_overfitting_a.metrics (fixed seeds throughout)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from backtest_overfitting_a.metrics import (
    EULER_MASCHERONI,
    deannualize_sharpe,
    deflated_sharpe_from_returns,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def _exact_mean_std_series(n: int, mean: float, std: float, seed: int) -> np.ndarray:
    """Return a series whose sample mean and sample std (ddof=1) are exactly as requested."""
    z = np.random.default_rng(seed).standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    return mean + std * z


# --- sharpe_ratio -----------------------------------------------------------


def test_sharpe_ratio_constant_mean_series_matches_closed_form() -> None:
    r = _exact_mean_std_series(500, mean=0.001, std=0.01, seed=1)
    assert r.mean() == pytest.approx(0.001, abs=1e-15)
    assert r.std(ddof=1) == pytest.approx(0.01, abs=1e-15)
    expected = math.sqrt(252) * 0.001 / 0.01
    assert sharpe_ratio(r) == pytest.approx(expected, rel=1e-10)


def test_sharpe_ratio_respects_periods_and_risk_free_rate() -> None:
    r = _exact_mean_std_series(300, mean=0.002, std=0.02, seed=2)
    rf = 0.0005
    expected = math.sqrt(12) * (0.002 - rf) / 0.02
    assert sharpe_ratio(r, periods_per_year=12, rf=rf) == pytest.approx(
        expected, rel=1e-10
    )


def test_sharpe_ratio_rejects_zero_volatility() -> None:
    with pytest.raises(ValueError, match="zero volatility"):
        sharpe_ratio([0.01] * 20)


def test_sharpe_ratio_rejects_nan_and_short_input() -> None:
    with pytest.raises(ValueError):
        sharpe_ratio([0.01, np.nan, 0.02])
    with pytest.raises(ValueError):
        sharpe_ratio([0.01])


def test_deannualize_sharpe_matches_per_period_ratio() -> None:
    r = _exact_mean_std_series(400, mean=0.001, std=0.01, seed=3)
    per_period = r.mean() / r.std(ddof=1)
    annual = sharpe_ratio(r, periods_per_year=252)
    assert deannualize_sharpe(annual, periods_per_year=252) == pytest.approx(
        per_period, rel=1e-10
    )


# --- expected_max_sharpe ----------------------------------------------------


def test_expected_max_sharpe_matches_formula() -> None:
    n, var = 100, 0.04
    g = EULER_MASCHERONI
    q = (1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e))
    assert expected_max_sharpe(n, var) == pytest.approx(math.sqrt(var) * q, rel=1e-12)


def test_expected_max_sharpe_increases_with_n_trials() -> None:
    values = [expected_max_sharpe(n, 1.0) for n in (2, 3, 5, 10, 50, 100, 1000)]
    assert all(later > earlier for earlier, later in itertools.pairwise(values))


def test_expected_max_sharpe_scales_with_sqrt_variance() -> None:
    assert expected_max_sharpe(20, 4.0) == pytest.approx(
        2.0 * expected_max_sharpe(20, 1.0), rel=1e-12
    )


def test_expected_max_sharpe_single_trial_is_zero() -> None:
    assert expected_max_sharpe(1, 0.5) == 0.0


def test_expected_max_sharpe_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 1.0)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -1.0)


# --- probabilistic / deflated Sharpe ratio ----------------------------------


def test_psr_is_half_at_its_own_benchmark() -> None:
    assert probabilistic_sharpe_ratio(
        0.12, 250, skew=-0.3, kurt=4.5, sr_benchmark=0.12
    ) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("skew", [0.0, -0.8])
@pytest.mark.parametrize("kurt", [3.0, 6.0])
def test_dsr_is_half_when_sharpe_equals_sr0(skew: float, kurt: float) -> None:
    n_trials, var = 50, 0.01
    sr0 = expected_max_sharpe(n_trials, var)
    dsr = deflated_sharpe_ratio(
        sr0, n_obs=252, skew=skew, kurt=kurt, n_trials=n_trials, var_sharpe_trials=var
    )
    assert dsr == pytest.approx(0.5, abs=1e-12)


def test_dsr_lies_in_unit_interval_over_a_grid() -> None:
    for sr in (-0.1, 0.0, 0.05, 0.1, 0.2):
        for n_trials in (1, 5, 50):
            for skew in (-1.0, 0.0):
                for kurt in (3.0, 6.0):
                    dsr = deflated_sharpe_ratio(sr, 252, skew, kurt, n_trials, 0.01)
                    assert 0.0 < dsr < 1.0, (sr, n_trials, skew, kurt, dsr)


def test_dsr_decreases_as_n_trials_grows() -> None:
    values = [
        deflated_sharpe_ratio(
            0.1, n_obs=500, skew=0.0, kurt=3.0, n_trials=n, var_sharpe_trials=0.01
        )
        for n in (1, 2, 5, 10, 50, 100, 1000)
    ]
    assert all(later < earlier for earlier, later in itertools.pairwise(values))


def test_dsr_from_returns_matches_component_computation() -> None:
    rng = np.random.default_rng(11)
    r = 0.0005 + 0.01 * rng.standard_t(df=5, size=750)
    sr_period = r.mean() / r.std(ddof=1)
    skew = float(stats.skew(r, bias=True))
    kurt = float(stats.kurtosis(r, fisher=False, bias=True))
    expected = deflated_sharpe_ratio(sr_period, r.size, skew, kurt, 20, 0.002)
    assert deflated_sharpe_from_returns(
        r, n_trials=20, var_sharpe_trials=0.002
    ) == pytest.approx(expected, rel=1e-10)


def test_dsr_from_returns_decreases_with_trials() -> None:
    rng = np.random.default_rng(5)
    r = 0.001 + 0.01 * rng.standard_normal(1000)
    values = [
        deflated_sharpe_from_returns(r, n, var_sharpe_trials=0.001)
        for n in (1, 10, 100, 1000)
    ]
    assert all(later < earlier for earlier, later in itertools.pairwise(values))


# --- closed-form checks of the PSR / DSR z-statistic ------------------------


@pytest.mark.parametrize(
    "sharpe, n_obs, skew, kurt, benchmark",
    [
        (0.1, 252, -0.5, 6.0, 0.02),  # skew and kurtosis terms both matter
        (-0.05, 120, 0.8, 4.5, 0.0),  # negative Sharpe, positive skew
        (0.2, 60, 0.3, 2.0, -0.1),  # platykurtic, benchmark below zero
    ],
)
def test_psr_matches_closed_form_with_skew_and_kurtosis(
    sharpe: float, n_obs: int, skew: float, kurt: float, benchmark: float
) -> None:
    # Bailey & Lopez de Prado (2014), eq. 2, written out term by term.
    variance_term = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(variance_term)
    expected = float(stats.norm.cdf(z))
    actual = probabilistic_sharpe_ratio(
        sharpe, n_obs, skew, kurt, sr_benchmark=benchmark
    )
    assert actual == pytest.approx(expected, rel=1e-12)


def test_psr_pinned_value() -> None:
    # Evaluated independently with math.erf: z = 0.08 * sqrt(251) / sqrt(1.0625) ~ 1.2296.
    assert probabilistic_sharpe_ratio(0.1, 252, -0.5, 6.0, 0.02) == pytest.approx(
        0.8905757520873623, abs=1e-12
    )


def test_dsr_matches_closed_form_at_nonzero_z() -> None:
    n_trials, var, n_obs, sharpe, skew, kurt = 20, 0.002, 500, 0.12, -0.4, 5.0
    sr0 = expected_max_sharpe(n_trials, var)
    variance_term = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    z = (sharpe - sr0) * math.sqrt(n_obs - 1) / math.sqrt(variance_term)
    actual = deflated_sharpe_ratio(sharpe, n_obs, skew, kurt, n_trials, var)
    assert actual == pytest.approx(float(stats.norm.cdf(z)), rel=1e-12)


# --- periods_per_year, rf and non-finite inputs -----------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -252.0])
def test_sharpe_ratio_rejects_bad_periods_per_year(bad: float) -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        sharpe_ratio([0.01, 0.02, 0.0, 0.03], periods_per_year=bad)


def test_sharpe_ratio_rejects_non_finite_rf() -> None:
    with pytest.raises(ValueError, match="rf"):
        sharpe_ratio([0.01, 0.02, 0.0, 0.03], rf=float("nan"))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0])
def test_deannualize_sharpe_rejects_bad_periods_per_year(bad: float) -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        deannualize_sharpe(1.0, periods_per_year=bad)


def test_deannualize_sharpe_rejects_non_finite_sharpe() -> None:
    with pytest.raises(ValueError, match="sharpe must be finite"):
        deannualize_sharpe(float("nan"), 252)


# --- large trial counts -----------------------------------------------------


def test_expected_max_sharpe_stays_finite_for_huge_n_trials() -> None:
    # Forming 1 - 1/N rounds to 1.0 here, so the quantile must use the survival form.
    baseline = expected_max_sharpe(10**6, 1.0)
    for n in (10**16, 10**17):
        value = expected_max_sharpe(n, 1.0)
        assert math.isfinite(value)
        assert value > baseline
    assert expected_max_sharpe(10**17, 0.0) == 0.0


def test_dsr_stays_in_unit_interval_for_huge_n_trials() -> None:
    dsr = deflated_sharpe_ratio(0.1, 252, 0.0, 3.0, 10**17, 0.01)
    assert math.isfinite(dsr)
    assert 0.0 <= dsr <= 1.0


# --- deflated_sharpe_from_returns: rf and input checks ----------------------


def test_dsr_from_returns_uses_excess_returns_over_rf() -> None:
    rng = np.random.default_rng(21)
    r = 0.001 + 0.01 * rng.standard_normal(500)
    rf = 0.001
    excess = r - rf
    sr = excess.mean() / excess.std(ddof=1)
    skew = float(stats.skew(excess, bias=True))
    kurt = float(stats.kurtosis(excess, fisher=False, bias=True))
    expected = deflated_sharpe_ratio(sr, r.size, skew, kurt, 10, 0.002)
    actual = deflated_sharpe_from_returns(
        r, n_trials=10, var_sharpe_trials=0.002, rf=rf
    )
    assert actual == pytest.approx(expected, rel=1e-10)
    # A mutant that ignores rf would return this value, which must differ.
    ignoring_rf = deflated_sharpe_from_returns(r, n_trials=10, var_sharpe_trials=0.002)
    assert abs(actual - ignoring_rf) > 1e-3


def test_dsr_from_returns_needs_two_observations() -> None:
    with pytest.raises(ValueError, match="two return observations"):
        deflated_sharpe_from_returns([0.01], n_trials=1, var_sharpe_trials=0.001)
