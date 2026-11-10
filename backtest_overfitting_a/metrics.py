r"""Sharpe-ratio statistics that correct for selection bias and non-normality.

Formulas follow Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality",
Journal of Portfolio Management 40(5), 94-107.

Units
-----
``probabilistic_sharpe_ratio`` and ``deflated_sharpe_ratio`` operate on the
*per-period* (non-annualized) Sharpe ratio, and the cross-trial variance
``var_sharpe`` must be expressed in the same per-period units.

* To convert an annualized Sharpe ratio, use :func:`deannualize_sharpe`, which
  divides by :math:`\sqrt{q}`.
* To convert a cross-trial variance of annualized Sharpe ratios, divide it by
  :math:`q` (``periods_per_year``). Variance scales with :math:`q`, not with
  :math:`\sqrt{q}`, so :func:`deannualize_sharpe` must not be applied to it.
* :func:`deflated_sharpe_from_returns` estimates the moments from a return
  series but does not convert ``var_sharpe_trials``. Pass the per-period
  variance to it.
"""

from __future__ import annotations

import math
import operator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = [
    "EULER_MASCHERONI",
    "deannualize_sharpe",
    "deflated_sharpe_from_returns",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
]

EULER_MASCHERONI: float = 0.5772156649015329

# Relative tolerance used to detect zero volatility in floating point.
_ZERO_VOL_RTOL: float = 1e-12


def _as_1d_float(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert to a 1-D float64 array and reject NaN or infinite entries."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _sharpe_per_period(returns: NDArray[np.float64], rf: float) -> float:
    r"""Non-annualized Sharpe ratio: ``mean(r - rf) / std(r - rf, ddof=1)``."""
    if not math.isfinite(rf):
        raise ValueError("rf must be finite")
    excess = returns - rf
    if excess.size < 2:
        raise ValueError("at least two return observations are required")
    vol = float(excess.std(ddof=1))
    scale = float(np.max(np.abs(excess)))
    if vol <= _ZERO_VOL_RTOL * scale:
        raise ValueError("returns have zero volatility; the Sharpe ratio is undefined")
    return float(excess.mean()) / vol


def _validate_periods_per_year(periods_per_year: float) -> float:
    """Require a finite, strictly positive number of periods per year."""
    if not (math.isfinite(periods_per_year) and periods_per_year > 0):
        raise ValueError("periods_per_year must be a finite, positive number")
    return float(periods_per_year)


def sharpe_ratio(
    returns: ArrayLike,
    periods_per_year: float = 252,
    rf: float = 0.0,
) -> float:
    r"""Annualized Sharpe ratio of a return series.

    .. math::

        SR_{ann} = \sqrt{q}\,\frac{\overline{r - r_f}}{s_{r - r_f}}

    where :math:`q` is ``periods_per_year`` and :math:`s` is the sample standard
    deviation (``ddof=1``).

    Parameters
    ----------
    returns:
        Per-period returns (simple returns, not percent).
    periods_per_year:
        Observations per year, e.g. 252 for daily, 12 for monthly data.
    rf:
        Per-period risk-free rate, in the same units as ``returns``.

    Returns
    -------
    float
        The annualized Sharpe ratio.

    Raises
    ------
    ValueError
        If there are fewer than two observations, non-finite values in
        ``returns``, zero volatility, a non-finite ``rf``, or a
        ``periods_per_year`` that is not finite and positive.
    """
    ppy = _validate_periods_per_year(periods_per_year)
    r = _as_1d_float(returns, "returns")
    return float(math.sqrt(ppy) * _sharpe_per_period(r, rf))


def deannualize_sharpe(sharpe: float, periods_per_year: float = 252) -> float:
    r"""Convert an annualized Sharpe ratio to its per-period equivalent.

    .. math::

        SR_{period} = \frac{SR_{ann}}{\sqrt{q}}

    Do not apply this to a variance. A cross-trial variance scales with ``q``,
    so divide it by ``q`` instead.

    Raises
    ------
    ValueError
        If ``sharpe`` is not finite, or ``periods_per_year`` is not finite and
        positive.
    """
    ppy = _validate_periods_per_year(periods_per_year)
    if not math.isfinite(sharpe):
        raise ValueError("sharpe must be finite")
    return float(sharpe) / math.sqrt(ppy)


def _validate_n_trials(n_trials: int) -> int:
    n = operator.index(n_trials)
    if n < 1:
        raise ValueError("n_trials must be at least 1")
    return n


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    r"""Expected maximum Sharpe ratio over ``n_trials`` trials under zero true Sharpe.

    This is the benchmark SR0 of Bailey & Lopez de Prado (2014), eq. 6:

    .. math::

        SR_0 = \sqrt{V[\{SR_n\}]}\left[(1-\gamma)\,Z^{-1}\!\left(1-\tfrac{1}{N}\right)
               + \gamma\,Z^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right]

    where :math:`\gamma` is the Euler-Mascheroni constant, :math:`Z^{-1}` the
    standard normal quantile function, and :math:`V[\{SR_n\}]` the variance of
    the Sharpe estimates across trials. For ``n_trials == 1`` there is no
    selection, so 0 is returned (the formula's limit as N -> 1 does not exist).

    Each :math:`Z^{-1}(1-p)` is evaluated as the inverse survival function
    :math:`\text{isf}(p) = -Z^{-1}(p)`. Forming ``1 - p`` first would round to 1.0
    for very large N, where ``ppf`` returns infinity.

    Parameters
    ----------
    n_trials:
        Number of independent trials (strategies or configurations) tried.
    var_sharpe:
        Cross-trial variance of the per-period Sharpe estimates.

    Returns
    -------
    float
        SR0, in the same units as ``var_sharpe``'s square root.

    Raises
    ------
    ValueError
        If ``n_trials`` is less than 1, or ``var_sharpe`` is not a finite,
        non-negative number.
    """
    n = _validate_n_trials(n_trials)
    if not math.isfinite(var_sharpe) or var_sharpe < 0:
        raise ValueError("var_sharpe must be a finite, non-negative number")
    if n == 1:
        return 0.0
    g = EULER_MASCHERONI
    quantile = (1.0 - g) * stats.norm.isf(1.0 / n) + g * stats.norm.isf(
        1.0 / (n * math.e)
    )
    return float(math.sqrt(var_sharpe) * quantile)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurt: float,
    sr_benchmark: float = 0.0,
) -> float:
    r"""Probabilistic Sharpe ratio: P(true per-period SR > ``sr_benchmark``).

    Bailey & Lopez de Prado (2014), eq. 2:

    .. math::

        PSR(SR^*) = \Phi\!\left(\frac{(\widehat{SR}-SR^*)\sqrt{T-1}}
            {\sqrt{1-\hat\gamma_3\,\widehat{SR}+\frac{\hat\gamma_4-1}{4}\,\widehat{SR}^{2}}}\right)

    Parameters
    ----------
    sharpe:
        Observed per-period (non-annualized) Sharpe ratio.
    n_obs:
        Number of return observations T.
    skew:
        Sample skewness :math:`\hat\gamma_3` of the returns.
    kurt:
        Raw sample kurtosis :math:`\hat\gamma_4` (3 for normal returns, not the
        excess kurtosis).
    sr_benchmark:
        Per-period Sharpe benchmark SR*.

    Returns
    -------
    float
        Probability in [0, 1].
    """
    n = operator.index(n_obs)
    if n < 2:
        raise ValueError("n_obs must be at least 2")
    for name, value in (
        ("sharpe", sharpe),
        ("skew", skew),
        ("kurt", kurt),
        ("sr_benchmark", sr_benchmark),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    denom_sq = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if denom_sq <= 0.0:
        raise ValueError(
            "non-positive variance term; check the skew and kurtosis inputs"
        )
    z = (sharpe - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
    var_sharpe_trials: float,
) -> float:
    r"""Deflated Sharpe ratio: the PSR evaluated against the selection benchmark SR0.

    .. math::

        DSR = PSR(SR_0) = \Phi(z), \qquad
        z = \frac{(SR - SR_0)\sqrt{T-1}}{\sqrt{1-\gamma_3\,SR+\frac{\gamma_4-1}{4}SR^2}}

    with :math:`SR_0` from :func:`expected_max_sharpe`. A DSR of 0.5 means the
    observed Sharpe equals the best-of-N null benchmark; values above 0.95 are
    commonly read as significant after deflation.

    Parameters
    ----------
    sharpe:
        Observed per-period (non-annualized) Sharpe ratio of the selected strategy.
    n_obs:
        Number of return observations T.
    skew:
        Sample skewness of the returns.
    kurt:
        Raw sample kurtosis (3 for normal returns).
    n_trials:
        Number of trials from which the strategy was selected.
    var_sharpe_trials:
        Variance across trials of the per-period Sharpe estimates.

    Returns
    -------
    float
        DSR in [0, 1].

    References
    ----------
    Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
    Journal of Portfolio Management, 40(5), 94-107.
    """
    sr0 = expected_max_sharpe(n_trials, var_sharpe_trials)
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurt, sr_benchmark=sr0)


def deflated_sharpe_from_returns(
    returns: ArrayLike,
    n_trials: int,
    var_sharpe_trials: float,
    rf: float = 0.0,
) -> float:
    r"""Deflated Sharpe ratio computed directly from a return series.

    Estimates the per-period Sharpe ratio, the sample skewness (moment
    definition, ``bias=True``) and the raw kurtosis (``fisher=False``) from
    ``returns``, then applies :func:`deflated_sharpe_ratio`.

    Parameters
    ----------
    returns:
        Per-period returns of the selected strategy.
    n_trials:
        Number of trials from which the strategy was selected.
    var_sharpe_trials:
        Variance across trials of the **per-period** Sharpe estimates. If the
        trial Sharpe ratios are annualized, divide their variance by
        ``periods_per_year`` first.
    rf:
        Per-period risk-free rate.

    Returns
    -------
    float
        DSR in [0, 1].

    Raises
    ------
    ValueError
        If ``returns`` has fewer than two observations, contains non-finite
        values, or has zero volatility after subtracting ``rf``. Also raised if
        ``rf`` is not finite, or ``n_trials`` or ``var_sharpe_trials`` is invalid.

    Notes
    -----
    The formula is defined for any T >= 2, but the skewness and kurtosis
    estimates are noisy on short samples. A DSR from a few dozen observations is
    weak evidence. No minimum sample size is enforced.
    """
    r = _as_1d_float(returns, "returns")
    excess = r - rf
    sr = _sharpe_per_period(r, rf)
    skew = float(stats.skew(excess, bias=True))
    kurt = float(stats.kurtosis(excess, fisher=False, bias=True))
    return deflated_sharpe_ratio(sr, r.size, skew, kurt, n_trials, var_sharpe_trials)
