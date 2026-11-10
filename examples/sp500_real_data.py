"""Real-data example: moving-average crossover sweep on S&P 500 daily prices, 1999-2018.

Needs the optional `arch` package (ships the S&P 500 series; no network access):
    pip install arch
    python examples/sp500_real_data.py

Three checks, in order:
1. Full-sample overfitting report (DSR, PBO, multiple testing, CV).
2. Chronological consistency: select on data before a split year, judge on
   data after it, for several split years.
3. Buy & hold as the benchmark a selected parameter set has to beat.
"""

import sys

import numpy as np
from parameter_sweep import ma_crossover_returns

import backtest_overfitting as bo

SPLIT_YEARS = (2003, 2006, 2009, 2012, 2015)


def load_sp500():
    try:
        from arch.data import sp500
    except ImportError:
        sys.exit("This example needs the optional package: pip install arch")
    df = sp500.load()
    return df["Adj Close"].to_numpy(dtype=float), df.index[1:]


if __name__ == "__main__":
    prices, dates = load_sp500()
    print(
        f"S&P 500 adjusted close: {prices.size} days, {dates[0].date()} to {dates[-1].date()}\n"
    )

    columns, labels = [], []
    for fast in range(5, 60, 5):
        for slow in range(60, 250, 10):
            columns.append(ma_crossover_returns(prices, fast, slow))
            labels.append((fast, slow))
    returns = np.column_stack(columns)
    buy_hold = prices[1:] / prices[:-1] - 1

    print("== 1. Full-sample report ==")
    report = bo.analyze(returns, periods_per_year=252)
    print(report.summary())
    print(f"best in-sample (fast, slow): {labels[report.best_trial]}\n")

    print("== 2. Chronological checks (annualized Sharpe, out-of-sample) ==")
    header = f"{'split':>6} {'rank rho':>9} {'IS-best pct':>12} {'IS top-10':>10} {'all trials':>11} {'buy&hold':>9}"
    print(header)
    print("-" * len(header))
    for year in SPLIT_YEARS:
        k = int(np.searchsorted(dates, np.datetime64(f"{year}-01-01")))
        c = bo.chronological_check(returns, split=k, top_k=10)
        bh = bo.sharpe_ratio(buy_hold[k:], periods_per_year=252)
        print(
            f"{year:>6} {c.rank_correlation:>+9.2f} {c.oos_percentile_best:>11.0f}% "
            f"{c.oos_sharpe_top_k:>+10.2f} {c.oos_sharpe_all:>+11.2f} {bh:>+9.2f}"
        )
    print(
        "\nrank rho: correlation of in-sample and out-of-sample Sharpe across all trials."
    )
    print("IS-best pct: out-of-sample percentile of the in-sample best trial.")
