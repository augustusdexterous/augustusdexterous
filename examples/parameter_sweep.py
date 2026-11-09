"""Parameter sweep of a moving-average crossover, checked for overfitting.

Builds a (T, N) returns matrix from a grid of (fast, slow) window pairs and
runs the full overfitting report on it. Replace `simulate_prices` with your own
price series (a 1-D array of closes) to analyse a real strategy.
"""

import numpy as np

import backtest_overfitting as bo


def simulate_prices(n=2000, drift=0.0, vol=0.01, seed=0):
    """Geometric random walk. drift=0 means no edge exists in the data."""
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))


def sma(x, w):
    """Simple moving average; NaN until the first full window."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def ma_crossover_returns(prices, fast, slow):
    """Per-period strategy returns: long when fast MA > slow MA, else flat.

    The position decided at time t (using prices up to t) earns the return t -> t+1.
    """
    ret = prices[1:] / prices[:-1] - 1
    f, s = sma(prices, fast), sma(prices, slow)
    pos = (f > s).astype(float)[:-1]  # NaN comparisons are False -> flat
    return pos * ret


if __name__ == "__main__":
    prices = simulate_prices()

    columns, labels = [], []
    for fast in range(5, 60, 5):
        for slow in range(60, 250, 10):
            columns.append(ma_crossover_returns(prices, fast, slow))
            labels.append((fast, slow))
    returns = np.column_stack(columns)

    print(f"{returns.shape[1]} parameter combinations, {returns.shape[0]} periods\n")
    report = bo.analyze(returns, periods_per_year=252)
    print(report.summary())
    print(f"\nbest in-sample (fast, slow): {labels[report.best_trial]}")
