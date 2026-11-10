# Changelog

## 0.2.0

- Merged the two parallel implementations into one package, `backtest_overfitting`.
- Added `analyze()` (one-call report), `OverfittingReport`, and the `backtest-overfitting` command line tool.
- Added multiple-testing corrections (Bonferroni, Holm, BH, BY), Sharpe p-values and the Harvey-Liu haircut.
- Added `chronological_check()` for selection carried across a time split, and `walk_forward_selection()`.
- Added Monte Carlo calibration tests, a real-data example on S&P 500 (1999–2018) and a demo CSV.
- Fixed Python 3.9 compatibility in the test suite.

## 0.1.0

- Initial deflated Sharpe ratio, CSCV probability of backtest overfitting and purged k-fold splitters.
