# backtest-overfitting

[![CI](https://github.com/augustusdexterous/augustusdexterous/actions/workflows/ci.yml/badge.svg)](https://github.com/augustusdexterous/augustusdexterous/actions/workflows/ci.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

**Your best backtest is probably the luckiest one you tried.** If you search over 200 parameter sets, the top one looks good even when there is no edge at all. This library tells you how much of a backtest result survives that search, using the statistics from Bailey & López de Prado (deflated Sharpe ratio, probability of backtest overfitting) and the standard multiple-testing corrections.

Give it the returns of every configuration you tried, as a table. Get back a report.

```
$ backtest-overfitting examples/demo_returns.csv
best trial: trial_01
observations x trials      : 504 x 12
best trial (full sample)   : #1  SR=1.06 (ann.)
expected max SR under null : 0.88 (ann.)
Deflated Sharpe Ratio      : 0.598
PBO (CSCV, S=10)           : 0.635  (252 partitions)
significant (holm, a=0.05) : 0/12
chronological split 252    : rank rho=-0.11, IS-best OOS pct=0%, OOS SR best=-1.44 vs all=0.25
```

The best trial has a Sharpe ratio of 1.06, but 12 trials were tried and the expected best of 12 pure-noise strategies is already 0.88. The deflated Sharpe ratio is 0.60, far below the 0.95 needed to call it significant. No trial survives the correction. The best in-sample trial then lands at the bottom of the out-of-sample ranking.

## Install

```bash
pip install git+https://github.com/augustusdexterous/augustusdexterous.git@claude/simdi-zsmku5
```

Requires Python 3.9+, NumPy and SciPy. Tests pass locally on Python 3.9 and 3.13; CI runs 3.9 to 3.13.

## Quick start

Your data is a return matrix of shape `(T, N)`: `T` periods in time order, one column per configuration you tested.

```python
import backtest_overfitting as bo

returns = bo.simulate_trials(
    n_obs=1000, n_trials=100, seed=1
)  # replace with your own (T, N) array or DataFrame
report = bo.analyze(returns, periods_per_year=252)
print(report.summary())
```

From the command line, a CSV with a header row works the same way (non-numeric columns such as dates are ignored):

```bash
backtest-overfitting my_trials.csv --periods-per-year 252 --splits 10
```

Input rules for the CSV: one header row, then one row per period, oldest first. Trial columns must hold decimal returns (0.01 means 1%). The tool skips, with a note on stderr, any column that contains text (such as dates), and any column with a value of 1 or more in magnitude (usually prices or YYYYMMDD dates). A blank or missing value in a trial column is an error, because silently dropping a trial would change the answer. To choose columns yourself, use `--columns trial_01,trial_02`.

## What it measures

| Question | Tool |
|---|---|
| Is the best Sharpe still significant after trying N configurations and accounting for skew and kurtosis? | `deflated_sharpe_ratio`, `deflated_sharpe_from_returns` |
| How often does the in-sample winner end up below the out-of-sample median? | `probability_of_backtest_overfitting` (CSCV) |
| Which trials survive a familywise or FDR correction? | `multiple_testing` (Bonferroni, Holm, BH, BY) |
| How much Sharpe should I haircut from the best trial? | `haircut_sharpe` |
| Does the best in-sample trial still rank well in a later period? | `chronological_check` |
| If I pick the best on a rolling window and trade it, what do I get? | `walk_forward_selection` |
| Clean train/test splits for labels that span several periods | `purged_kfold_split`, `walk_forward_split` |

The core checks (DSR, PBO, multiple testing, chronological check) run together in `analyze()`, which returns an `OverfittingReport`. Walk-forward selection, the splitters and the Sharpe haircut are separate functions.

## Real-data example: S&P 500, 1999–2018

```bash
pip install arch            # ships the S&P 500 adjusted-close series; no network access needed
python examples/sp500_real_data.py
```

The example sweeps 209 moving-average crossover parameter pairs (fast 5–55, slow 60–240) over 5,030 daily returns.

```
== 1. Full-sample report ==
best trial (full sample)   : #204  SR=0.60 (ann.)
expected max SR under null : 0.30 (ann.)
Deflated Sharpe Ratio      : 0.902
PBO (CSCV, S=10)           : 0.349
significant (holm, a=0.05) : 0/209

== 2. Chronological checks (annualized Sharpe, out-of-sample) ==
 split  rank rho  IS-best pct  IS top-10  all trials  buy&hold
  2003     +0.49          84%      +0.62       +0.58     +0.45
  2006     +0.36          71%      +0.57       +0.57     +0.38
  2009     -0.13          42%      +0.62       +0.68     +0.70
  2012     +0.52          54%      +0.80       +0.75     +0.83
  2015     +0.18          40%      +0.41       +0.37     +0.43
```

How to read it:

- The full-sample statistics look reasonable. The DSR of 0.90 is below 0.95, and the PBO of 0.35 is moderate.
- The chronological checks are mixed. The ranking carries over in four of five splits and fails in 2009: the in-sample winners fell to the 42nd out-of-sample percentile.
- Top-10 selections beat buy & hold out of sample only in 2003 and 2006.
- The full-sample statistics alone would not have revealed the 2009 failure. That is why the chronological check exists.
- Costs, slippage and shorting are not modelled, and this is one asset over one period.

## Calibration: does it do what it claims?

These are Monte Carlo checks on `simulate_trials`. Replication counts are given in brackets. A nominal rate is what the theory promises.

| Check | Setting | Measured | Nominal |
|---|---|---|---|
| Holm familywise false positive rate | pure noise, T=1000, N=100 [1500] | 0.057 (±0.006) | 0.05 |
| BH any-rejection rate | pure noise, T=1000, N=100 [1500] | 0.059 | — |
| Sharpe p-values below 0.05 | pure noise, T=1000, N=100 [300] | 0.052 | 0.05 |
| DSR > 0.95 false positive rate, for the selected best trial | pure noise, T=1000, N=100 [1000] | 0.001 | ≤ 0.05 |
| DSR > 0.95, one skilled trial (evaluated directly), annual SR 1 | T=1000, N=100 [300] | 0.003 | — |
| DSR > 0.95, one skilled trial (evaluated directly), annual SR 2 | T=1000, N=100 [300] | 0.31 | — |
| DSR > 0.95, one skilled trial (evaluated directly), annual SR 3 | T=1500, N=50 [300] | 1.00 | — |
| PBO under pure noise / with skill (SR 1.5) | T=1000, N=50, S=8 [100] | 0.48 / 0.27 | 0.5 under noise |

What this means:

- **Error control holds.** Familywise and FDR false positive rates sit at the nominal level within Monte Carlo error. The DSR of the selected best trial fired on pure noise in 1 of 1000 runs, so it is conservative.
- **Power is limited by sample length.** With T=1000 (about four years of daily data), a skilled trial with annualized Sharpe 2 is flagged in 31% of replications. At T=1500 (six years), Sharpe 3 is flagged in all of them. A DSR below 0.95 on short data does not prove there is no edge.
- **PBO behaves as designed.** It is near 0.5 under noise and drops with real skill.

`tests/test_calibration.py` runs smaller versions of three of these checks (Holm, DSR false positives, DSR power) on every push.

## Reading the metrics

**Deflated Sharpe ratio.** The probability that the observed Sharpe beats the expected best Sharpe of `N` pure-noise trials, after correcting for skew, kurtosis and sample length. Above 0.95 is significant after deflation. Near 0.5 means no better than luck.

- `n_trials` must be the number of configurations you actually tried. Leaving out failed ones inflates the DSR.
- With highly correlated trials, the effective number of independent trials is smaller and the DSR is conservative.

**Probability of backtest overfitting (PBO).** The data is cut into `S` contiguous blocks. For each half-split, the in-sample winner is ranked out of sample. PBO is the share of splits where that winner falls at or below the out-of-sample median.

- Below 0.1 is low risk. Above 0.5 means the selection procedure does worse than random.
- Use an even `S`. 8–16 is typical.

**Multiple testing.** Each trial gets a one-sided p-value for `H0: SR ≤ 0`, using the probabilistic Sharpe ratio. Holm and Bonferroni control the chance of any false discovery. BH and BY control the expected share of false discoveries among those reported. BY is valid under any dependence.

**Chronological check.** Select on everything before a split, judge on everything after it.

- A rank correlation near zero, or an in-sample best trial near the out-of-sample median, means the selection did not carry over.
- Check several split points, as in the S&P 500 example. One split can be a coincidence.

## Recipes

```python
import numpy as np
import backtest_overfitting as bo

returns = bo.simulate_trials(n_obs=1000, n_trials=100, seed=1)

# Deflated Sharpe of one chosen trial, deflated by the whole search
dsr = bo.deflated_sharpe_from_returns(
    returns[:, 0],
    n_trials=100,
    var_sharpe_trials=float(
        np.var([bo.sharpe_ratio(c, periods_per_year=1) for c in returns.T], ddof=1)
    ),
)

# PBO with the full result
pbo = bo.probability_of_backtest_overfitting(
    returns, n_splits=16
)  # C(16, 8) = 12,870 partitions
pbo["pbo"], pbo["logits"], pbo["rank_oos"]

# Multiple testing with a choice of correction
mt = bo.multiple_testing(
    returns, method="bh", alpha=0.05
)  # bonferroni | holm | bh | by
mt.adjusted_pvalues, mt.reject

# Walk-forward: select on 500 periods, trade the choice on the next 100, roll forward
wf = bo.walk_forward_selection(returns, train_size=500, test_size=100)
wf.efficiency, wf.oos_sharpe_annual

# Purged k-fold for labels that span several periods (e.g. a 5-day holding period)
t0 = np.arange(len(returns))
t1 = np.minimum(t0 + 5, len(returns) - 1)
for train_idx, test_idx in bo.purged_kfold_split(t0, t1, n_splits=5, embargo_pct=0.01):
    ...
```

## Limitations

- **Your search space is part of the answer.** The tools only know about the trials you pass in. A grid you chose yourself counts as a search, and so does every idea you tried and discarded.
- **Normal approximations.** P-values and the DSR use large-sample approximations. They are reasonable for daily data over years, and less so for a few dozen observations.
- **Stationarity.** Everything assumes the return process is stable enough over the sample. The chronological check is there to test that, not to prove it.
- **Not a backtester.** Inputs are returns. Costs, slippage and execution are your responsibility.

## Module layout

```
backtest_overfitting/
  metrics.py           Sharpe, probabilistic and deflated Sharpe ratios, expected max Sharpe
  pbo.py               Probability of backtest overfitting (CSCV)
  multiple_testing.py  Bonferroni, Holm, BH, BY; Sharpe p-values; Harvey-Liu haircut
  consistency.py       chronological_check: select before a split, judge after it
  selection.py         walk_forward_selection: rolling pick-the-best evaluation
  cv.py                purged_kfold_split, walk_forward_split
  report.py            analyze(): all of the above in one call
  cli.py               backtest-overfitting command
  data.py              simulate_trials: synthetic return matrices
examples/              S&P 500 real-data run, parameter sweep, demo CSV
tests/                 unit, edge-case and Monte Carlo calibration tests
```

## Development

```bash
git clone https://github.com/augustusdexterous/augustusdexterous.git
cd augustusdexterous && git checkout claude/simdi-zsmku5
pip install -e ".[dev,examples]" ruff
python -m pytest -q
ruff check . && ruff format --check .
```

CI runs the tests on Python 3.9 to 3.13, lints with ruff, runs the CLI on the demo CSV, and runs the parameter-sweep example.

## Citation

If this helps your research, see [`CITATION.cff`](CITATION.cff). The methods come from:

- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management*, 40(5), 94–107.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4), 39–69.
- Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13–28.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 7.

## License

MIT. See [`LICENSE`](LICENSE).
