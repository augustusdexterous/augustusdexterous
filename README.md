# backtest-overfitting

[![CI](https://github.com/augustusdexterous/augustusdexterous/actions/workflows/ci.yml/badge.svg)](https://github.com/augustusdexterous/augustusdexterous/actions/workflows/ci.yml) ![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

Tools for detecting overfitting in strategy backtests. You feed in the results of a parameter search, a **returns matrix** of shape `(T, N)` (rows = time periods, columns = parameter trials), and get back:

| Tool | Function | Question it answers |
|---|---|---|
| Deflated Sharpe Ratio | `deflated_sharpe_ratio` | Is the best Sharpe still significant after accounting for the number of trials and non-normal returns? |
| Probability of Backtest Overfitting | `pbo_cscv` | How often does the in-sample winner fall below the median out-of-sample? |
| Multiple-testing correction | `multiple_testing`, `adjust_pvalues`, `haircut_sharpe` | Which trials survive Bonferroni / Holm / BH / BY? |
| Walk-forward CV | `WalkForward` + `cv_select` | What is the out-of-sample result of "pick the best on past data"? |
| Purged k-fold CV | `PurgedKFold` + `cv_select` | The same, with no label leakage (purge + embargo) |

All Sharpe ratios are **per period** internally. `annualize_sharpe` and the report print annualized values.

## Installation

```bash
git clone https://github.com/augustusdexterous/augustusdexterous.git
cd augustusdexterous
git checkout claude/simdi-zsmku5
pip install -e ".[dev]"        # or: pip install -r requirements-dev.txt
python -m pytest -q
```

Dependencies: `numpy`, `scipy` (`pytest` for tests). Python ≥ 3.9.

## Usage

Full example (moving-average crossover grid on simulated prices): `python examples/parameter_sweep.py`
Real-data example (S&P 500, 1999–2018): `python examples/sp500_real_data.py` (needs `pip install arch`)

```python
import numpy as np
import backtest_overfitting as bo

# returns: (T, N) — e.g. 1000 days x 100 parameter combinations
# (a pandas DataFrame also works; anything with .to_numpy() is accepted)
returns = bo.simulate_trials(n_obs=1000, n_trials=100, seed=1)  # synthetic pure noise

report = bo.analyze(returns, periods_per_year=252)
print(report.summary())
```

```
observations x trials      : 1000 x 100
best trial (IS)            : #9  SR=1.11 (ann.)
expected max SR under null : 1.32 (ann.)
Deflated Sharpe Ratio      : 0.339
PBO (CSCV, S=10)           : 0.611
P(OOS loss | IS-best)      : 0.655
significant (holm, a=0.05) : 0/100
walk-forward OOS SR        : 0.82 (ann.), efficiency=0.48
purged k-fold OOS SR       : -0.56 (ann.), efficiency=-0.47
```

The best trial shows an annualized Sharpe of 1.11, but that is below what you'd expect from 100 random trials (1.32). DSR is low, PBO is high, and no trial survives the correction, so the search found noise. When one genuinely skilled trial is added (`n_skilled=1, true_sharpe_annual=3.0`) the results are DSR ≈ 1.00, PBO = 0.00, and 1/20 trials significant.

### Using the tools individually

```python
dsr = bo.deflated_sharpe_ratio(returns)               # best column by default
dsr = bo.deflated_sharpe_ratio(returns, n_trials=500) # if more configurations were actually tried
dsr.dsr, dsr.sr_benchmark, dsr.is_significant(0.05)

pbo = bo.pbo_cscv(returns, n_splits=16)               # C(16, 8) = 12,870 combinations
pbo = bo.pbo_cscv(returns, n_splits=16, max_combinations=2000, random_state=0)
pbo = bo.pbo_cscv(returns, metric=lambda m: m.mean(axis=0))   # custom metric
pbo.pbo, pbo.logits, pbo.degradation_slope, pbo.prob_oos_loss

mt = bo.multiple_testing(returns, method="bh", alpha=0.05)    # bonferroni | holm | bh | by
mt.adjusted_pvalues, mt.reject
bo.adjust_pvalues([0.01, 0.04, 0.03], "holm")
bo.haircut_sharpe(sr=0.1, n_obs=1000, n_trials=100)          # per-period SR

wf = bo.cv_select(returns, bo.WalkForward(train_size=500, test_size=100, expanding=True, gap=5))
pk = bo.cv_select(returns, bo.PurgedKFold(n_splits=5, label_horizon=10, embargo=0.01))
wf.oos_sharpe, wf.efficiency, wf.selected, wf.oos_returns

for train_idx, test_idx in bo.PurgedKFold(5, label_horizon=10).split(len(returns)):
    ...  # use it in your own model training as well
```

## Chronological check

Block-based methods (CSCV, purged k-fold) mix data from across the whole sample. `chronological_check` uses one time-ordered split instead: select on the data before the split, judge on the data after it.

```python
c = bo.chronological_check(returns, split=2000, top_k=10)   # row index of the split
print(c.summary())
```

It reports the in-sample best trial's out-of-sample percentile, the Spearman correlation of in-sample and out-of-sample Sharpe across all trials, and the out-of-sample Sharpe of the in-sample top-k versus all trials.

## Real-data example: S&P 500, 1999–2018

`python examples/sp500_real_data.py` (needs `pip install arch`, which ships the adjusted-close series; no network access needed). It sweeps 209 moving-average crossover parameter pairs on 5,030 daily returns.

```
== 1. Full-sample report ==
Deflated Sharpe Ratio      : 0.902
PBO (CSCV, S=10)           : 0.349
significant (holm, a=0.05) : 0/209
walk-forward OOS SR        : 0.53 (ann.), efficiency=0.80
purged k-fold OOS SR       : 0.42 (ann.), efficiency=0.66

== 2. Chronological checks (annualized Sharpe, out-of-sample) ==
 split  rank rho  IS-best pct  IS top-10  all trials  buy&hold
  2003     +0.49          84%      +0.62       +0.58     +0.45
  2006     +0.36          71%      +0.57       +0.57     +0.38
  2009     -0.13          42%      +0.62       +0.68     +0.70
  2012     +0.52          54%      +0.80       +0.75     +0.83
  2015     +0.18          40%      +0.41       +0.37     +0.43
```

What this shows: the full-sample statistics look reasonable (DSR 0.90, PBO 0.35), but the chronological checks are mixed. The rank correlation is positive in four of five splits and negative in one (2009). Top-10 selections beat buy & hold out of sample only in 2003 and 2006. The 2009 split is the clearest failure: the in-sample winners fell to the 42nd out-of-sample percentile. Full-sample statistics alone would not have shown that. Costs, slippage and shorting are not modelled.

## Calibration

Measured with `simulate_trials` (pure noise unless stated; counts are Monte Carlo replications):

| Check | Setting | Result | Nominal / note |
|---|---|---|---|
| Holm familywise false positive rate | null, T=1000, N=100, 3000 reps | 0.056 | 0.05 (±0.004 MC error) |
| BH any-rejection rate | null, T=1000, N=100, 3000 reps | 0.056 | — |
| Sharpe p-values, share below 0.05 | null, T=1000, N=100, 3000 reps | 0.050 | 0.05 |
| DSR > 0.95 false positive rate | null, T=1000, N=100, 300 reps | 0.000 | ≤ 0.05 (conservative) |
| DSR > 0.95, skilled trial, annual SR 1 | T=1000, N=100, 1000 reps | 0.008 | low power |
| DSR > 0.95, skilled trial, annual SR 2 | T=1000, N=100, 1000 reps | 0.342 | low power |
| DSR > 0.95, skilled trial, annual SR 3 | T=1500, N=50, 100 reps | 1.000 | — |
| PBO under null / skilled (SR 1.5) | T=1000, 50 trials, S=8, 100 reps | 0.48 / 0.27 | null ≈ 0.5 |

The false positive controls are close to nominal: Holm and BH sit at 0.056 against 0.05, within Monte Carlo error, and DSR is conservative at 0.000. DSR's power is low for modest skill: at T=1000 a skilled trial with annualized Sharpe 2 is flagged in only 34% of replications, while Sharpe 3 at T=1500 is flagged in all of them. A DSR below 0.95 on short data does not prove there is no edge.

## Interpreting the metrics

**Deflated Sharpe Ratio (DSR).** The probability that the observed best Sharpe is greater than the expected *maximum* Sharpe from `N` random trials. It corrects for skewness, kurtosis and sample length.
- `> 0.95`: significant at the 5% level after deflation.
- `≈ 0.5`: the result is no better than the best you'd get by chance.
- `n_trials` must be the **number of configurations actually tried**. If you only include the good ones in the matrix, the DSR will be inflated. With highly correlated trials you can pass a smaller effective trial count.

**PBO (CSCV).** The data is split into `S` blocks. For every combination of S/2 blocks (in-sample, IS), the IS winner's rank in the remaining blocks (out-of-sample, OOS) is computed. PBO is the fraction of combinations in which the winner fell to or below the OOS median.
- `< 0.1`: low overfitting risk. `0.1–0.5`: be cautious. `> 0.5`: the selection procedure is worse than random.
- `degradation_slope < 0`: IS success predicts OOS failure.
- `prob_oos_loss`: how often the IS winner lost money OOS.
- `S` should be even. Larger values give more combinations but shorter blocks. 10–16 is typical.

**Multiple-testing correction.** Each trial gets a one-sided p-value for `H0: SR ≤ 0` (= 1 − PSR).
- Bonferroni and Holm control the family-wise error rate (FWER, strict).
- BH controls the false discovery rate (FDR, more lenient). BY is valid under any dependence.
- `haircut_sharpe` gives the Sharpe ratio implied by the corrected p-value (Harvey & Liu).

**Walk-forward / purged k-fold (`cv_select`).** On each fold the best trial is chosen on train and scored on test.
- `oos_sharpe`: Sharpe of the concatenated OOS series. This is your realistic expectation.
- `efficiency = mean(OOS score) / mean(IS score)`. Near 1 is good. Below 0.5, or negative, means you are fitting noise.
- `label_horizon`: if labels/returns depend on future `h` periods (e.g. holding period), the training rows that overlap the test block are dropped (purge). `embargo` drops an extra margin after the test block.

**Rules of thumb.** Low DSR, high PBO, low efficiency → don't trust the result. Look at all of them together; a single metric isn't enough.

## Module layout

```
backtest_overfitting/
  sharpe.py            Sharpe, PSR, expected max SR, DSR, MinTRL
  pbo.py               PBO via CSCV
  multiple_testing.py  Bonferroni, Holm, BH, BY, Sharpe p-values, haircut
  cv.py                WalkForward, PurgedKFold, cv_select
  report.py            analyze() -> OverfittingReport
  consistency.py       chronological_check() — select before a split, judge after it
  data.py              simulate_trials() — synthetic data
tests/test_kit.py      pytest tests
```

## References

- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*.
- Bailey, Borwein, López de Prado & Zhu (2016), *The Probability of Backtest Overfitting*.
- Harvey & Liu (2015), *Backtesting*.
- López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7.
