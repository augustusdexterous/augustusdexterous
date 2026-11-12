# Launch posts (drafts)

Posting is manual. Check each community's rules on self-promotion before posting, and reply to comments yourself.

## Show HN

**Title:** Show HN: backtest-overfitting, how much of your backtest survives the search

**Text:**
If you try 200 parameter sets on a price series, the best one will look good even if the series has no edge. This Python library measures how much of a backtest result survives that search. It implements the deflated Sharpe ratio (Bailey and López de Prado), the probability of backtest overfitting via CSCV, Holm/BH/BY corrections across trials, and a chronological check: pick the winner on data before a date, see whether it still ranks well afterwards.

I ran it on S&P 500 daily data (1999–2018) with a moving-average crossover grid of 209 parameter pairs. The full-sample numbers look fine (deflated Sharpe 0.90, PBO 0.35). The chronological check does not agree: for the split at 2009 the in-sample ranking had a rank correlation of −0.13 with the out-of-sample ranking, and the in-sample winners fell to the 42nd percentile. The full-sample statistics alone would not have shown that. I left the result as it is.

Limits: the tools only know about the trials you pass in, so an unreported search makes the result look better than it is. Costs and execution are not modelled.

Repo: https://github.com/augustusdexterous/augustusdexterous/tree/claude/simdi-zsmku5

## Reddit (r/algotrading or r/quant, check the rules first)

**Title:** Deflated Sharpe, PBO and a chronological check for parameter searches, with an S&P 500 example

**Text:**
I wrote a small Python library that tests whether the best result in a parameter search is likely to be luck. It has the deflated Sharpe ratio, the probability of backtest overfitting (CSCV), multiple-testing corrections, and a chronological check that selects on data before a split and judges on the data after it.

On S&P 500 daily returns from 1999 to 2018, the full-sample statistics for a moving-average crossover grid look reasonable. The chronological check at 2009 does not: the in-sample ranking has no positive relationship with the later ranking, and the in-sample winners fall to the 42nd out-of-sample percentile. I think this kind of check is worth running on any search you do.

Feedback on the statistics is welcome, especially the choice of the chronological split and the calibration numbers in the README.

Repo: https://github.com/augustusdexterous/augustusdexterous/tree/claude/simdi-zsmku5

## Short post (LinkedIn or X)

I put together backtest-overfitting, an open-source Python library that measures how much of a backtest result survives a parameter search (deflated Sharpe ratio, PBO, multiple-testing corrections, chronological checks). On S&P 500 data the full-sample numbers look fine, but the chronological check at 2009 does not: the in-sample winners fell to the 42nd percentile out of sample. Code and README: https://github.com/augustusdexterous/augustusdexterous/tree/claude/simdi-zsmku5

---

# Notes for the maintainer

- This repository is a GitHub profile repository. A dedicated repository for the library (for example `backtest-overfitting`) would collect stars better, and the link above would then point there.
- The test suite has 139 tests. Confirm that CI is green on the branch before posting.
