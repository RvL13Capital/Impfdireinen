# Does `vpts` have a real edge? — an honest validation log

This document records a deliberately adversarial search for **out-of-sample, survivorship-free
predictive edge** in the Volume-Profile system (`vpts`). It is written to be read by a skeptic.
The headline is a negative, and that is the point: the value delivered is a *validated* "no",
plus a reusable harness that judges any future idea honestly.

> **Bottom line.** Across six experiments — a walk-forward backtest and five fitted models, all
> evaluated with purged combinatorial cross-validation and label-shuffle permutation tests — **no
> input studied here produced a robust, statistically-significant out-of-sample edge.** The single
> near-miss (cross-sectional rank, p≈0.10 on 20 names) **vanished when properly powered** (p≈0.86 on
> 88 names), confirming it was a thin-cross-section artifact. On this universe, the binding
> constraint is the *data* (survivorship + a small, liquid survivor universe), not model
> sophistication.

---

## The question

`vpts` generates directional biases from hand-set confluence weights over Volume-Profile, regime
and volume-pattern factors. A single backtest of the breakout style on 2012–2017 large-caps showed
**+14.5%**. The question this log answers is not "is that number positive?" but:

> Is there any **learnable, out-of-sample, survivorship-free** signal in these factors — or is the
> apparent performance drift, compounding, and survivorship?

## Methodology (the harness)

Every claim below clears the same bars, implemented in `vpts.validation` and `vpts.ml` and covered
by 121 unit tests:

- **No look-ahead.** Features at bar *t* use only data ≤ *t*; labels are strictly future. The
  dataset/panel builders are unit-tested for this.
- **Purged + embargoed CPCV** (`CombinatorialPurgedCV`, López de Prado). The timeline is split into
  groups; every combination of test groups is held out; train rows whose label window overlaps a
  test block are **purged**, and a post-block **embargo** breaks serial-correlation leakage. Scores
  are distributions over recombined OOS paths, not a single split.
- **Permutation significance.** The decisive test everywhere is a label shuffle that destroys the
  feature→outcome link while preserving structure (per-row for time-series, **within-date** for the
  cross-section). The p-value is the fraction of shuffles that match or beat the real statistic. An
  effect that cannot clear its own shuffled null is reported as no edge.
- **Honest scope, stated every time.** All data below is **survivorship-biased** (see Data). These
  are validity checks on OOS information content, **not** tradeable results.

## Data

Free, no-API-key, network-restriction-friendly: split/dividend-adjusted daily OHLCV for **88 US
large-caps, 2012–2017**, committed to the public [`stocknet-dataset`](https://github.com/yumoxu/stocknet-dataset)
(`vpts.data` back-adjusts via Adj Close / Close). **Every name is a 2017 survivor** — the dominant,
unavoidable confound throughout. There is no delisted/point-in-time data in this source.

---

## The six experiments

| # | Experiment | OOS statistic | Significance | Verdict |
|---|------------|---------------|--------------|---------|
| 1 | Rule-based backtest, CPCV (8 names, 80 paths, net 5 bps) | **−0.68%/path**, median −1.20%, 36% paths profitable | — | apparent +14.5% was drift/compounding; **no edge** |
| 2 | Learned ridge factor weights (CPCV) | OOS IC **+0.028** | did not beat the hand-set baseline | **no learnable improvement** |
| 3 | Triple-barrier **meta-labeling** | survivors AUC **0.576** (p=0.005) → with delisted injected **0.493** | p **0.801** | **survivorship artifact** |
| 4 | **Enriched** per-name features (momentum/vol/microstructure) | pooled IC **+0.010** (baseline +0.028) | p **0.348** | richer inputs don't help; **no edge** |
| 5 | **Cross-sectional rank**, 20 names | combined OOS IC **+0.021** | p **0.100** | suggestive, **not significant** |
| 6 | **Cross-sectional rank, 88 names (well-powered)** | combined OOS IC **−0.009** | p **0.856** | near-miss **washed out**; **no edge** |

### 1 — The single backtest doesn't survive purged CV
The breakout style's +14.5% (85% of names profitable, single full-period backtest) collapses under
CPCV to **−0.68% per OOS path**, median −1.20%, only 36% of paths profitable. The apparent edge was
bull-market drift and compounding — exactly what rigorous validation is meant to expose.

### 2 — Learning the factor weights doesn't help
A ridge model fit on the four confluence factors (train-only standardization, OOS-scored per CPCV
fold) reaches pooled **OOS IC ≈ +0.028** and does not beat the hand-weighted `bias_score` baseline.
No improvement from learning the weights.

### 3 — Meta-labeling is significant *only because of survivorship*
Predicting whether a primary signal *works* (triple-barrier, volatility-scaled, first-touch) and
filtering on it looked real on survivors: pooled **AUC 0.576, p=0.005**, cost-surviving and
threshold-stable. But injecting synthetic **delisted** names (a vol-elevated decline to pennies)
collapses the pooled permutation test to **AUC 0.493, p=0.801**. A per-name AUC t-test stayed >0.5
only because each decliner got its own model; the realistic single cross-sectional model has no
edge. **Survivorship was the explanation.**

### 4 — Genuinely new per-name features don't rescue it
Adding momentum (20/60/12-1), volatility (σ, ATR/price), volume-trend and distance-to-POC — 11
features through the same harness — yields pooled **IC +0.010**, *below* the 4-factor baseline
(+0.028), at **p=0.348**. Ridge shrank every weight to ≈0. Richer inputs carry no OOS signal here.

### 5 → 6 — Cross-sectional rank: a near-miss that proper power kills
Ranking names against each other each rebalance day (1-month reversal, 12-1 momentum, 60-day vol,
volume-trend) is the standard equity-alpha construction the per-name models never tried. On **20
names** it was the best result of the arc — combined OOS rank IC **+0.021, p=0.100** — but with only
~20 names per date the per-date IC is dominated by noise (σ 0.28). Per-date IC noise scales ~1/√N,
so the decisive test is width: re-run on the **full 88-name** universe (16,873 rows, σ 0.20). The
faint positive **washes out to −0.009, p=0.856** — and the strongest single factor (60-day vol,
+0.045 on 20 names) decays to +0.013. The near-miss was a thin-cross-section artifact, not signal.

---

## Honest conclusion

On 88 survivorship-biased US large-caps (2012–2017, daily), **none** of the studied inputs — the
hand-set rules, learned factor weights, meta-labeling, enriched per-name features, or cross-sectional
ranks — shows a robust, statistically-significant out-of-sample edge. The one apparently-significant
result (meta-labeling) was explained by survivorship, and the one near-miss (cross-sectional rank)
was explained by insufficient statistical power.

**What would actually change this** (in rough order of expected value):

1. **Survivorship-free / point-in-time data**, including delisted names — the dominant confound,
   untestable in this source. This is the real wall, not model complexity.
2. **A wider, deeper cross-section** (hundreds–thousands of names). The 88-name washout suggests
   breadth *within survivors* isn't enough; genuine breadth + delisted names is the test.
3. **Different data regimes** — intraday microstructure, or non-equity assets where Volume-Profile
   structure may carry more information.

Model sophistication is **not** on that list: four straight feature/model variations through the
same purged harness all returned ≈0. That is informative.

## What is durable here

The negative is the finding; the **harness** is the asset. Any new idea now plugs in and is judged
honestly:

- `vpts.validation` — purged + embargoed Combinatorial Purged CV.
- `vpts.ml` — no-look-ahead dataset/panel builders, ridge/logistic models, CPCV evaluators, and
  label-shuffle permutation tests for per-name, meta-labeling, and cross-sectional settings.
- 121 unit tests, including signal-detection *and* null-clearing checks for every evaluator.

## Reproduce

```bash
python examples/github_data_scan.py --plot .          # 1: backtest sweep / regime split
python examples/cpcv_demo.py                          # 1: CPCV on the backtester
python examples/factor_model_demo.py                  # 2: learned factor weights, OOS
python examples/meta_labeling_demo.py                 # 3: triple-barrier meta-labeling
python examples/meta_stress_test.py                   # 3: + survivorship injection
python examples/enriched_factor_demo.py --perms 200   # 4: enriched features + permutation
python examples/cross_sectional_demo.py --perms 200   # 5: cross-sectional rank (20 names)
# 6: well-powered cross-section — pass the full 88-name universe via --tickers
```

## Limitations

Survivorship bias throughout; a single 2012–2017 in-sample period; daily bars only; gross-of-cost
except where noted (meta-labeling tested net of 10 bps); a thin universe by cross-sectional
standards. None of the above is a forward guarantee or financial advice — it is a research log.
