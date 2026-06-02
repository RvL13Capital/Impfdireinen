# Does `vpts` have a real edge? — an honest validation log

> 📄 **[Download this study as a PDF](docs/Quiet-Volume-Research.pdf)**  ·  [Architecture](docs/ARCHITECTURE.md)  ·  [Changelog](CHANGELOG.md)  ·  [README](README.md)

This document records a deliberately adversarial search for **out-of-sample, survivorship-free
predictive edge** in the Volume-Profile system (`vpts`). It is written to be read by a skeptic.
The value delivered is *validated* findings — mostly negatives, one qualified positive — plus a
reusable harness that judges any future idea honestly.

> **Bottom line.** Across thirteen experiments — a walk-forward backtest, eleven fitted models, and a
> feature-orthogonality audit, evaluated with purged combinatorial cross-validation (label-shuffle
> permutation tests where applicable) — no
> input produced a **survivorship-robust, tradeable** out-of-sample edge. The **structural
> microstructure features** (synthetic delta, profile shape, cost-basis migration) produce a real OOS
> correlation (IC ≈ +0.035, p = 0.005) that **survives widening to 88 names**, and — traded as a
> long/short book that goes **flat** in the noisy middle and only bets the conviction tails — is even
> **profitable net of 10 bps on the survivor universe** (+0.26%/bet). But that edge is a
> **survivorship mirage**: carried by the dip-buying features, the conviction-bucket curve **inverts**
> when synthetic delisted names are injected (+0.26%/bet → **−1.07%/bet**) — the patterns that look
> bullish on names that *survived* are what precedes a death-spiral in names that *didn't*. The one
> component that *doesn't* invert is the **meta-labeling selectivity** of a swing setup-rater (which
> entries are higher-R:R). It earned a dedicated stress-test, which found the survivors lift **robust
> across 9/9 parameter settings** and significant (p = 0.023) — but **carried by the same dip-buying
> features** (not regime) and **not significant once delisted names are present** (p = 0.106), so that
> thread is closed too. So: *no survivorship-robust tradeable edge; the binding constraint is the data,
> not the model.*

<p align="center"><img src="docs/img/arc_scorecard.png" width="88%" alt="The ladder to a tradeable edge: none cross the survivorship-robust line."/></p>

---

## The question

`vpts` generates directional biases from hand-set confluence weights over Volume-Profile, regime
and volume-pattern factors. A single backtest of the breakout style on 2012–2017 large-caps showed
**+14.5%**. The question this log answers is not "is that number positive?" but:

> Is there any **learnable, out-of-sample, survivorship-free** signal in these factors — or is the
> apparent performance drift, compounding, and survivorship?

## Methodology (the harness)

Every claim below clears the same bars, implemented in `vpts.validation` and `vpts.ml` and covered
by 144 unit tests:

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

## The thirteen experiments

| # | Experiment | OOS statistic | Significance | Verdict |
|---|------------|---------------|--------------|---------|
| 1 | Rule-based backtest, CPCV (8 names, 80 paths, net 5 bps) | **−0.68%/path**, median −1.20%, 36% paths profitable | — | apparent +14.5% was drift/compounding; **no edge** |
| 2 | Learned ridge factor weights (CPCV) | OOS IC **+0.028** | did not beat the hand-set baseline | **no learnable improvement** |
| 3 | Triple-barrier **meta-labeling** | survivors AUC **0.576** (p=0.005) → with delisted injected **0.493** | p **0.801** | **survivorship artifact** |
| 4 | **Enriched** per-name features (momentum/vol/microstructure) | pooled IC **+0.010** (baseline +0.028) | p **0.348** | richer inputs don't help; **no edge** |
| 5 | **Cross-sectional rank**, 20 names | combined OOS IC **+0.021** | p **0.100** | suggestive, **not significant** |
| 6 | **Cross-sectional rank, 88 names (well-powered)** | combined OOS IC **−0.009** | p **0.856** | near-miss **washed out**; **no edge** |
| 7 | **Structural microstructure** (synthetic delta, shape, VACR-z, decay) | OOS IC **+0.103** (8 names) → **+0.035** (88 names, 1,308 folds) | p **0.005** (both) | **real signal — survives widening** |
| 8 | **Structural + survivorship injection** | pooled IC +0.041 → +0.013 (5 dead, 20%) → +0.001 (9 dead, 31%) | p 0.005 → **0.085** → 0.473 | **survivorship-*sensitive*; graceful decay, not a cliff** |
| 9 | **Structural decomposition + cost** | DIP features carry it (REGIME n.s., p 0.254); tails-only L/S **+0.26%/bet net (survivors) → −1.07%/bet (injected)** — curve inverts | — | **survivorship mirage: the edge inverts off survivors** |
| 10 | **Swing setup-rater (MFE/MAE meta-labeling)** | direction +0.17%→−0.58%/trade (survivorship); selectivity LIFT +0.14%/bet (surv) → +0.09% (injected) | p 0.005 → **0.10** | **selectivity resists inversion but loses significance & stays unprofitable injected** |
| 11 | **Selectivity stress-test** (grid + decomposition + power) | survivors lift positive in **9/9** param cells; carried by **DIP** (+0.08) not REGIME (−0.02); injected lift +0.075% | p 0.023 → **0.106** | **robust but DIP-carried & n.s. injected — thread closed** |
| 12 | **EM-GMM profile decomposition** (parametric, vs a no-GMM VWAP baseline) | best feature `gmm_gravity` OOS IC **+0.090** (survivors); a one-line `vwap_dist` scores **+0.125** | **0.91**-corr w/ `vwap_dist`; in-sample **partial corr 0.016** (ctrl momentum+VWAP) — no signal beyond it | **decomposition adds nothing — its one signal is just price-minus-VWAP** |
| 13 | **Orthogonality purge** (Spearman clustering + per-feature OOS IC, 23 feats) | ~19 clusters (not collinear); only **6/23** clear \|IC\|≥0.05 (top is momentum/VWAP +0.14); ~17 are independent yet ~0 IC | — | **wide but shallow — the matrix isn't redundant, it's mostly null; signal = momentum/VWAP + a thin dip tail** |

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

### 7 → 8 — Structural microstructure: the one signal that survives stress
Transforming the static profile into quantifiable features — **synthetic delta** (Close-Location-Value
× volume, an OHLC order-flow estimate), volume-weighted **skew/kurtosis** and P/b/B/D **shape**,
**value-area-compression z-score**, **POC-migration slope**, **cost-basis migration** (decayed vs
lifetime POC), ledges and poor highs — 13 features through the same harness. This is the first input
to clear the bars:

- **8 survivors:** pooled OOS IC **+0.103**, p **0.005** (the single delta@POC feature alone is
  −0.043; the *combination* predicts).
- **Stress 1 — widening to 88 names:** IC shrinks to **+0.035** but, with 1,308 folds (null σ 0.006),
  is still ≈5.8σ out, **p 0.005**. Unlike the cross-sectional near-miss it **did not wash out** —
  proof it is not a small-sample artifact. Per-name dispersion is sensible: BABA scores **−0.424**
  (a genuine decliner — the dip-features correctly *anti*-predict).
- **Stress 2 — survivorship injection:** adding synthetic decline-to-pennies names degrades the
  signal *gracefully* — +0.041 (0 dead, p 0.005) → +0.029 (1, p 0.015) → +0.013 (5 dead ≈20%,
  **p 0.085, lost**) → +0.001 (9 dead ≈31%, p 0.473). This is **categorically unlike meta-labeling**,
  which collapsed from p 0.005 straight to p 0.80. The structural signal survives *low, realistic*
  large-cap delisting rates (≲10%) but **not** heavy survivorship (≳15–20%).

<p align="center"><img src="docs/img/structural_ic_sweep.png" width="70%" alt="Structural IC decays gracefully under survivorship injection"/></p>

### 9 — Decomposition + cost: the signal is survivorship-leaning and economically empty
Three diagnostics settle what the +0.035 actually is — and the answer is sobering:

- **Per-feature OOS IC** (survivors → +delisted): the signal is carried by the **dip-buying / order-flow**
  features — `cost_basis_migration` (+0.055 → +0.035) and `delta_net` (+0.052 → +0.031) — exactly the
  survivorship-prone ones. The regime feature `vacr_z` is mildly **anti-predictive** (−0.031), so the
  hopeful "regime carries a genuine edge" hypothesis is **falsified**.
- **Subgroup ablation:** the **REGIME** sub-model is **not** significant even on survivors (IC +0.009,
  p 0.254); the **DIP** sub-model is (IC +0.030, p 0.020) but **collapses** under injection (p 0.527).
- **Cost-aware, traded properly:** the naive always-in-market `sign()` book loses (−0.08%/bet) — but
  that forces a short position through half a bull market and is the wrong test. A real book goes
  **long the top signal quintile, short the bottom, and flat the middle 60%** (in the market only ~40%
  of the time). On survivors the conviction-bucket curve **rises monotonically** (+1.08% → +1.54%) and
  the tails-only long/short earns **+0.46%/bet gross, +0.26%/bet net of 10 bps**. So — traded with a
  flat middle — it *is* economically meaningful on the survivor universe.
- **…but it is a survivorship mirage.** Inject the synthetic delisted names and the bucket curve
  **inverts** (−0.23% → −1.09%): the bars the signal flags *most bullish* become the *worst* future
  performers, and the same strategy flips to **−1.07%/bet net**. The dip-buying structural footprint
  that marks a bottom in a name that *recovered* is indistinguishable from the one that marks the next
  leg down in a name that *delisted* — survival is doing the labeling.

So the structural result is **real and even tradeable-looking on survivors, but the apparent edge is
manufactured by survivorship** — it does not merely fade, it reverses sign. The decomposition is the
discipline working: betting the conviction tails turned a dismissive "−0.08%, empty" into a tempting
"+0.26% net," and only the injection test revealed that tempting number to be a survivorship artifact.

<p align="center"><img src="docs/img/survivorship_inversion.png" width="70%" alt="Conviction-bucket forward return inverts under survivorship injection"/></p>

**Phase C — the MFE/MAE re-framing + XGBoost don't rescue it.** Re-labeling each bar by whether a long
bet's *Maximum Favorable Excursion* beat its *Maximum Adverse Excursion* (a volatility-scaled triple
barrier) and learning `P(win)` from the structural features gives, on identical purged-CPCV splits:
a **logistic** OOS AUC of **0.529** (in-sample 0.689; permutation **p = 0.07, not significant**) and
an **XGBoost** that memorizes the training set (in-sample AUC **0.943**) yet scores **0.496 OOS — below
0.5, *worse* than logistic**, a +0.447 over-fitting gap. The nonlinear model adds nothing out of
sample; its gaudy in-sample number is exactly the false-confidence trap rigorous evaluation exists to
catch. Neither the MFE/MAE framing nor gradient boosting turns the curiosity into an edge.

<p align="center"><img src="docs/img/xgboost_overfit.png" width="58%" alt="XGBoost memorizes in-sample (0.943) but is 0.496 out-of-sample"/></p>

### 10 — Swing setup-rater: separating *direction* from *selectivity*
The product goal is concrete: for a **swing** horizon (days–weeks), rate the setup in front of you
0–100 and act only when the risk/reward is favorable — otherwise stay flat. Mechanically this is
meta-labeling with the triple barrier *defining* the R:R (default **2:1**, take-profit 2×vol / stop
1×vol, breakeven win-rate 33%): a logistic rater learns `P(win)` from the structural features, and we
trade only the **best-rated 20%** of long setups (`select_top`), net of 10 bps. Decomposing the result
is what matters:

- **Direction** (take every long signal): **+0.17%/trade** on survivors → **−0.58%/trade** with
  delisted injected. The *decision to be long* is survivorship-dependent — same story as everywhere.
- **Selectivity** (does the rating pick better setups *among* longs?): the expectancy LIFT of the
  best-rated 20% over taking all is **+0.14%/trade on survivors (permutation p = 0.005)**, and — unlike
  the directional bucket curve — it **does not invert** under injection: it stays mildly positive
  (**+0.09%/trade**). But it **loses significance (p = 0.10)** and only 53% of folds beat take-all.

So the rater's *selectivity* is the most survivorship-**resilient** signal found in the whole arc — it
degrades rather than reverses — which fits the meta-labeling thesis (the secondary model filters; it
does not pick direction). Yet it falls short on the two tests that matter: it is **not significant**
once delisted names are present, and even the rated subset stays **unprofitable** on the realistic
universe (−0.49%/trade), because no amount of setup-selection repairs a survivorship-driven direction.
The rater is a clean, usable *interface* (a 0–100 rating + expected R-multiple per setup); on this
data it is not a validated edge.

### 11 — Stress-testing the selectivity: robust, but DIP-carried and not significant injected
The selectivity lift was the one thread that resisted inversion, so it earned a dedicated, adversarial
follow-up — three pre-registered tests, *thread closes unless it passes all three* (31 survivors + 12
synthetic delisted, top-20% rated, 10 bps):

1. **Robustness grid** — vary horizon ∈ {5,10,15}, R:R ∈ {1.5,2,3}:1, selection ∈ {10,20,30}%. The
   survivors lift is **positive in all 9/9 cells** (+0.075% … +0.116%): *not* a lucky parameter pick. ✓
2. **Feature decomposition** — the lift is carried by the **DIP** (dip-buying) subgroup (+0.079% on
   survivors) with **REGIME contributing nothing** (−0.016%). It is the *same survivorship-prone
   feature family* that drove the directional mirage, not a survivorship-agnostic regime signal. ✗
3. **Significance at power** — survivors lift +0.075% is significant (**p = 0.023**), but with delisted
   names injected the same +0.075% lift sits in a wider null and is **not significant (p = 0.106)**. ✗

So the selectivity is **genuinely robust on survivors** yet fails the two tests that decide whether it
is *survivorship-free*: it lives in the dip-buying features, and it cannot clear its shuffled null once
delisted names are present (and, from §10, never makes the realistic universe profitable). By the
pre-stated bar, the thread is **closed** — honestly, on evidence gathered to *disconfirm* it. That the
lift *degrades* (p 0.023 → 0.106) rather than *inverting* (like the direction) is the one durable
nuance: meta-labeling selectivity is the least-survivorship-fragile thing here — just not enough.

<p align="center"><img src="docs/img/selectivity_grid.png" width="72%" alt="Selectivity lift robust on survivors across 9/9 cells but not significant injected"/></p>

---

### 12 — A parametric decomposition (EM-GMM) reduces to price-minus-VWAP
The §7–§11 thread decomposed the profile *heuristically* (smoothed peaks, P/b/B/D shapes). A fair
objection: a *parametric* decomposition might recover hidden structure the smoothing blurs. So I fit a
**1-D Gaussian mixture by weighted EM** to each profile (`vpts.structure.gmm` — pure numpy, BIC
model-selection over k ∈ {1,2,3}) → seven scale-free features (hidden-POC separation, antimode,
fair-value gravity).

A harness lesson first, because it nearly fooled me: the **seven-feature ridge IC is ≈ 0 (+0.02) — but
that is *ridge dilution*, not absence of signal.** Read per-feature, one component carries it all:
**`gmm_gravity`** (signed distance from price to the dominant hidden POC), single-factor OOS IC **+0.090**
on survivors — pooling it with six far-weaker features in one ridge buried it to +0.02. *Read features
individually before trusting a multi-feature score.*

But the signal doesn't earn the machinery. `gmm_gravity` is **0.91-correlated with a one-line `vwap_dist`
= (close − VWAP) / range** — no mixture model at all — and **that baseline scores higher** (OOS IC
**+0.125 vs +0.090**). Adding gravity to `vwap_dist` doesn't lift the IC — it **lowers** it
(**+0.125 → +0.102**, the dilution of a redundant feature); and its **in-sample partial correlation with
forward return, controlling for momentum and VWAP-distance, is +0.016** — zero information beyond "how far
price has extended from its volume-weighted average." Economically it is the same survivorship family:
traded as conviction buckets its long-only return **collapses under delisted injection** (+2.2% → −0.2% /
bet — an inversion), as momentum and VWAP-distance do too. The "hidden-POC gravity" is a moving-average distance in disguise — and the fancier
tool scored **below** the one-liner. §7's lesson, sharpened: feature **content** (trend/extension) is the
axis; decomposition **machinery** is incidental.

---

### 13 — The orthogonality purge: wide but shallow
Experiment 12 was n = 1 — *one* parametric feature collapsing into a VWAP-distance baseline. Does the
*whole* matrix? I pooled all **13 structural + 7 EM-GMM features plus a no-GMM `vwap_dist` / momentum
baseline** (20 names, ~7k rows), took the Spearman rank-correlation matrix, hierarchically clustered it
(distance = 1 − |ρ|), and overlaid each feature's standalone OOS IC.

The result refines *both* the §12 lesson and the obvious objection to it. The matrix is **not a collinear
blob** — 23 features fall into ~19 clusters at |ρ| ≥ 0.7; most features are statistically *independent*,
so the strong claim "everything collapses to momentum/VWAP" is **false**. But independence isn't signal:
**only 6 of 23 features clear |IC| ≥ 0.05** — `vwap_dist` (+0.14), `mom_120` (+0.11) and `gmm_gravity`
(+0.11) in one momentum/VWAP cluster, short-horizon `mom_20` (+0.08), and a thin dip/flow tail
(`cost_basis_migration` +0.06, `delta_net` +0.05). The remaining **~17 — most of the GMM geometry and the
entire profile-shape family (skew, kurtosis, ledges, poor-highs, P/b/B/D one-hots) — are orthogonal yet
carry ≈ 0 IC.**

So the matrix is **wide but shallow**: not redundant, *mostly null*. The honest purge is an **IC filter,
not a correlation filter** — clustering barely shrinks it (the dead features aren't collinear copies,
they're independent noise), but an IC threshold cuts it to a handful: a momentum/VWAP axis plus two dip
features, which §7–§12 already pinned as the same survivorship-prone family. Four experiments of elaborate
profile geometry reduce to *one moving-average distance and two flow features* — none survivorship-robust.
The wall is unmoved; it is just mapped more precisely.

---

## Honest conclusion

On 88 survivorship-biased US large-caps (2012–2017, daily), **none** of the studied inputs yields a
**survivorship-robust** out-of-sample edge. The hand-set rules, learned factor weights, meta-labeling,
enriched per-name features and cross-sectional ranks show no robust signal at all (meta-labeling's was
survivorship; the cross-sectional near-miss was low power). The **structural microstructure features**
go further than anything else: a real OOS correlation (IC ≈ +0.035, p = 0.005) that survives
universe-widening and, traded as a long/short book that stays **flat** in the middle and bets only the
conviction tails, is **profitable net of cost on the survivor universe** (+0.26%/bet). But that edge
is a **survivorship mirage** — carried by the dip-buying features, it does not just fade under delisted
injection, it **inverts**: the conviction-bucket curve flips, the most-bullish-flagged bars become the
worst performers, and the strategy goes from +0.26% to **−1.07%/bet**. The closest thing to a robust
result is the **selectivity** of a swing setup-rater — *which* long entries are higher-R:R, as opposed
to *whether* to be long: its expectancy lift is significant on survivors (p = 0.005) and, uniquely,
**resists inversion** under injection (degrades to +0.09%/bet rather than flipping) — but it loses
significance (p = 0.10) and never makes the realistic universe profitable. Model sophistication is not
the limiting factor (XGBoost over-fit to a sub-0.5 OOS AUC; the linear book did better; a parametric EM-GMM decomposition reduced to a price-minus-VWAP feature a one-line baseline beat, §12) — and neither,
ultimately, is feature content: the **data** is the wall. Conditioning on names that *survived*
manufactures an edge that reverses the moment you stop conditioning on survival; the most resilient
signal (meta-labeling **selectivity**) was pushed hard in a dedicated stress-test — robust across 9/9
parameter settings on survivors, but carried by the same dip-buying features and not significant once
delisted names are present, so it too is closed. Thirteen experiments, one consistent wall.

**What would actually change this** (in rough order of expected value):

1. **Survivorship-free / point-in-time data**, including delisted names — the dominant confound,
   untestable in this source. This is the real wall, not model complexity.
2. **A wider, deeper cross-section** (hundreds–thousands of names). The 88-name washout suggests
   breadth *within survivors* isn't enough; genuine breadth + delisted names is the test.
3. **Different data regimes** — intraday microstructure, or non-equity assets where Volume-Profile
   structure may carry more information.

Model sophistication mostly is **not** the answer — four feature/model variations returned ≈0 — but
the *right kind* of feature (structural microstructure, not momentum/vol/rank) did surface the arc's
one real signal. The lesson: feature *content* mattered where feature *complexity* did not.

## What is durable here

The findings — eight negatives and one qualified positive — are the result; the **harness** is the
asset. Any new idea plugs in and is judged honestly:

- `vpts.validation` — purged + embargoed Combinatorial Purged CV.
- `vpts.ml` — no-look-ahead dataset/panel builders, ridge/logistic models, CPCV evaluators, and
  label-shuffle permutation tests for per-name, meta-labeling, and cross-sectional settings.
- `vpts.structure` — synthetic delta, profile-shape moments, footprints, time-decay and a parametric
  **EM-GMM** profile decomposition, emitted as a `FactorDataset`/`MetaDataset` straight into the harness;
  plus survivorship-injection, feature-decomposition and MFE/MAE-XGBoost stress tests.
- 144 unit tests, including signal-detection *and* null-clearing checks for every evaluator.

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
python examples/structural_demo.py --perms 200        # 7: structural microstructure features
python examples/structural_survivorship.py            # 8: structural + survivorship injection
python examples/structural_decompose.py               # 9: per-feature + subgroup + cost decomposition
python examples/structural_mfe_xgb.py                 # 9: MFE/MAE triple-barrier + XGBoost (optional)
python examples/structural_swing_rater.py             # 10: swing setup-rater (R:R + selectivity)
python examples/structural_selectivity.py             # 11: selectivity stress-test (grid/decomp/power)
python examples/structural_gmm.py                     # 12: EM-GMM decomposition vs heuristic + survivorship
python examples/feature_purge.py                      # 13: orthogonality purge — feature clustering + IC
```

## Limitations

Survivorship bias throughout; a single 2012–2017 in-sample period; daily bars only; gross-of-cost
except where noted (meta-labeling tested net of 10 bps); a thin universe by cross-sectional
standards. None of the above is a forward guarantee or financial advice — it is a research log.
