# Changelog

All notable changes to `vpts`, by version. The project grew in two acts — **a product** (Phases 1–6, `v0.1`→`v1.0`) and then **its adversarial validation** (`v1.1`→`v1.9`). Format loosely follows [Keep a Changelog](https://keepachangelog.com); research findings are noted where a version produced one.

The canonical research narrative is [`RESEARCH.md`](RESEARCH.md); experiment numbers below refer to it.

---

## Act II — the validation

### `1.9.0` — Feature-orthogonality audit (the purge)
- **Added** `examples/feature_purge.py` + `cluster_features` — pool all 13 structural + 7 EM‑GMM features plus a no‑GMM VWAP/momentum baseline, Spearman‑cluster them (scipy hierarchical, distance 1−|ρ|), and overlay each feature's standalone OOS IC. +3 unit tests (144 total).
- **Finding (experiment 13):** the matrix is **wide but shallow**. It is *not* collinear (23 features → ~19 independent clusters, so "everything collapses to momentum/VWAP" is false), but only **6/23 features clear |IC|≥0.05** — a momentum/VWAP cluster (`vwap_dist` +0.14, `mom_120`/`gmm_gravity` +0.11) plus short‑horizon momentum and a thin dip tail (`cost_basis_migration` +0.06, `delta_net` +0.05). The other ~17 (most GMM geometry + the profile‑shape family) are orthogonal yet ≈0 IC. The honest purge is an **IC filter, not a correlation filter**: the matrix isn't redundant, it's mostly *null*. **No new edge.**

### `1.8.0` — Parametric EM‑GMM profile decomposition
- **Added** `vpts.structure.gmm`: a pure‑numpy weighted **1‑D Gaussian‑mixture EM** (deterministic quantile init, BIC model‑selection over k∈{1,2,3}) that decomposes the volume profile into hidden POCs → 7 scale‑free features (mode separation, antimode/LVN transition zone, fair‑value gravity) → `build_gmm_dataset` → `FactorDataset`, straight into the existing CPCV factor harness. +6 unit tests (141 total).
- **Finding (experiment 12):** the 7‑feature ridge IC ≈ 0 was **ridge dilution**, not absence of signal — read per‑feature, `gmm_gravity` carries a real survivor signal (OOS IC **+0.090**). But it is **0.91‑correlated with a one‑line `vwap_dist` = (close−VWAP)/range** that scores **higher** (+0.125), adds **no** incremental IC, and has **+0.016 partial correlation** controlling for momentum/VWAP. The "hidden‑POC gravity" is a moving‑average distance in disguise; the decomposition machinery adds nothing (and the one‑liner beats it). **No edge.**

### `1.7.0` — Structural microstructure analytics *(the strongest, and most instructive, signal)*
- **Added** `vpts.structure`: synthetic delta (CLV×volume), profile skew/kurtosis, P/b/B/D shape, ledges, poor highs, value‑area‑compression z‑score, time‑decayed cost‑basis migration → `FactorDataset` **and** `MetaDataset` (MFE/MAE triple‑barrier).
- **Added** `cpcv_factor_quantile_returns` (long/short/**flat** conviction buckets) and a relative `select_top` mode for the meta‑eval (act on the best‑rated fraction).
- **Added** a swing setup‑rater and the survivorship‑injection / decomposition / selectivity stress harnesses.
- **Findings (experiments 6–11):** a real OOS correlation (IC +0.035, p=0.005) that survives universe‑widening — but decomposition shows it is a **survivorship mirage** (the conviction edge *inverts* off survivors, +0.26 → −1.07%/bet), carried by dip‑buying features. The most resilient thread, meta‑labeling **selectivity**, is robust on survivors (9/9 params, p=0.023) yet not significant once delisted names are injected (p=0.106). XGBoost on an MFE/MAE target overfits to a sub‑0.5 OOS AUC. **No survivorship‑robust edge.**

### `1.6.0` — Cross‑sectional rank factors
- **Added** `build_cross_sectional_panel` + `cross_sectional_ic_eval` — the standard equity‑alpha construction, scored **within‑date** with a within‑date permutation null.
- **Finding (experiment 5):** a cross‑sectional near‑miss that vanished when properly powered — no edge.

### `1.5.0` — Enriched features + factor permutation test
- **Added** `build_enriched_factor_dataset` (richer per‑name inputs beyond the four coarse confluence factors) and `permutation_test_factor`.
- **Finding (experiment 4):** richer inputs, still OOS IC ≈ 0 — no edge.

### `1.4.0` — Cost‑aware meta‑eval + permutation significance
- **Added** per‑trade cost to `cpcv_meta_eval` and `permutation_test_meta` (label‑shuffle null for AUC and return‑lift), plus the survivorship stress harness for meta‑labeling.
- **Finding (experiment 3):** the meta‑labeling “edge” was **survivorship** — significant on survivors (p≈0.005), gone under injection (p≈0.80).

### `1.3.0` — Triple‑barrier meta‑labeling
- **Added** `triple_barrier_labels` (first‑touch profit/stop/vertical = MFE/MAE outcome), `build_meta_dataset`, `LogisticMetaModel`, `cpcv_meta_eval`.

### `1.2.0` — Learned factor weights
- **Added** `vpts.ml`: `RidgeFactorModel` + `cpcv_factor_eval` — the first *fitted* model, scored as a distribution of OOS IC across CPCV paths, with the hand‑weighted baseline for comparison.
- **Finding (experiment 2):** learned weights on the confluence factors → OOS IC ≈ 0.

### `1.1.0` — Validation harness *(the turning point)*
- **Added** `vpts.validation`: `CombinatorialPurgedCV` with purging + embargo, and immutable split results.
- **Fixed** 5 correctness bugs surfaced by a max‑effort review (regression‑tested).

---

## Act I — the product

### `1.0.0` — Phases 1–6 complete *(the trading system)*
- **Added** `vpts.backtest`: walk‑forward, no‑look‑ahead engine with realistic free costs (slippage + spread + commission), fixed‑fractional sizing, equity curve + blotter + stats. *A truth‑teller, not a money‑printer.*

### `0.5.0` — Phase 5 · Dashboard
- **Added** `vpts.dashboard`: pure Plotly figure builders (unit‑tested headless) + a thin Streamlit app (deep‑dive + watchlist scanner); deployable free on Streamlit Community Cloud.

### `0.4.0` — Phase 4 · Signals
- **Added** `vpts.signals`: `SignalGenerator` with reversion/breakout styles, structure‑based entry/stop/targets, minimum‑R:R gating, fixed‑fractional sizing, and journal‑ready `explain()`.

### `0.3.0` — Phase 3 · Confluence scoring
- **Added** `vpts.scoring`: `ConfluenceScorer` → `setup_quality` (0–100) + signed `bias_score`, from four transparent weighted components.

### `0.2.0` — Phase 2 · Regime
- **Added** `vpts.regime`: `QuietPhaseDetector` (percentile‑ranked vol/volume/compression) + `VolumePatternDetector` (dry‑up, accumulation, divergence, climax), on dependency‑free indicators.

### `0.1.0` — Phase 1 · Volume Profile
- **Added** `vpts.profile`: `VolumeProfileCalculator` (POC, VAH/VAL, HVN/LVN) with volume‑conserving intra‑bar distribution and volatility‑aware auto‑binning; `vpts.data` robust fetcher.

---

*Versions are tracked in `vpts.__version__`. Each minor bump corresponds to one snap‑in module or one validation milestone.*
