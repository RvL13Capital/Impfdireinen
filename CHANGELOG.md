# Changelog

All notable changes to `vpts`, by version. The project grew in two acts — **a product** (Phases 1–6, `v0.1`→`v1.0`) and then **its adversarial validation**, capped by a forward paper‑walk and real‑volume crypto tests that close the synthesized‑volume question (`v1.1`→`v1.12`). Format loosely follows [Keep a Changelog](https://keepachangelog.com); research findings are noted where a version produced one.

The canonical research narrative is [`RESEARCH.md`](RESEARCH.md); experiment numbers below refer to it.

---

## Act II — the validation

### `1.12.0` — Real intraday volume profile (experiment 15) — the synthesized‑volume close
- **Added** `examples/crypto_intraday_profile.py` — builds the 13 structural features on **hourly** crypto bars (a 120‑bar window aggregates 120 *real* hourly volumes → a genuine real intraday profile, not a synthesized one) and scores each through CPCV. Reuses the `vpts.data.crypto` fetcher (`frequency="hours"`) + the structural builder; no new library code (158 tests).
- **Finding (experiment 15):** on real intraday volume the profile **geometry** (shape, skew, kurtosis, value‑area compression) is dead‑to‑anti‑predictive — geometry‑family mean OOS IC **−0.026** (kurtosis −0.13, 0/4 coins), full 13‑feature ridge **−0.006**; only the trend/dip features (cost‑basis migration +0.06, POC slope +0.04) flicker, as on every dataset. Real volume did **not** rescue the profile thesis — the synthesized input was never the bottleneck. **The volume‑profile thesis is closed.**

### `1.11.0` — Real volume & order flow (crypto) — the first non‑fabricated input
- **Added** `vpts.data.crypto` — a free, **keyless** crypto OHLCV **+ real aggressor buy/sell volume** fetcher (CCData spot API, paginated over the 100/call limit), so experiments can use *real* order flow instead of the synthetic close‑location‑value proxy, on continuously‑listed (survivorship‑light) majors. + `examples/crypto_realvol.py` and 6 offline tests (158 total).
- **Finding (experiment 14):** on 8 majors / 15.8k events, **real order flow beat the synthetic proxy ~5×** (pooled OOS IC **+0.020 vs +0.004**) and `vwap_dist` was **7/8 coins positive** (+0.049) — but per‑coin the signals are small, dispersed, alt‑concentrated, and **negative on BTC** (the most liquid coin), and the pooled p is inflated by cross‑coin correlation. Real volume + real flow **modestly improved the features but did not break the wall**. **No robust edge.**

### `1.10.0` — Forward paper‑walk *(survivorship‑free evidence, paper only)*
- **Added** `vpts.execution` — `run_paper_walk` / `PaperLedger` / `PaperOrder`: **decide** on bars ≤ as‑of (`SignalGenerator`, no look‑ahead), log actionable calls to an append‑only JSONL ledger (idempotent per `(symbol, date)`), and **resolve** prior open orders first‑touch against the bars since arrived — next‑bar‑open fill, stop / first‑target, time‑stop at `max_hold`. `summary()` reports **% profitable (R>0)**, avg R, and the exit‑type breakdown. Loader‑ and `as_of`‑injected → deterministic and network‑free to test. **Paper only — never places an order or moves money.** +8 unit tests (152 total).
- **Added** `examples/paper_walk.py` — `--live` (one honest day via free yfinance; drop behind a daily cron) and `--demo` (replays the mechanism on the static sample, no network).
- **Why:** thirteen experiments found no survivorship‑robust edge and named the **data** as the wall; a forward paper‑walk is the one evidence source that is **survivorship‑free by construction**. It will not manufacture an edge — it gathers clean, unbiased forward evidence one bar at a time.

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
