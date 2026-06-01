# Quiet-Volume — a free Volume Profile trading system

A modular, explainable **Volume Profile** trading system for retail traders,
designed to detect institutional activity at price and to shine in **quiet,
low-volatility** market phases.

> **100% free.** No paid APIs, no premium data. Only open-source libraries and
> free data (Yahoo Finance via `yfinance`). Runs locally or on Google Colab
> (free tier).

---

## Status — all 6 phases complete ✅

The project is built in self-contained phases that snap together:

| Phase | Module           | What it does                                            | State |
|------:|------------------|---------------------------------------------------------|:-----:|
| **1** | `vpts.profile`   | Volume Profile Calculator — POC, VAH/VAL, HVN, LVN      | ✅ done |
| **2** | `vpts.regime`    | Quiet-phase detector + volume-pattern recognition       | ✅ done |
| **3** | `vpts.scoring`   | Confluence & scoring engine (0–100 + bias)              | ✅ done |
| **4** | `vpts.signals`   | Signal generator with trade plans & explanations        | ✅ done |
| **5** | `vpts.dashboard` | Streamlit + Plotly dashboard (deep-dive + scanner)      | ✅ done |
| **6** | `vpts.backtest`  | Walk-forward backtester with realistic (free) costs     | ✅ done |

**83 offline, deterministic tests** (no network) cover the whole stack and pass under `pytest`.

---

## Install

```bash
pip install -r requirements.txt        # full stack
# Phase 1 alone only needs:
pip install numpy pandas scipy yfinance
```

> **Note on `pandas-ta`:** not required. It is fragile right now (`0.3.14b0`
> breaks on numpy ≥ 2.0; `0.4.x` needs Python ≥ 3.12), so Phases 1–2 compute
> everything they need (ATR, slopes, percentile ranks, bandwidth) with a small
> dependency-free helper module (`vpts.regime.indicators`). `pandas-ta` stays
> optional.

## Quick start

```python
from vpts import MarketDataFetcher, VolumeProfileCalculator

# 1) Fetch clean OHLCV (cached, retried, interval-limit aware)
df = MarketDataFetcher().fetch("AAPL", period="6mo", interval="1d")

# 2) Build the volume profile
profile = VolumeProfileCalculator(num_bins=100).calculate(df, symbol="AAPL")

print(profile.summary())
print("POC:", profile.poc, "VAH:", profile.vah, "VAL:", profile.val)
print("HVNs:", [round(n.price, 2) for n in profile.hvn])
```

Run the bundled demo / tests:

```bash
python examples/phase1_demo.py AAPL 6mo 1d   # live (needs internet)
python tests/test_phase1.py                  # offline, deterministic
```

---

## Phase 1 concepts

* **POC** (Point of Control) — the single price level with the most traded
  volume; the profile's center of gravity.
* **Value Area (VAH / VAL)** — the contiguous band around the POC that holds
  ~70% of volume (configurable). Acceptance = price inside; imbalance = outside.
* **HVN** (High Volume Nodes) — volume *peaks*: zones of acceptance that tend to
  act as support/resistance.
* **LVN** (Low Volume Nodes) — volume *valleys*: "air pockets" price moves
  through quickly; natural targets and breakout levels.

### How intra-bar volume is handled

Free OHLCV data has no tick detail, so the volume *inside* a bar is approximated:

* `"uniform"` (default) — spreads each bar's volume across its `[Low, High]`
  range, proportional to bin overlap. Faithful and **conserves total volume
  exactly**.
* `"typical"` — assigns each bar's whole volume to the bin of its typical price
  `(H+L+C)/3`. Faster, coarser.

### Configuring the calculator

Everything is set on the constructor:

```python
VolumeProfileCalculator(
    num_bins=100,            # fixed-mode resolution
    value_area_pct=0.70,     # value-area target — configurable (e.g. 0.68, 0.80)
    distribution="uniform",  # or "typical"
    bin_mode="fixed",        # or "auto"
)
```

**Auto-binning (`bin_mode="auto"`)** sizes bins from *volatility* instead of a
fixed count: target bin width ≈ `atr_bin_fraction × ATR(atr_period)`, with the
resulting count clamped to `[min_bins, max_bins]`. Quiet / low-ATR regimes get
**finer** resolution (more bins); volatile regimes get coarser bins — a natural
fit for a system built around quiet phases. The chosen ATR, target width and bin
count are recorded on `profile.extra` and shown in `summary()`.

```python
VolumeProfileCalculator(bin_mode="auto", atr_period=14, atr_bin_fraction=0.25)
```

### `yfinance` intraday limits (handled automatically)

`MarketDataFetcher` clamps the requested period to what Yahoo actually serves and
retries with exponential backoff:

| Interval        | Max history |
|-----------------|-------------|
| `1m`            | ~7 days     |
| `2m/5m/15m/30m/90m` | ~60 days |
| `1h` / `60m`    | ~730 days   |
| `1d` and coarser| full        |

> Cash indices (e.g. `^GDAXI`, `^GSPC`) report **zero volume** on Yahoo, so a
> volume profile is undefined — use a tradable proxy (an ETF like `SPY`/`EWG` or
> a futures contract). The fetcher raises a clear `NoVolumeError` in that case.

---

## Phase 2 — quiet phases & volume patterns

```python
from vpts import (MarketDataFetcher, VolumeProfileCalculator,
                  QuietPhaseDetector, VolumePatternDetector)

df = MarketDataFetcher().fetch("AAPL", period="1y", interval="1d")
profile = VolumeProfileCalculator().calculate(df)

quiet = QuietPhaseDetector().detect(df)
print(quiet.summary())
print("Quiet right now?", quiet.is_quiet, "->", quiet.latest.explanation)

# Volume patterns, anchored to the Phase-1 profile levels:
patterns = VolumePatternDetector().detect(df, profile=profile)
for p in patterns.recent(5):
    print(p.explanation)
```

### Quiet-Phase Detector

Blends three **self-normalising** signals — each a trailing *percentile rank*, so
no hand-tuned absolute thresholds and comparable across instruments/timeframes:

* **Low volatility** — ATR ranked against its own history.
* **Declining / dry volume** — a volume moving average ranked against history.
* **Range compression** — Bollinger Bandwidth ranked against history.

They combine into a `quiet_score` (0–100) and an `is_quiet` flag; the result also
exposes per-bar analytics, contiguous `quiet_segments()`, and a plain-language
`explanation` ("volatility in the bottom 12% of its range; volume in the bottom
20% and falling; range compressed").

### Volume Pattern Detector

Recognises four institution-revealing behaviours, each returned with a natural-
language reason and — when a profile is supplied — **anchored to a level**
(`"climax at POC 204.52"`):

* **Volume Dry-up** — sustained volume below its longer-term baseline (pre-breakout coil).
* **Accumulation** — tight, flat range on below-average volume (quiet absorption).
* **Volume Divergence** — price trend not confirmed by volume.
* **Volume Climax** — extreme volume on a wide-range bar (potential exhaustion).

Everything is computed with the dependency-free `vpts.regime.indicators` helpers
(no `pandas-ta`). Run the demo / tests:

```bash
python examples/phase2_demo.py AAPL 1y 1d   # live (needs internet)
python tests/test_phase2.py                 # offline, deterministic
```

---

## Phase 3 — confluence scoring

Fuses everything above into one explainable read of *now*:

```python
from vpts import ConfluenceScorer

score = ConfluenceScorer().analyze(df)        # builds profile/quiet/patterns for you
print(score.summary())
print(score.bias, score.setup_quality, score.bias_score)   # e.g. 'bullish' 74 +38
```

* **`setup_quality`** `0–100` — how much aligned evidence is present now.
* **`bias`** — `bullish` / `bearish` / `neutral`, with a signed **`bias_score`**
  `-100..100`. By construction `|bias_score| ≤ setup_quality` (conviction can't
  exceed the evidence).

Four transparent, weighted components — **value-area location**, **key-level
proximity** (HVN/LVN), the **quiet regime** (a non-directional quality amplifier
— the system's edge), and **active volume patterns** — each carry a strength,
direction and a one-line reason, surfaced in a readable `summary()` and a
`breakdown()` dict. Weights are configurable. A sample read-out:

```
Confluence — AAPL 1d @ 232.41
  Setup quality : 74/100
  Directional   : BULLISH  (bias +38)
  Rationale     : Bullish setup (quality 74/100): quiet coil (score 81/100) — primed
                  for a move; holding above HVN 231.80 (support).
  Components:
    quiet        [w1.5] █████░ 0.81  · neut  quiet coil (score 81/100) — primed for a move
    key_level    [w1.0] ████░░ 0.66  ↑ bull  holding above HVN 231.80 (support)
    patterns     [w1.5] ███░░░ 0.55  ↑ bull  accumulation near HVN 231.80
    value_area   [w1.0] ██░░░░ 0.30  · neut  balanced inside the value area (POC 230.9)
```

```bash
python examples/phase3_demo.py AAPL 1y 1d   # live (needs internet)
python tests/test_phase3.py                 # offline, deterministic
```

---

## Phase 4 — trade signals

Turns the confluence read into an actionable, fully-explained plan:

```python
from vpts import SignalGenerator

signal = SignalGenerator(style="reversion").analyze(df, account_equity=10_000)
print(signal.explain())          # journal-ready write-up
if signal.is_actionable:
    print(signal.action, signal.entry, signal.stop, signal.targets)
    print(signal.risk_reward_ratio, signal.suggested_size)
```

* **Gating** — only acts when `setup_quality`, `|bias_score|` (and optionally a
  quiet phase) clear configurable thresholds; otherwise a reasoned `NO_TRADE`.
* **Two styles** — `"reversion"` (fade value-area edges / HVN back toward the
  POC) and `"breakout"` (trade the bias as price expands out of the coil).
* **Plan from structure** — entry / stop / targets come from profile levels
  (stop beyond a level or an ATR multiple; targets at POC / value edges).
* **Risk** — minimum R:R filter, plus a free **fixed-fractional** position size
  (`risk %` ÷ stop distance). `risk_reward_ratio` and `suggested_size` are
  exposed directly on the `TradeSignal`.

A sample `explain()`:

```
TRADE SIGNAL — LONG AAPL 1d [reversion]
  Setup       : quality 57/100, bias BULLISH (+26)
  Entry       : 100.95
  Stop        : 99.78   (risk 1.17/unit)
  Target(s)   : 108.02
  R:R         : 6.08  (to first target)
  Size        : 85 units  (risk 100.00 = 1.0% of 10,000)
  Why         : Long (reversion, fade toward value): Bullish setup (quality 57/100):
                quiet coil (score 95/100) — primed for a move; Possible accumulation.
```

```bash
python examples/phase4_demo.py AAPL 1y 1d reversion   # live (needs internet)
python tests/test_phase4.py                           # offline, deterministic
```

---

## Phase 5 — Streamlit dashboard

A dark-themed, single-page app that visualises the whole stack for a ticker:
candles with the **volume-profile histogram** overlaid (POC/VAH/VAL lines,
value-area band, HVN/LVN), **pattern markers**, the **quiet-score** panel with
shaded quiet segments, a **confluence gauge + component breakdown**, and the
**trade-signal card** with entry/stop/targets drawn on price. A second **Scanner**
tab ranks a watchlist by setup quality. (Defaults to `bin_mode="auto"`.)

```bash
pip install -r requirements.txt          # needs the streamlit + plotly extras
streamlit run vpts/dashboard/app.py
```

The Plotly figure builders live in `vpts.dashboard.charts` as **pure functions**
(`go.Figure` in → out), so they're unit-tested offline; `app.py` is just the thin
interactive shell. Tests even run the whole app headless via Streamlit's
`AppTest`:

```bash
python tests/test_phase5.py     # figure builders + headless app run (offline)
```

---

## Phase 6 — backtester

A **walk-forward, no-look-ahead** backtest of the whole stack:

```python
from vpts import Backtester, SignalGenerator, CostModel

bt = Backtester(
    lookback=120,                                   # rolling window per decision
    signal_generator=SignalGenerator(style="reversion"),
    cost_model=CostModel(slippage_bps=5),           # free retail: 0 commission
)
result = bt.run(df, symbol="AAPL", interval="1d")
print(result.summary())
print(result.trades_dataframe().tail())
```

* **No look-ahead** — at the close of bar *t* the profile/regime/confluence/signal
  are computed from a rolling window ending at *t*; entries fill at the **open of
  *t+1*** (cost-adjusted). Only past data is ever used.
* **Realistic free costs** — slippage + spread + commission (`CostModel`), all
  adverse; overnight-gap fills handled.
* **One position at a time**, managed against its stop / target(s) with an
  optional time stop; **fixed-fractional** sizing on *current* equity.
* **`BacktestResult`** — equity curve, trade blotter, and headline stats:
  total return %, win rate, profit factor, max drawdown, expectancy, avg R,
  Sharpe. A reusable `charts.equity_curve_figure(result)` plots the curve with its
  drawdown envelope.

> The backtester is a **truth-teller, not a money-printer**: default settings are
> not auto-profitable, and that's the point — it measures an edge honestly so you
> can tune and validate before risking capital. *Not financial advice.*

```bash
python examples/phase6_demo.py AAPL 5y 1d reversion   # live (needs internet)
python tests/test_phase6.py                           # offline, deterministic
```

---

## Project layout

```
vpts/
  __init__.py            # public API (re-exports Phases 1–2)
  data/
    fetcher.py           # robust, cached yfinance wrapper
  profile/               # Phase 1
    calculator.py        # VolumeProfileCalculator
    models.py            # VolumeProfile + VolumeNode (immutable results)
  regime/                # Phase 2
    indicators.py        # dependency-free ATR / slope / percentile / bandwidth
    quiet.py             # QuietPhaseDetector + QuietState / QuietPhaseResult
    patterns.py          # VolumePatternDetector + VolumePattern(s)
  scoring/               # Phase 3
    scorer.py            # ConfluenceScorer
    models.py            # ConfluenceScore + ConfluenceComponent (immutable)
  signals/               # Phase 4
    generator.py         # SignalGenerator
    models.py            # TradeSignal + SignalAction (immutable)
  dashboard/             # Phase 5
    charts.py            # pure Plotly figure builders (unit-tested)
    app.py               # thin Streamlit shell — `streamlit run`
  backtest/              # Phase 6
    engine.py            # Backtester (no-look-ahead walk-forward)
    models.py            # BacktestResult + Trade + CostModel (immutable)
examples/
  phase1_demo.py … phase6_demo.py   # one live demo per phase
tests/
  test_phase1.py         # offline, deterministic (23 tests)
  test_phase2.py         # offline, deterministic (17 tests)
  test_phase3.py         # offline, deterministic (11 tests)
  test_phase4.py         # offline, deterministic (13 tests)
  test_phase5.py         # offline, deterministic (8 tests)
  test_phase6.py         # offline, deterministic (11 tests)
requirements.txt         # 83 tests total
```

*Not financial advice. For research and education.*
