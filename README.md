# Quiet-Volume — a free Volume Profile trading system

A modular, explainable **Volume Profile** trading system for retail traders,
designed to detect institutional activity at price and to shine in **quiet,
low-volatility** market phases.

> **100% free.** No paid APIs, no premium data. Only open-source libraries and
> free data (Yahoo Finance via `yfinance`). Runs locally or on Google Colab
> (free tier).

---

## Status — Phases 1–2 of 6 ✅

The project is built in self-contained phases that snap together:

| Phase | Module           | What it does                                            | State |
|------:|------------------|---------------------------------------------------------|:-----:|
| **1** | `vpts.profile`   | Volume Profile Calculator — POC, VAH/VAL, HVN, LVN      | ✅ done |
| **2** | `vpts.regime`    | Quiet-phase detector + volume-pattern recognition       | ✅ done |
| 3     | `vpts.scoring`   | Confluence & scoring engine (0–100)                     | ⏳ next |
| 4     | `vpts.signals`   | Signal generator with natural-language explanations     | ⏳ |
| 5     | `vpts.dashboard` | Streamlit dashboard with volume-profile visualization   | ⏳ |
| 6     | `vpts.backtest`  | Backtester with realistic (free) cost simulation        | ⏳ |

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
examples/
  phase1_demo.py         # live volume-profile demo
  phase2_demo.py         # live quiet-phase + volume-pattern demo
tests/
  test_phase1.py         # offline, deterministic (23 tests)
  test_phase2.py         # offline, deterministic (17 tests)
requirements.txt
```

*Not financial advice. For research and education.*
