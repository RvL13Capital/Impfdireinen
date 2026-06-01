# Quiet-Volume — a free Volume Profile trading system

A modular, explainable **Volume Profile** trading system for retail traders,
designed to detect institutional activity at price and to shine in **quiet,
low-volatility** market phases.

> **100% free.** No paid APIs, no premium data. Only open-source libraries and
> free data (Yahoo Finance via `yfinance`). Runs locally or on Google Colab
> (free tier).

---

## Status — Phase 1 of 6 ✅

The project is built in self-contained phases that snap together:

| Phase | Module           | What it does                                            | State |
|------:|------------------|---------------------------------------------------------|:-----:|
| **1** | `vpts.profile`   | Volume Profile Calculator — POC, VAH/VAL, HVN, LVN      | ✅ done |
| 2     | `vpts.regime`    | Quiet-phase detector + volume-pattern recognition       | ⏳ next |
| 3     | `vpts.scoring`   | Confluence & scoring engine (0–100)                     | ⏳ |
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

> **Note on `pandas-ta`:** it is only needed from Phase 2 (ATR). It is currently
> fragile (`0.3.14b0` breaks on numpy ≥ 2.0; `0.4.x` needs Python ≥ 3.12), so
> Phase 2 will compute indicators with a small dependency-free helper and keep
> `pandas-ta` optional. Phase 1 does not use it.

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

## Project layout

```
vpts/
  __init__.py            # public API: MarketDataFetcher, VolumeProfileCalculator, …
  data/
    fetcher.py           # robust, cached yfinance wrapper
  profile/
    calculator.py        # VolumeProfileCalculator (the Phase-1 engine)
    models.py            # VolumeProfile + VolumeNode (immutable results)
examples/
  phase1_demo.py         # live demo against a real ticker
tests/
  test_phase1.py         # offline, deterministic test-suite
requirements.txt
```

*Not financial advice. For research and education.*
