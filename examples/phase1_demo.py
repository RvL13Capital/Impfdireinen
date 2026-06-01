"""Phase 1 live demo — fetch a real ticker and print its volume profile.

Requires network access (Yahoo Finance via ``yfinance``). For an offline,
deterministic check of the maths, run ``tests/test_phase1.py`` instead.

Examples
--------
    python examples/phase1_demo.py                 # AAPL, 6mo daily
    python examples/phase1_demo.py MSFT 1y 1d
    python examples/phase1_demo.py SPY 5d 5m       # intraday (auto-clamped)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import DataFetchError, MarketDataFetcher, VolumeProfileCalculator  # noqa: E402


def main(symbol: str = "AAPL", period: str = "6mo", interval: str = "1d") -> int:
    fetcher = MarketDataFetcher(cache_ttl=3600)  # 1h cache; polite to Yahoo
    try:
        df = fetcher.fetch(symbol, period=period, interval=interval)
    except DataFetchError as exc:
        print(f"Could not fetch {symbol}: {exc}")
        return 1

    print(f"Fetched {len(df)} bars for {symbol} [{period}/{interval}].")
    print(f"Latest close: {df['Close'].iloc[-1]:.2f}\n")

    profile = VolumeProfileCalculator(num_bins=100).calculate(
        df, symbol=symbol, interval=interval
    )
    print(profile.summary())

    last = float(df["Close"].iloc[-1])
    print(f"\nPrice location: last close {last:.2f} is "
          f"'{profile.location(last)}' relative to the value area.")
    nearest_hvn = profile.nearest_node(last, kind="HVN")
    if nearest_hvn is not None:
        print(f"Nearest HVN to price: {nearest_hvn}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    symbol = args[0] if len(args) > 0 else "AAPL"
    period = args[1] if len(args) > 1 else "6mo"
    interval = args[2] if len(args) > 2 else "1d"
    raise SystemExit(main(symbol, period, interval))
