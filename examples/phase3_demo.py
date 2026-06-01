"""Phase 3 live demo — full stack: profile → quiet → patterns → confluence.

Fetches a real ticker and prints a single, explainable confluence read for the
latest bar (setup quality 0–100 + directional bias), with the component
breakdown and rationale.

Requires network (Yahoo via ``yfinance``). For an offline, deterministic check,
run ``tests/test_phase3.py`` instead.

Examples
--------
    python examples/phase3_demo.py                 # AAPL, 1y daily
    python examples/phase3_demo.py NVDA 2y 1d
    python examples/phase3_demo.py SPY 60d 1h
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
    ConfluenceScorer,
    DataFetchError,
    MarketDataFetcher,
    QuietPhaseDetector,
    VolumePatternDetector,
    VolumeProfileCalculator,
)


def main(symbol: str = "AAPL", period: str = "1y", interval: str = "1d") -> int:
    try:
        df = MarketDataFetcher().fetch(symbol, period=period, interval=interval)
    except DataFetchError as exc:
        print(f"Could not fetch {symbol}: {exc}")
        return 1

    print(f"Fetched {len(df)} bars for {symbol} [{period}/{interval}].\n")

    # Build the Phase 1/2 pieces explicitly so we can show them too …
    profile = VolumeProfileCalculator().calculate(df, symbol=symbol, interval=interval)
    quiet = QuietPhaseDetector().detect(df, symbol=symbol, interval=interval)
    patterns = VolumePatternDetector().detect(
        df, profile=profile, symbol=symbol, interval=interval
    )

    # … then fuse them into the confluence score.
    score = ConfluenceScorer().score(
        df, profile, quiet, patterns, symbol=symbol, interval=interval
    )

    print(profile.summary())
    print()
    print(quiet.summary())
    print()
    print(patterns.summary())
    print()
    print(score.summary())
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    symbol = args[0] if len(args) > 0 else "AAPL"
    period = args[1] if len(args) > 1 else "1y"
    interval = args[2] if len(args) > 2 else "1d"
    raise SystemExit(main(symbol, period, interval))
