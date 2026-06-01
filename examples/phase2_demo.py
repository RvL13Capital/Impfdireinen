"""Phase 2 live demo — quiet-phase + volume patterns on a real ticker.

Builds the Phase-1 volume profile, then runs the Phase-2 quiet-phase detector and
volume-pattern detector (anchored to the profile's levels) and prints everything.

Requires network (Yahoo via ``yfinance``). For an offline, deterministic check,
run ``tests/test_phase2.py`` instead.

Examples
--------
    python examples/phase2_demo.py                 # AAPL, 1y daily
    python examples/phase2_demo.py MSFT 2y 1d
    python examples/phase2_demo.py SPY 60d 1h
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
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

    # Phase 1 — volume profile (levels the patterns get anchored to).
    profile = VolumeProfileCalculator(num_bins=100).calculate(
        df, symbol=symbol, interval=interval
    )
    print(profile.summary())

    # Phase 2 — quiet-phase regime.
    print()
    quiet = QuietPhaseDetector().detect(df, symbol=symbol, interval=interval)
    print(quiet.summary())

    # Phase 2 — volume patterns, anchored to the profile.
    print()
    patterns = VolumePatternDetector().detect(
        df, profile=profile, symbol=symbol, interval=interval
    )
    print(patterns.summary())

    # A simple, explainable read of "now".
    print("\n--- Current read ---")
    state = "a QUIET phase" if quiet.is_quiet else "an ACTIVE phase"
    print(f"{symbol} is in {state} (score {quiet.latest.quiet_score:.0f}/100).")
    last = float(df["Close"].iloc[-1])
    print(f"Last close {last:.2f} is '{profile.location(last)}' vs the value area.")
    if patterns.latest is not None:
        print(f"Most recent volume pattern: {patterns.latest.explanation}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    symbol = args[0] if len(args) > 0 else "AAPL"
    period = args[1] if len(args) > 1 else "1y"
    interval = args[2] if len(args) > 2 else "1d"
    raise SystemExit(main(symbol, period, interval))
