"""Phase 4 live demo — end-to-end trade signal on a real ticker.

Runs the full stack (profile → quiet → patterns → confluence → signal) and prints
the journal-ready trade plan.

Requires network (Yahoo via ``yfinance``). For an offline, deterministic check,
run ``tests/test_phase4.py`` instead.

Examples
--------
    python examples/phase4_demo.py                          # AAPL, 1y daily, reversion
    python examples/phase4_demo.py NVDA 2y 1d breakout
    python examples/phase4_demo.py SPY 60d 1h reversion
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import DataFetchError, MarketDataFetcher, SignalGenerator  # noqa: E402


def main(
    symbol: str = "AAPL",
    period: str = "1y",
    interval: str = "1d",
    style: str = "reversion",
    equity: float = 10_000.0,
) -> int:
    try:
        df = MarketDataFetcher().fetch(symbol, period=period, interval=interval)
    except DataFetchError as exc:
        print(f"Could not fetch {symbol}: {exc}")
        return 1

    print(f"Fetched {len(df)} bars for {symbol} [{period}/{interval}].\n")
    signal = SignalGenerator(style=style).analyze(
        df, account_equity=equity, symbol=symbol, interval=interval
    )
    print(signal.explain())
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    symbol = args[0] if len(args) > 0 else "AAPL"
    period = args[1] if len(args) > 1 else "1y"
    interval = args[2] if len(args) > 2 else "1d"
    style = args[3] if len(args) > 3 else "reversion"
    raise SystemExit(main(symbol, period, interval, style))
