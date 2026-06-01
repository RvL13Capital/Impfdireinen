"""Phase 6 live demo — walk-forward backtest on a real ticker.

Runs the full no-look-ahead backtest and prints the stats + trade blotter.

Requires network (Yahoo via ``yfinance``). For an offline, deterministic check,
run ``tests/test_phase6.py`` instead.

Examples
--------
    python examples/phase6_demo.py                       # AAPL, 5y daily, reversion
    python examples/phase6_demo.py MSFT 5y 1d breakout
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
    Backtester,
    CostModel,
    DataFetchError,
    MarketDataFetcher,
    SignalGenerator,
)


def main(
    symbol: str = "AAPL",
    period: str = "5y",
    interval: str = "1d",
    style: str = "reversion",
) -> int:
    try:
        df = MarketDataFetcher().fetch(symbol, period=period, interval=interval)
    except DataFetchError as exc:
        print(f"Could not fetch {symbol}: {exc}")
        return 1

    print(f"Backtesting {symbol} on {len(df)} bars [{period}/{interval}, {style}].\n")
    bt = Backtester(
        lookback=120,
        signal_generator=SignalGenerator(style=style, min_quality=45, min_abs_bias=12),
        cost_model=CostModel(slippage_bps=5.0),   # free retail: 0 commission, light slippage
    )
    result = bt.run(df, symbol=symbol, interval=interval)
    print(result.summary())

    blotter = result.trades_dataframe()
    if not blotter.empty:
        print("\nLast 8 trades:")
        print(blotter.tail(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    symbol = args[0] if len(args) > 0 else "AAPL"
    period = args[1] if len(args) > 1 else "5y"
    interval = args[2] if len(args) > 2 else "1d"
    style = args[3] if len(args) > 3 else "reversion"
    raise SystemExit(main(symbol, period, interval, style))
