"""Multi-ticker backtest sweep — 20 mid-cap stocks across industries.

The build sandbox can't reach Yahoo, and the bundled free data source only
serves a few large-caps — so this script is meant to be run **locally** (or on
Colab), where ``yfinance`` works:

    python examples/midcap_scan.py                 # 5y daily, both styles
    python examples/midcap_scan.py --period 3y --plot .

It backtests a diversified basket of ~20 mid-cap names in both the ``reversion``
and ``breakout`` styles, prints a per-ticker table and per-style aggregates, and
(optionally) renders an equal-weight aggregate equity curve + a per-ticker return
bar chart.

The data-fetching is injected (``load_fn``) so :func:`run_scan` is testable
offline with synthetic data; ``main()`` wires in the real ``MarketDataFetcher``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from vpts import (  # noqa: E402
    Backtester,
    CostModel,
    DataFetchError,
    MarketDataFetcher,
    SignalGenerator,
)

# 20 mid-cap names spread across ~10 industries (recognisable US-listed tickers).
MIDCAPS: list[tuple[str, str]] = [
    ("SAIA", "Industrials"),        ("AAON", "Industrials"),
    ("TXRH", "Consumer Disc."),     ("CROX", "Consumer Disc."),
    ("COKE", "Consumer Staples"),   ("CALM", "Consumer Staples"),
    ("WAL", "Financials"),          ("EVR", "Financials"),
    ("EXEL", "Healthcare"),         ("HALO", "Healthcare"),
    ("MEDP", "Healthcare"),         ("PCTY", "Technology"),
    ("LSCC", "Technology"),         ("AMKR", "Technology"),
    ("MGY", "Energy"),              ("CIVI", "Energy"),
    ("CUBE", "Real Estate"),        ("EGP", "Real Estate"),
    ("AVNT", "Materials"),          ("IDA", "Utilities"),
]
STYLES = ("reversion", "breakout")


def backtest_one(
    df: pd.DataFrame,
    symbol: str,
    style: str,
    *,
    equity: float = 10_000.0,
    slippage_bps: float = 5.0,
    lookback: int = 120,
    min_quality: float = 50.0,
    min_abs_bias: float = 12.0,
):
    """Run one ticker/style backtest and return the :class:`BacktestResult`."""
    bt = Backtester(
        lookback=lookback,
        initial_equity=equity,
        signal_generator=SignalGenerator(
            style=style, min_quality=min_quality, min_abs_bias=min_abs_bias
        ),
        cost_model=CostModel(slippage_bps=slippage_bps),
    )
    return bt.run(df, symbol=symbol, interval="1d")


def _fmp_to_df(payload) -> pd.DataFrame:
    """Convert an FMP ``historical-price-full`` payload into an OHLCV frame."""
    hist = payload.get("historical") if isinstance(payload, dict) else payload
    if not hist:
        raise DataFetchError("no 'historical' data in FMP payload")
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    df["Volume"] = df["Volume"].astype(float)
    return df


def fmp_loader(api_key: str, years: int = 5) -> Callable[[str], pd.DataFrame]:
    """Return a ``load(symbol) -> df`` backed by the FMP free EOD endpoint.

    The free tier (used with *your own* key) serves daily history for US stocks
    incl. mid-caps, ~250 calls/day — plenty for a 20-name sweep. Requires
    network access to financialmodelingprep.com (so run this locally).
    """
    import datetime as _dt

    import requests  # bundled via yfinance

    end = _dt.date.today()
    start = end - _dt.timedelta(days=365 * years + 30)

    def load(symbol: str) -> pd.DataFrame:
        url = (
            "https://financialmodelingprep.com/api/v3/historical-price-full/"
            f"{symbol}?from={start}&to={end}&apikey={api_key}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return _fmp_to_df(resp.json())

    return load


def run_scan(
    load_fn: Callable[[str], pd.DataFrame],
    tickers: list[tuple[str, str]],
    styles: tuple[str, ...] = STYLES,
    **bt_kwargs,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.Series]]]:
    """Backtest every ``(ticker, sector)`` in each style.

    Returns ``(results_df, curves)`` where *curves[style][symbol]* is the
    equity curve normalised to start at 100.
    """
    rows: list[dict] = []
    curves: dict[str, dict[str, pd.Series]] = {s: {} for s in styles}
    for sym, sector in tickers:
        try:
            df = load_fn(sym)
        except (DataFetchError, Exception) as exc:  # noqa: BLE001 - skip bad tickers
            print(f"  ! {sym}: skipped ({type(exc).__name__}: {exc})")
            continue
        for style in styles:
            try:
                res = backtest_one(df, sym, style, **bt_kwargs)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {sym}/{style}: {type(exc).__name__}: {exc}")
                continue
            rows.append({
                "symbol": sym, "sector": sector, "style": style,
                "return_%": round(res.total_return_pct, 1),
                "trades": res.n_trades,
                "win_%": round(res.win_rate * 100, 1),
                "PF": (float("inf") if res.profit_factor == float("inf")
                       else round(res.profit_factor, 2)),
                "maxDD_%": round(res.max_drawdown_pct, 1),
                "Sharpe": round(res.sharpe, 2),
            })
            eq = res.equity_curve
            if len(eq) and eq.iloc[0]:
                curves[style][sym] = eq / eq.iloc[0] * 100.0
    return pd.DataFrame(rows), curves


def aggregate_curve(curves_for_style: dict[str, pd.Series]) -> pd.Series | None:
    """Equal-weight mean of normalised equity curves (a simple 'portfolio')."""
    if not curves_for_style:
        return None
    mat = pd.concat(curves_for_style.values(), axis=1).sort_index().ffill()
    return mat.mean(axis=1)


def _print_report(results: pd.DataFrame) -> None:
    if results.empty:
        print("No results (no tickers returned data).")
        return
    # Per-ticker returns pivoted by style, with sector.
    pivot = results.pivot_table(index=["sector", "symbol"], columns="style",
                                values="return_%")
    print("\nPer-ticker total return % by style:")
    print(pivot.to_string())

    print("\nPer-style aggregate:")
    agg = results.groupby("style").agg(
        mean_return_pct=("return_%", "mean"),
        median_return_pct=("return_%", "median"),
        pct_profitable=("return_%", lambda s: round((s > 0).mean() * 100, 1)),
        mean_sharpe=("Sharpe", "mean"),
        total_trades=("trades", "sum"),
    ).round(2)
    print(agg.to_string())


def _plot(curves: dict[str, dict[str, pd.Series]], results: pd.DataFrame,
          out_dir: str) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        print("  (plotly not installed — skipping plots)")
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    colors = {"reversion": "#ef5350", "breakout": "#26a69a"}
    for style, per_sym in curves.items():
        agg = aggregate_curve(per_sym)
        if agg is not None:
            fig.add_trace(go.Scatter(x=agg.index, y=agg.to_numpy(), name=style,
                                     line=dict(color=colors.get(style), width=2)))
    fig.add_hline(y=100, line=dict(color="#90a4ae", width=1, dash="dash"))
    fig.update_layout(template="plotly_dark", height=420,
                      title="Equal-weight aggregate equity (start = 100)")
    fig.write_image(str(out / "midcap_aggregate_equity.png"), width=1200, height=420, scale=2)
    print(f"  wrote {out / 'midcap_aggregate_equity.png'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest 20 mid-cap stocks in both styles.")
    ap.add_argument("--source", choices=["yfinance", "fmp"], default="yfinance",
                    help="data backend (default yfinance)")
    ap.add_argument("--fmp-key", default=os.environ.get("FMP_API_KEY"),
                    help="FMP API key (or set FMP_API_KEY); required for --source fmp")
    ap.add_argument("--period", default="5y", help="yfinance look-back period")
    ap.add_argument("--years", type=int, default=5, help="FMP look-back in years")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--style", choices=STYLES, help="limit to one style")
    ap.add_argument("--plot", metavar="DIR", help="render aggregate charts into DIR")
    args = ap.parse_args()

    if args.source == "fmp":
        if not args.fmp_key:
            ap.error("--source fmp requires --fmp-key or the FMP_API_KEY env var")
        load_fn = fmp_loader(args.fmp_key, years=args.years)
    else:
        fetcher = MarketDataFetcher()
        load_fn = lambda s: fetcher.fetch(s, period=args.period, interval=args.interval)  # noqa: E731
    styles = (args.style,) if args.style else STYLES

    print(f"Scanning {len(MIDCAPS)} mid-caps [{args.period}/{args.interval}], "
          f"styles={styles} … (this fetches live data; please be patient)\n")
    results, curves = run_scan(load_fn, MIDCAPS, styles=styles, equity=args.equity)
    _print_report(results)
    if args.plot:
        _plot(curves, results, args.plot)
    print("\nNote: in-sample, single-period, small basket — a demonstration of the "
          "framework, not a validated strategy. Tune & validate out-of-sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
