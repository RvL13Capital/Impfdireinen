"""No-key backtest sweep on free GitHub-hosted data (works in restricted networks).

Many sandboxes block Yahoo/FMP/EODHD but allow ``raw.githubusercontent.com``.
This script backtests a **sector-diverse basket of 20 US large-caps** using the
real, split/dividend-adjusted **5-year daily OHLCV** committed to the public
`stocknet-dataset <https://github.com/yumoxu/stocknet-dataset>`_ — **no API key,
no paid data**.

    python examples/github_data_scan.py --plot .                       # full 2012-2017
    python examples/github_data_scan.py --window 2015-01-01:2016-06-30  # a chop window

The ``--window FROM:TO`` slice makes it easy to compare regimes (e.g. a bull
window vs the 2015→2016 correction), which is the cleanest way to *see* that the
right style depends on the regime.

Reuses :func:`examples.midcap_scan.run_scan` / ``aggregate_curve`` — only the data
loader changes (GitHub instead of yfinance/FMP).
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))        # examples/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # repo root

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from midcap_scan import aggregate_curve, run_scan  # noqa: E402 - sibling example
from vpts import DataFetchError  # noqa: E402

RAW = "https://raw.githubusercontent.com/yumoxu/stocknet-dataset/master/price/raw/{}.csv"

# 20 names across 9 industries (all present in stocknet-dataset).
GITHUB_TICKERS: list[tuple[str, str]] = [
    ("AAPL", "Technology"), ("MSFT", "Technology"), ("INTC", "Technology"),
    ("JPM", "Financials"), ("BAC", "Financials"), ("C", "Financials"),
    ("JNJ", "Healthcare"), ("PFE", "Healthcare"), ("UNH", "Healthcare"),
    ("AMZN", "Consumer Disc."), ("HD", "Consumer Disc."), ("MCD", "Consumer Disc."),
    ("KO", "Consumer Staples"), ("PG", "Consumer Staples"),
    ("XOM", "Energy"), ("SLB", "Energy"),
    ("CAT", "Industrials"), ("BA", "Industrials"),
    ("NEE", "Utilities"), ("VZ", "Telecom"),
]


def github_loader(window: Optional[tuple[str, str]] = None):
    """Return ``load(symbol) -> df`` backed by raw GitHub CSVs (split-adjusted)."""
    import requests  # bundled via yfinance

    def load(symbol: str) -> pd.DataFrame:
        resp = requests.get(RAW.format(symbol), timeout=30)
        if resp.status_code != 200:
            raise DataFetchError(f"{symbol}: HTTP {resp.status_code} from GitHub")
        raw = pd.read_csv(io.StringIO(resp.text))
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw = raw.set_index("Date").sort_index()
        # Back-adjust OHLC for splits/dividends via the Adj Close / Close ratio.
        factor = (raw["Adj Close"] / raw["Close"]).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1.0)
        out = pd.DataFrame({
            "Open": raw["Open"] * factor, "High": raw["High"] * factor,
            "Low": raw["Low"] * factor, "Close": raw["Adj Close"],
            "Volume": raw["Volume"].astype(float),
        })
        if window:
            out = out.loc[window[0]:window[1]]
        return out

    return load


def _report(results: pd.DataFrame) -> None:
    if results.empty:
        print("No results."); return
    print("\nPer-ticker total return % by style:")
    print(results.pivot_table(index=["sector", "symbol"], columns="style",
                              values="return_%").to_string())
    print("\nPer-style aggregate:")
    print(results.groupby("style").agg(
        mean_return_pct=("return_%", "mean"),
        median_return_pct=("return_%", "median"),
        pct_profitable=("return_%", lambda s: round((s > 0).mean() * 100, 1)),
        mean_sharpe=("Sharpe", "mean"),
        total_trades=("trades", "sum"),
    ).round(2).to_string())
    print("\nBreakout return % by sector (mean):")
    bo = results[results["style"] == "breakout"]
    print(bo.groupby("sector")["return_%"].mean().round(1)
          .sort_values(ascending=False).to_string())


def _plot(curves: dict, out_dir: str, label: str) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        print("  (plotly not installed — skipping plot)"); return
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    color = {"reversion": "#ef5350", "breakout": "#26a69a"}
    for style, per_sym in curves.items():
        agg = aggregate_curve(per_sym)
        if agg is not None:
            x = agg.index.to_pydatetime() if isinstance(agg.index, pd.DatetimeIndex) else agg.index
            fig.add_trace(go.Scatter(x=x, y=agg.to_numpy(), name=style,
                                     line=dict(color=color.get(style), width=2.2)))
    fig.add_hline(y=100, line=dict(color="#90a4ae", width=1, dash="dash"))
    fig.update_layout(template="plotly_dark", height=440, paper_bgcolor="#0e1117",
                      plot_bgcolor="#0e1117", legend=dict(orientation="h", y=1.02),
                      title=f"Equal-weight aggregate equity — {label} (start=100)")
    path = out / "github_aggregate_equity.png"
    fig.write_image(str(path), width=1200, height=440, scale=2)
    print(f"  wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="No-key GitHub-data backtest sweep (20 stocks).")
    ap.add_argument("--window", help="date slice 'YYYY-MM-DD:YYYY-MM-DD' (default: full history)")
    ap.add_argument("--stride", type=int, default=2, help="recompute signal every N flat bars")
    ap.add_argument("--style", choices=("reversion", "breakout"), help="limit to one style")
    ap.add_argument("--plot", metavar="DIR", help="render the aggregate equity chart into DIR")
    args = ap.parse_args()

    window = None
    label = "2012-2017 (full)"
    if args.window:
        parts = args.window.split(":")
        if len(parts) != 2:
            ap.error("--window must look like 2015-01-01:2016-06-30")
        window = (parts[0], parts[1])
        label = f"{parts[0]} → {parts[1]}"

    styles = (args.style,) if args.style else ("reversion", "breakout")
    print(f"GitHub-data sweep — {len(GITHUB_TICKERS)} stocks, window={label}, "
          f"styles={styles} … (fetching free data from raw.githubusercontent.com)\n")
    results, curves = run_scan(github_loader(window), GITHUB_TICKERS,
                               styles=styles, recompute_stride=args.stride)
    _report(results)
    if args.plot:
        _plot(curves, args.plot, label)
    print("\nNote: real but in-sample data; large-caps; the value is the cross-sectional, "
          "regime-dependent style comparison — not a forward guarantee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
