"""Combinatorial Purged CV on free GitHub data (no API key).

Evaluates the strategy out-of-sample with CPCV + embargo across a basket of
sector-diverse names, using the same free stocknet-dataset as
``github_data_scan.py``. For each ticker it prints the OOS performance
*distribution* across recombined held-out periods, then a pooled cross-sectional
distribution — the honest read on how stable the edge is.

    python examples/cpcv_demo.py
    python examples/cpcv_demo.py --style reversion --n-groups 6 --n-test 2 --plot .

Honest scope: the strategy has no fitted parameters, so CPCV here measures OOS
*dispersion/robustness*, not protection against parameter-selection overfitting,
and it does NOT fix the survivorship bias in the underlying basket.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from vpts import (  # noqa: E402
    Backtester,
    CombinatorialPurgedCV,
    CostModel,
    DataFetchError,
    SignalGenerator,
)

DEFAULT_BASKET = [t for t, _ in GITHUB_TICKERS][:8]  # keep the demo quick


def main() -> int:
    ap = argparse.ArgumentParser(description="CPCV out-of-sample evaluation (no-key data).")
    ap.add_argument("--style", choices=("reversion", "breakout"), default="breakout")
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--purge", type=int, default=5)
    ap.add_argument("--embargo", type=float, default=0.01)
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_BASKET)
    ap.add_argument("--plot", metavar="DIR", help="render a pooled path-return histogram")
    args = ap.parse_args()

    load = github_loader()
    cv = CombinatorialPurgedCV(n_groups=args.n_groups, n_test_groups=args.n_test,
                               purge=args.purge, embargo_pct=args.embargo)
    bt = Backtester(
        lookback=args.lookback, recompute_stride=2,
        signal_generator=SignalGenerator(style=args.style, min_quality=50, min_abs_bias=12),
        cost_model=CostModel(slippage_bps=5.0),
    )
    print(f"CPCV [{args.style}] — N={args.n_groups}, k={args.n_test}, purge={args.purge}, "
          f"embargo={args.embargo:.0%} — {len(args.tickers)} names\n")

    pooled: list[float] = []
    for sym in args.tickers:
        try:
            df = load(sym)
            res = cv.backtest_paths(df, bt, symbol=sym, interval="1d")
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")
            continue
        pooled.extend(res.path_returns)
        print(res.summary())
        print()

    if pooled:
        arr = np.array(pooled, dtype=float)
        print("#" * 56)
        print(f"POOLED across {len(args.tickers)} names — {arr.size} OOS paths")
        print(f"  mean {arr.mean():+.2f}%  median {np.median(arr):+.2f}%  "
              f"σ {arr.std():.2f}  [{arr.min():+.2f}%, {arr.max():+.2f}%]")
        print(f"  paths profitable: {(arr > 0).mean() * 100:.0f}%")
        if args.plot:
            _plot_hist(arr, args.style, args.plot)
    print("\nNote: OOS dispersion/robustness on survivorship-biased large-caps — "
          "a validity check on the framework, not a forward guarantee.")
    return 0


def _plot_hist(arr: np.ndarray, style: str, out_dir: str) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        print("  (plotly not installed — skipping plot)"); return
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(go.Histogram(x=arr, marker_color="#26a69a", nbinsx=40))
    fig.add_vline(x=0, line=dict(color="#ef5350", width=1, dash="dash"))
    fig.add_vline(x=float(arr.mean()), line=dict(color="#ffca28", width=1.5),
                  annotation_text=f"mean {arr.mean():+.1f}%")
    fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="#0e1117",
                      plot_bgcolor="#0e1117",
                      title=f"CPCV pooled OOS path returns [{style}] (n={arr.size})",
                      xaxis_title="path return %", yaxis_title="count")
    path = out / "cpcv_path_returns.png"
    fig.write_image(str(path), width=1100, height=420, scale=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
