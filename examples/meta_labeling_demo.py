"""Triple-barrier meta-labeling, evaluated out-of-sample with CPCV (no API key).

The primary model = the confluence `bias` (side). Triple-barrier labels say
whether each side-bet won; a logistic meta-model then learns *whether to take*
the signal, and CPCV asks — honestly — whether filtering by the meta-model
improves OOS precision and per-trade return vs taking every primary signal.

    python examples/meta_labeling_demo.py
    python examples/meta_labeling_demo.py --horizon 20 --pt 2 --sl 2 --threshold 0.55 --plot .

Honest scope: survivorship-biased large-caps; gross of costs (costs would only
lower the bar). This asks whether meta-labeling adds value, not whether to trade.
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
    DataFetchError,
    build_meta_dataset,
    cpcv_meta_eval,
)

DEFAULT_BASKET = [t for t, _ in GITHUB_TICKERS][:6]


def main() -> int:
    ap = argparse.ArgumentParser(description="Triple-barrier meta-labeling, CPCV-evaluated.")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pt", type=float, default=2.0, help="profit-take, x volatility")
    ap.add_argument("--sl", type=float, default=2.0, help="stop-loss, x volatility")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_BASKET)
    ap.add_argument("--plot", metavar="DIR", help="render the per-fold return-improvement histogram")
    args = ap.parse_args()

    load = github_loader()
    print(f"Meta-labeling — lookback={args.lookback}, horizon={args.horizon}, "
          f"barriers ±{args.pt}/{args.sl}×vol, threshold={args.threshold} — "
          f"{len(args.tickers)} names\n")

    pooled_impr: list[float] = []
    precisions: list[float] = []
    base_rates: list[float] = []
    aucs: list[float] = []
    for sym in args.tickers:
        try:
            df = load(sym)
            ds = build_meta_dataset(df, lookback=args.lookback, horizon=args.horizon,
                                    stride=args.stride, pt_mult=args.pt, sl_mult=args.sl,
                                    symbol=sym, interval="1d")
            res = cpcv_meta_eval(ds, threshold=args.threshold, l2=args.l2)
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")
            continue
        pooled_impr.extend(res.fold_improvements)
        precisions.append(res.oos_precision_mean)
        base_rates.append(res.base_win_rate)
        aucs.append(res.oos_auc_mean)
        print(res.summary())
        print()

    if pooled_impr:
        imp = np.array(pooled_impr, dtype=float)
        print("#" * 56)
        print(f"POOLED across {len(precisions)} names — {imp.size} OOS folds")
        print(f"  Base win rate   : {np.nanmean(base_rates) * 100:.1f}%  ->  "
              f"meta precision {np.nanmean(precisions) * 100:.1f}%")
        print(f"  Meta AUC        : {np.nanmean(aucs):.3f}  (0.5 = no skill)")
        print(f"  Return Δ/trade  : {imp.mean() * 100:+.3f}%  "
              f"(meta beats primary in {(imp > 0).mean() * 100:.0f}% of folds)")
        helps = (np.nanmean(precisions) > np.nanmean(base_rates) + 0.01
                 and imp.mean() > 0 and np.nanmean(aucs) > 0.52)
        print(f"  VERDICT         : "
              f"{'meta-labeling ADDS value' if helps else 'meta-labeling does NOT help'}")
        if args.plot:
            _plot_hist(imp, args.plot)
    print("\nNote: survivorship-biased universe, gross of costs — a validity check, "
          "not a tradeable result.")
    return 0


def _plot_hist(imp: np.ndarray, out_dir: str) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        print("  (plotly not installed — skipping plot)"); return
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(go.Histogram(x=imp * 100, marker_color="#ab47bc", nbinsx=40))
    fig.add_vline(x=0, line=dict(color="#ef5350", width=1, dash="dash"))
    fig.add_vline(x=float(imp.mean() * 100), line=dict(color="#ffca28", width=1.5),
                  annotation_text=f"mean {imp.mean() * 100:+.2f}%")
    fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="#0e1117",
                      plot_bgcolor="#0e1117",
                      title=f"Meta vs primary return Δ per CPCV fold (n={imp.size})",
                      xaxis_title="meta − primary return per trade, %", yaxis_title="count")
    path = out / "meta_return_improvement.png"
    fig.write_image(str(path), width=1100, height=420, scale=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
