"""Learn confluence-factor weights and evaluate them out-of-sample with CPCV.

For each name it builds (factor features -> forward return) samples from the free
GitHub data, fits a ridge model on each CPCV train fold, and scores the held-out
(purged + embargoed) test fold. The headline is the distribution of out-of-sample
IC (does the learned model actually predict forward returns?), with the hand-set
weights' IC as a baseline.

    python examples/factor_model_demo.py
    python examples/factor_model_demo.py --horizon 20 --stride 3 --alpha 1 --plot .

Honest scope: real but survivorship-biased large-caps; this asks "is there ANY
learnable, out-of-sample factor edge here?" — not whether to trade it.
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
    CombinatorialPurgedCV,
    DataFetchError,
    build_factor_dataset,
    cpcv_factor_eval,
)

DEFAULT_BASKET = [t for t, _ in GITHUB_TICKERS][:6]


def main() -> int:
    ap = argparse.ArgumentParser(description="Learned factor weights, CPCV-evaluated (no key).")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_BASKET)
    ap.add_argument("--plot", metavar="DIR", help="render a pooled OOS-IC histogram")
    args = ap.parse_args()

    load = github_loader()
    print(f"Learned factor weights — lookback={args.lookback}, horizon={args.horizon}, "
          f"stride={args.stride}, ridge α={args.alpha} — {len(args.tickers)} names\n")

    pooled_ic: list[float] = []
    weights_acc: list[np.ndarray] = []
    base_ic: list[float] = []
    names = None
    for sym in args.tickers:
        try:
            df = load(sym)
            ds = build_factor_dataset(df, lookback=args.lookback, horizon=args.horizon,
                                      stride=args.stride, symbol=sym, interval="1d")
            cv = CombinatorialPurgedCV(n_groups=args.n_groups, n_test_groups=args.n_test,
                                       purge=ds.purge_samples, embargo_pct=0.01)
            res = cpcv_factor_eval(ds, cv=cv, alpha=args.alpha)
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")
            continue
        names = res.feature_names
        pooled_ic.extend(res.fold_ics)
        weights_acc.append(np.array(res.mean_weights, dtype=float))
        if np.isfinite(res.baseline_ic_mean):
            base_ic.append(res.baseline_ic_mean)
        print(res.summary())
        print()

    if pooled_ic:
        ic = np.array(pooled_ic, dtype=float)
        w = np.mean(np.vstack(weights_acc), axis=0)
        print("#" * 56)
        print(f"POOLED across {len(weights_acc)} names — {ic.size} OOS folds")
        print(f"  OOS IC          : mean {ic.mean():+.3f}  median {np.median(ic):+.3f}  "
              f"σ {ic.std():.3f}  ({(ic > 0).mean() * 100:.0f}% folds > 0)")
        if names is not None:
            print("  Mean weights    : "
                  + ", ".join(f"{n} {v:+.2f}" for n, v in zip(names, w)))
        if base_ic:
            print(f"  Baseline IC     : {np.mean(base_ic):+.3f}  (hand-set weights)")
        verdict = ("NO learnable OOS edge (IC ~ 0)" if abs(ic.mean()) < 0.02
                   else ("weak positive OOS signal" if ic.mean() > 0
                         else "anti-predictive OOS"))
        print(f"  VERDICT         : {verdict}")
        if args.plot:
            _plot_hist(ic, args.plot)
    print("\nNote: survivorship-biased universe; this is a validity check on whether "
          "the factors carry out-of-sample information, not a tradeable result.")
    return 0


def _plot_hist(ic: np.ndarray, out_dir: str) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        print("  (plotly not installed — skipping plot)"); return
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(go.Histogram(x=ic, marker_color="#42a5f5", nbinsx=40))
    fig.add_vline(x=0, line=dict(color="#ef5350", width=1, dash="dash"))
    fig.add_vline(x=float(ic.mean()), line=dict(color="#ffca28", width=1.5),
                  annotation_text=f"mean {ic.mean():+.3f}")
    fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="#0e1117",
                      plot_bgcolor="#0e1117",
                      title=f"Learned-factor OOS IC across CPCV folds (n={ic.size})",
                      xaxis_title="out-of-sample IC", yaxis_title="count")
    path = out / "factor_oos_ic.png"
    fig.write_image(str(path), width=1100, height=420, scale=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
