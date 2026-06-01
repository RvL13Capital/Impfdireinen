"""Do the structural microstructure features carry out-of-sample signal?

Runs the full structural feature matrix — synthetic delta (net + at-POC), profile
skew/kurtosis, POC location, value-area compression z-score, POC-migration slope,
cost-basis migration, ledges, poor-highs and one-hot P/b/B shapes — through the
*same* purged-CPCV harness that judged every prior experiment, with:

  1. a head-to-head vs the single-feature baseline (synthetic delta at POC), and
  2. a label-shuffle permutation test on the pooled OOS IC.

    python examples/structural_demo.py
    python examples/structural_demo.py --horizon 20 --stride 3 --perms 200 --plot .

Honest scope: survivorship-biased large-caps, 2012-2017 daily. This asks whether
the structural math holds ANY OOS information, not whether to trade it. No claim
about win rates is made before the permutation test reports.
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
    STRUCTURAL_FEATURES,
    build_structural_dataset,
    cpcv_factor_eval,
)
from vpts.ml.factor_model import permutation_test_factor  # noqa: E402
from vpts.ml.models import FactorDataset  # noqa: E402

DEFAULT_BASKET = [t for t, _ in GITHUB_TICKERS][:8]


def _pooled(folds: list[np.ndarray]) -> float:
    return float(np.concatenate(folds).mean()) if folds else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Structural-feature OOS edge test (no key).")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_BASKET)
    ap.add_argument("--plot", metavar="DIR", help="render the pooled null histogram")
    args = ap.parse_args()

    load = github_loader()
    print(f"Structural features — lookback={args.lookback}, horizon={args.horizon}, "
          f"stride={args.stride}, ridge α={args.alpha} — {len(args.tickers)} names")
    print(f"  features ({len(STRUCTURAL_FEATURES)}): {', '.join(STRUCTURAL_FEATURES)}\n")

    built: list[tuple[str, FactorDataset, CombinatorialPurgedCV]] = []
    pooled: list[np.ndarray] = []
    base_ic: list[float] = []
    weights_acc: list[np.ndarray] = []
    for sym in args.tickers:
        try:
            df = load(sym)
            ds = build_structural_dataset(df, lookback=args.lookback, horizon=args.horizon,
                                          stride=args.stride, symbol=sym, interval="1d")
            cv = CombinatorialPurgedCV(n_groups=args.n_groups, n_test_groups=args.n_test,
                                       purge=ds.purge_samples, embargo_pct=0.01)
            res = cpcv_factor_eval(ds, cv=cv, alpha=args.alpha)
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")
            continue
        built.append((sym, ds, cv))
        pooled.append(np.array(res.fold_ics, dtype=float))
        weights_acc.append(np.array(res.mean_weights, dtype=float))
        if np.isfinite(res.baseline_ic_mean):
            base_ic.append(res.baseline_ic_mean)
        print(f"  {sym:5s}  OOS IC {res.oos_ic_mean:+.3f}  (σ {res.oos_ic_std:.2f}, "
              f"{res.pct_folds_positive_ic:.0f}% folds>0)   "
              f"delta@POC baseline {res.baseline_ic_mean:+.3f}")

    if not built:
        print("\nNo usable names."); return 1

    real = _pooled(pooled)
    w = np.mean(np.vstack(weights_acc), axis=0)
    n_folds = int(np.concatenate(pooled).size)
    order = np.argsort(-np.abs(w))
    print("\n" + "#" * 60)
    print(f"POOLED across {len(built)} names — {n_folds} OOS folds")
    print(f"  Structural OOS IC : {real:+.3f}")
    if base_ic:
        print(f"  delta@POC IC      : {np.mean(base_ic):+.3f}  (single-feature baseline)")
    print("  Top weights       : "
          + ", ".join(f"{STRUCTURAL_FEATURES[i]} {w[i]:+.2f}" for i in order[:6]))

    # ---- pooled label-shuffle permutation test ---------------------------- #
    print(f"\nPooled permutation test ({args.perms} shuffles) …")
    rng = np.random.default_rng(0)
    null = np.empty(args.perms, dtype=float)
    for p in range(args.perms):
        folds: list[np.ndarray] = []
        for _sym, ds, cv in built:
            perm = rng.permutation(len(ds))
            shuf = FactorDataset(
                X=ds.X, y=ds.y[perm], baseline=ds.baseline, feature_names=ds.feature_names,
                horizon=ds.horizon, stride=ds.stride, symbol=ds.symbol)
            try:
                folds.append(np.array(cpcv_factor_eval(shuf, cv=cv, alpha=args.alpha).fold_ics, float))
            except ValueError:
                continue
        null[p] = _pooled(folds)
    null = null[np.isfinite(null)]
    p_value = float((np.sum(null >= real) + 1) / (null.size + 1))
    print(f"  Real pooled IC    : {real:+.3f}")
    print(f"  Null pooled IC    : mean {null.mean():+.3f}  σ {null.std():.3f}  "
          f"(95th pct {np.quantile(null, 0.95):+.3f})")
    print(f"  p-value           : {p_value:.3f}  "
          f"({'SIGNIFICANT' if p_value < 0.05 else 'not significant'})")

    verdict = ("structural features carry REAL, significant OOS signal"
               if (p_value < 0.05 and real > 0.02)
               else "NO robust OOS edge — real IC sits inside the shuffled null")
    print(f"\n  VERDICT           : {verdict}")

    if args.plot:
        _plot(null, real, p_value, args.plot)
    print("\nNote: survivorship-biased universe; a validity check on whether the "
          "structural math holds OOS information, not a tradeable result.")
    return 0


def _plot(null: np.ndarray, real: float, p_value: float, out_dir: str) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        print("  (plotly not installed — skipping plot)"); return
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(go.Histogram(x=null, marker_color="#42a5f5", nbinsx=40, name="shuffled null"))
    fig.add_vline(x=real, line=dict(color="#ffca28", width=2),
                  annotation_text=f"real {real:+.3f} (p={p_value:.3f})")
    fig.add_vline(x=0, line=dict(color="#ef5350", width=1, dash="dash"))
    fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="#0e1117",
                      plot_bgcolor="#0e1117",
                      title=f"Structural-feature pooled OOS IC vs label-shuffled null (n={null.size})",
                      xaxis_title="pooled out-of-sample IC", yaxis_title="count")
    path = out / "structural_oos_ic.png"
    fig.write_image(str(path), width=1100, height=420, scale=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
