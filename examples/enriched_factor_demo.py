"""Does a *richer* feature set carry out-of-sample signal the coarse factors miss?

This is option-#2 of the validation arc: keep hunting with **genuinely new inputs**.
On top of the four confluence factors it adds time-series **momentum** (20/60/12-1),
**volatility** (20-bar σ, ATR/price), a **volume-trend** microstructure ratio and a
continuous **distance-to-POC** — eleven features in all — and pushes them through the
*same* purged-CPCV harness that judged everything else.

Two honesty guards beyond a single IC number:

  1. Head-to-head vs the 4-factor baseline (did the extra inputs add anything?).
  2. A **pooled label-shuffle permutation test**: each name's targets are shuffled
     independently and the whole pooled OOS-IC is recomputed; the p-value is how
     often noise matches the real pooled IC. An edge that can't clear its own
     shuffled null is not an edge.

    python examples/enriched_factor_demo.py
    python examples/enriched_factor_demo.py --horizon 20 --stride 3 --perms 200 --plot .

Honest scope: survivorship-biased large-caps, 2012-2017 daily. This asks "is there
ANY learnable OOS signal in richer features here?", not "trade this".
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
    build_enriched_factor_dataset,
    build_factor_dataset,
    cpcv_factor_eval,
)

DEFAULT_BASKET = [t for t, _ in GITHUB_TICKERS][:8]


def _pooled_ic(per_name_folds: list[np.ndarray]) -> float:
    """Pooled mean OOS IC = mean over every fold of every name."""
    if not per_name_folds:
        return float("nan")
    return float(np.concatenate(per_name_folds).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="Enriched-feature OOS edge test (no key).")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--perms", type=int, default=200, help="pooled permutation count")
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_BASKET)
    ap.add_argument("--plot", metavar="DIR", help="render the pooled null histogram")
    args = ap.parse_args()

    load = github_loader()
    print(f"Enriched factors — lookback={args.lookback}, horizon={args.horizon}, "
          f"stride={args.stride}, ridge α={args.alpha} — {len(args.tickers)} names\n")

    # Build every name's dataset + a shared CV once; reuse for the permutation null.
    built: list[tuple[str, object, CombinatorialPurgedCV]] = []
    enriched_folds: list[np.ndarray] = []
    plain_folds: list[np.ndarray] = []
    weights_acc: list[np.ndarray] = []
    base_ic: list[float] = []
    names = None
    for sym in args.tickers:
        try:
            df = load(sym)
            ds = build_enriched_factor_dataset(
                df, lookback=args.lookback, horizon=args.horizon,
                stride=args.stride, symbol=sym, interval="1d")
            cv = CombinatorialPurgedCV(n_groups=args.n_groups, n_test_groups=args.n_test,
                                       purge=ds.purge_samples, embargo_pct=0.01)
            res = cpcv_factor_eval(ds, cv=cv, alpha=args.alpha)
            # 4-factor baseline on the SAME name for a head-to-head.
            ds0 = build_factor_dataset(df, lookback=args.lookback, horizon=args.horizon,
                                       stride=args.stride, symbol=sym, interval="1d")
            res0 = cpcv_factor_eval(
                ds0, cv=CombinatorialPurgedCV(
                    n_groups=args.n_groups, n_test_groups=args.n_test,
                    purge=ds0.purge_samples, embargo_pct=0.01), alpha=args.alpha)
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")
            continue
        names = res.feature_names
        built.append((sym, ds, cv))
        enriched_folds.append(np.array(res.fold_ics, dtype=float))
        plain_folds.append(np.array(res0.fold_ics, dtype=float))
        weights_acc.append(np.array(res.mean_weights, dtype=float))
        if np.isfinite(res.baseline_ic_mean):
            base_ic.append(res.baseline_ic_mean)
        print(f"  {sym:5s}  enriched OOS IC {res.oos_ic_mean:+.3f}  "
              f"(σ {res.oos_ic_std:.2f}, {res.pct_folds_positive_ic:.0f}% folds>0)   "
              f"|  4-factor {res0.oos_ic_mean:+.3f}   "
              f"Δ {res.oos_ic_mean - res0.oos_ic_mean:+.3f}")

    if not built:
        print("\nNo usable names."); return 1

    real_enriched = _pooled_ic(enriched_folds)
    real_plain = _pooled_ic(plain_folds)
    w = np.mean(np.vstack(weights_acc), axis=0)
    n_folds = int(np.concatenate(enriched_folds).size)

    print("\n" + "#" * 60)
    print(f"POOLED across {len(built)} names — {n_folds} OOS folds")
    print(f"  Enriched OOS IC : {real_enriched:+.3f}")
    print(f"  4-factor OOS IC : {real_plain:+.3f}   "
          f"(enriched adds {real_enriched - real_plain:+.3f})")
    if names is not None:
        order = np.argsort(-np.abs(w))
        print("  Top weights     : "
              + ", ".join(f"{names[i]} {w[i]:+.2f}" for i in order[:5]))
    if base_ic:
        print(f"  Baseline IC     : {np.mean(base_ic):+.3f}  (hand-set bias_score)")

    # ---- pooled label-shuffle permutation test ---------------------------- #
    print(f"\nPooled permutation test ({args.perms} shuffles) …")
    rng = np.random.default_rng(0)
    null = np.empty(args.perms, dtype=float)
    from vpts.ml.models import FactorDataset
    for p in range(args.perms):
        folds: list[np.ndarray] = []
        for _sym, ds, cv in built:
            perm = rng.permutation(len(ds))
            shuf = FactorDataset(
                X=ds.X, y=ds.y[perm], baseline=ds.baseline,
                feature_names=ds.feature_names, horizon=ds.horizon,
                stride=ds.stride, symbol=ds.symbol)
            try:
                r = cpcv_factor_eval(shuf, cv=cv, alpha=args.alpha)
            except ValueError:
                continue
            folds.append(np.array(r.fold_ics, dtype=float))
        null[p] = _pooled_ic(folds)
    null = null[np.isfinite(null)]
    p_value = float((np.sum(null >= real_enriched) + 1) / (null.size + 1))
    print(f"  Real pooled IC  : {real_enriched:+.3f}")
    print(f"  Null pooled IC  : mean {null.mean():+.3f}  σ {null.std():.3f}  "
          f"(95th pct {np.quantile(null, 0.95):+.3f})")
    print(f"  p-value         : {p_value:.3f}  "
          f"({'SIGNIFICANT' if p_value < 0.05 else 'not significant'})")

    helps = real_enriched > real_plain + 0.01
    if p_value < 0.05 and helps and real_enriched > 0.02:
        verdict = "richer features carry REAL, significant OOS signal beyond the baseline"
    elif p_value < 0.05 and real_enriched > 0.02:
        verdict = "significant OOS signal, but no clear gain over the 4-factor baseline"
    else:
        verdict = "NO robust OOS edge — real IC sits inside the shuffled null"
    print(f"\n  VERDICT         : {verdict}")

    if args.plot:
        _plot_null(null, real_enriched, p_value, args.plot)
    print("\nNote: survivorship-biased universe; a validity check on whether richer "
          "features hold OOS information, not a tradeable result.")
    return 0


def _plot_null(null: np.ndarray, real: float, p_value: float, out_dir: str) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        print("  (plotly not installed — skipping plot)"); return
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(go.Histogram(x=null, marker_color="#42a5f5", nbinsx=40,
                                 name="shuffled null"))
    fig.add_vline(x=real, line=dict(color="#ffca28", width=2),
                  annotation_text=f"real {real:+.3f} (p={p_value:.3f})")
    fig.add_vline(x=0, line=dict(color="#ef5350", width=1, dash="dash"))
    fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="#0e1117",
                      plot_bgcolor="#0e1117",
                      title=f"Enriched-factor pooled OOS IC vs label-shuffled null "
                            f"(n={null.size})",
                      xaxis_title="pooled out-of-sample IC", yaxis_title="count")
    path = out / "enriched_factor_null.png"
    fig.write_image(str(path), width=1100, height=420, scale=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
