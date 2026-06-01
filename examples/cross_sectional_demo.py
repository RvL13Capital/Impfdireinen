"""Do cross-sectional rank factors carry OOS signal the per-name models missed?

Everything before scored each name on its own series. This ranks names **against
each other each rebalance day** — 1-month reversal, 12-1 momentum, 60-day vol
(low-vol anomaly) and a volume-trend proxy — combines them with a purged-CPCV
ridge, and scores the per-date **rank IC** (Spearman of prediction vs forward
return across names). Two honesty guards:

  1. Each raw factor's model-free pooled rank IC (is any single factor informative?).
  2. A **within-date label-shuffle permutation test** on the combined OOS IC — the
     forward returns are permuted among the names on each date, so a combination
     that can't beat its own shuffled null is reported as no edge.

    python examples/cross_sectional_demo.py
    python examples/cross_sectional_demo.py --horizon 20 --rebalance 5 --perms 200 --plot .

Honest scope: 20 survivor large-caps, 2012-2017 daily — a thin, survivorship-biased
cross-section. This asks whether cross-sectional rank holds ANY OOS information,
not whether to trade it.
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
    build_cross_sectional_panel,
    cross_sectional_ic_eval,
)
from vpts.ml.cross_sectional import permutation_test_cross_sectional  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-sectional rank-factor OOS test (no key).")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--rebalance", type=int, default=5)
    ap.add_argument("--min-names", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--tickers", nargs="*", default=[t for t, _ in GITHUB_TICKERS])
    ap.add_argument("--plot", metavar="DIR", help="render the null histogram")
    args = ap.parse_args()

    load = github_loader()
    frames = {}
    for sym in args.tickers:
        try:
            frames[sym] = load(sym)
        except DataFetchError as exc:
            print(f"  ! {sym}: skipped ({exc})")
    print(f"Cross-sectional rank factors — {len(frames)} names, horizon={args.horizon}d, "
          f"rebalance={args.rebalance}d, ridge α={args.alpha}\n")
    if len(frames) < args.min_names:
        print("Not enough names loaded for a cross-section."); return 1

    panel = build_cross_sectional_panel(
        frames, horizon=args.horizon, rebalance=args.rebalance, min_names=args.min_names)
    cv = CombinatorialPurgedCV(n_groups=args.n_groups, n_test_groups=args.n_test,
                               purge=panel.purge_dates, embargo_pct=0.01)
    res = cross_sectional_ic_eval(panel, cv=cv, alpha=args.alpha)
    print(res.summary())

    print(f"\nWithin-date permutation test ({args.perms} shuffles) …")
    pt = permutation_test_cross_sectional(panel, cv=cv, n_permutations=args.perms, seed=0)
    print(f"  Real combined IC : {pt.real_ic:+.3f}")
    print(f"  Null combined IC : mean {pt.null_ic_mean:+.3f}")
    print(f"  p-value          : {pt.p_value:.3f}  "
          f"({'SIGNIFICANT' if pt.p_value < 0.05 else 'not significant'})")

    sig = pt.p_value < 0.05 and res.combined_oos_ic_mean > 0.01
    verdict = ("cross-sectional rank carries REAL, significant OOS signal" if sig
               else "NO robust cross-sectional OOS edge — real IC sits inside the null")
    print(f"\n  VERDICT          : {verdict}")

    if args.plot:
        _plot_null(panel, cv, res, pt, args.plot, args.perms, args.alpha)
    print(f"\nNote: survivorship-biased cross-section ({panel.n_names} survivors, "
          f"≥{args.min_names}/date); a validity check on OOS information content, "
          "not a tradeable result.")
    return 0


def _plot_null(panel, cv, res, pt, out_dir, perms, alpha) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        print("  (plotly not installed — skipping plot)"); return
    # Recompute the null array for the histogram (cheap; reuses the same CV).
    rng = np.random.default_rng(1)
    from vpts.ml.cross_sectional import _date_groups
    groups = _date_groups(panel.date_id, panel.n_dates)
    null = []
    from vpts.ml.models import CrossSectionalPanel
    for _ in range(perms):
        yp = panel.y.copy()
        for g in groups:
            if g.size > 1:
                yp[g] = panel.y[g][rng.permutation(g.size)]
        sh = CrossSectionalPanel(
            X=panel.X, y=yp, date_id=panel.date_id, feature_names=panel.feature_names,
            horizon=panel.horizon, rebalance=panel.rebalance, n_dates=panel.n_dates,
            symbols=panel.symbols)
        try:
            null.append(cross_sectional_ic_eval(sh, cv, alpha).combined_oos_ic_mean)
        except ValueError:
            continue
    null = np.array(null, dtype=float)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(go.Histogram(x=null, marker_color="#42a5f5", nbinsx=40,
                                 name="within-date null"))
    fig.add_vline(x=res.combined_oos_ic_mean, line=dict(color="#ffca28", width=2),
                  annotation_text=f"real {res.combined_oos_ic_mean:+.3f} (p={pt.p_value:.3f})")
    fig.add_vline(x=0, line=dict(color="#ef5350", width=1, dash="dash"))
    fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="#0e1117",
                      plot_bgcolor="#0e1117",
                      title=f"Cross-sectional combined OOS rank IC vs within-date null "
                            f"(n={null.size})",
                      xaxis_title="combined out-of-sample rank IC", yaxis_title="count")
    path = out / "cross_sectional_null.png"
    fig.write_image(str(path), width=1100, height=420, scale=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
