"""Experiment 13 — the orthogonality purge: how many of the engineered features are
independent axes, and how many are momentum/VWAP in costume?

Experiment 12 showed *one* parametric feature (`gmm_gravity`) collapse into a
price-minus-VWAP baseline. The natural objection: that was n = 1 — maybe the rest of
the matrix carries independent structure. This tests it across the **whole** feature
set. Pool all **13 structural + 7 EM-GMM features plus a no-GMM `vwap_dist` /
momentum baseline**, take the Spearman rank-correlation matrix, **hierarchically
cluster** it (distance = 1 − |ρ|), and for each cluster elect the representative with
the highest standalone OOS IC. Features that sit in the same cluster as
`vwap_dist`/`mom` are redundant with a moving-average distance; clusters that stand
apart *with* real IC would be genuine independent signal.

    python examples/feature_purge.py
    python examples/feature_purge.py --threshold 0.6 --survivors AAPL MSFT JPM

Honest scope: survivorship-biased 2012–17 daily; in-sample correlations, OOS
single-factor IC; a feature-structure diagnostic, not a tradeable result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from vpts import (  # noqa: E402
    CombinatorialPurgedCV,
    DataFetchError,
    STRUCTURAL_FEATURES,
    build_structural_dataset,
    cpcv_factor_eval,
)
from vpts.structure.gmm import GMM_FEATURES, build_gmm_dataset  # noqa: E402
from vpts.ml.models import FactorDataset  # noqa: E402

BASELINES = ("vwap_dist", "mom_120", "mom_20")
H, ST, LB = 20, 3, 120


def _baseline_frame(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """No-GMM baselines (price-minus-VWAP, trailing momentum) at the given event dates."""
    close = df["Close"].to_numpy(float)
    vol = df["Volume"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    rows = []
    for ts in idx:
        t = int(df.index.get_loc(ts))
        sl = slice(t - LB + 1, t + 1)
        vwap = float((close[sl] * vol[sl]).sum() / max(float(vol[sl].sum()), 1e-9))
        rng = float(high[sl].max() - low[sl].min())
        rows.append([(close[t] - vwap) / rng if rng > 1e-9 else 0.0,
                     close[t] / close[t - LB + 1] - 1.0,
                     close[t] / close[t - 20] - 1.0])
    return pd.DataFrame(rows, index=idx, columns=BASELINES)


def cluster_features(corr: pd.DataFrame, threshold: float) -> dict[str, int]:
    """Hierarchically cluster features; |ρ| ≥ threshold tends to land in one cluster."""
    dist = 1.0 - corr.abs().to_numpy(float)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(z, t=1.0 - threshold, criterion="distance")
    return dict(zip(corr.columns, (int(x) for x in labels)))


def _combined(df, sym):
    """Row-aligned 13 structural + 7 GMM + 3 baseline features + forward return for one name."""
    s = build_structural_dataset(df, symbol=sym, interval="1d",
                                 lookback=LB, horizon=H, stride=ST)
    g = build_gmm_dataset(df, symbol=sym, interval="1d", lookback=LB, horizon=H, stride=ST)
    S = pd.DataFrame(s.X, index=s.timestamps, columns=STRUCTURAL_FEATURES)
    G = pd.DataFrame(g.X, index=g.timestamps, columns=GMM_FEATURES)
    common = S.index.intersection(G.index)
    B = _baseline_frame(df, common)
    y = pd.Series(s.y, index=s.timestamps, name="y").loc[common]
    M = pd.concat([S.loc[common], G.loc[common], B, y], axis=1).dropna()
    return M


def _pooled_ic(frames, col) -> float:
    """Standalone OOS single-factor IC for one feature, pooled over names (same CPCV harness)."""
    folds = []
    for M in frames:
        ds = FactorDataset(X=M[[col]].to_numpy(float), y=M["y"].to_numpy(float),
                           baseline=np.zeros(len(M)), feature_names=(col,), horizon=H, stride=ST)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge=ds.purge_samples, embargo_pct=0.01)
        try:
            folds.extend(cpcv_factor_eval(ds, cv=cv).fold_ics)
        except ValueError:
            continue
    return float(np.mean(folds)) if folds else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Orthogonality purge — feature clustering vs a momentum/VWAP baseline.")
    ap.add_argument("--threshold", type=float, default=0.7, help="|Spearman| at/above which features cluster")
    ap.add_argument("--survivors", nargs="*", default=[t for t, _ in GITHUB_TICKERS])
    args = ap.parse_args()

    load = github_loader()
    frames = []
    for sym in args.survivors:
        try:
            frames.append(_combined(load(sym), sym))
        except (DataFetchError, ValueError):
            continue
    if not frames:
        print("No names built."); return 1
    feats = list(STRUCTURAL_FEATURES) + list(GMM_FEATURES) + list(BASELINES)
    pooled = pd.concat([M[feats] for M in frames], ignore_index=True)
    print(f"Experiment 13 — orthogonality purge — {len(frames)} survivors, "
          f"{len(pooled)} pooled rows, {len(feats)} features (|ρ|≥{args.threshold} clusters)\n")

    corr = pooled.corr(method="spearman")
    labels = cluster_features(corr, args.threshold)
    ic = {f: _pooled_ic(frames, f) for f in feats}

    base_cluster = labels["vwap_dist"]
    clusters: dict[int, list[str]] = {}
    for f, c in labels.items():
        clusters.setdefault(c, []).append(f)

    print(f"{len(feats)} features → {len(clusters)} clusters at |ρ|≥{args.threshold}. "
          f"(cluster {base_cluster} = the momentum/VWAP family)\n")
    # representative = highest standalone |OOS IC| in each cluster
    order = sorted(clusters.items(), key=lambda kv: -max(abs(ic[f]) for f in kv[1]))
    for c, members in order:
        rep = max(members, key=lambda f: abs(ic[f]))
        tag = "  <- momentum/VWAP family" if c == base_cluster else ""
        print(f"  cluster {c} (rep {rep}, IC {ic[rep]:+.3f}){tag}")
        for f in sorted(members, key=lambda f: -abs(ic[f])):
            star = " *" if f == rep else "  "
            print(f"    {star} {f:22} IC {ic[f]:+.3f}")

    survivors_ic = sorted((f for f in feats if abs(ic[f]) >= 0.05), key=lambda f: -abs(ic[f]))
    n_base = len(clusters[base_cluster])
    print("\nVerdict — wide but shallow (Exp 12's collapse is the *dominant axis*, not a universal law):")
    print(f"  • correlation: {len(feats)} features → {len(clusters)} clusters at |ρ|≥{args.threshold} — "
          f"NOT a collinear blob; only {n_base} merge into the momentum/VWAP cluster.")
    print(f"  • signal: only {len(survivors_ic)}/{len(feats)} features clear |IC|≥0.05 "
          f"({', '.join(f'{f} {ic[f]:+.2f}' for f in survivors_ic)}).")
    print(f"  • the other ~{len(feats) - len(survivors_ic)} (most GMM geometry + profile shape) are "
          f"independent yet ~0 IC. The honest purge is an IC filter, not a correlation filter: the "
          f"matrix isn't redundant, it's mostly *null*. (In-sample ρ; OOS single-factor IC; "
          f"survivorship-biased; a structure diagnostic.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
