"""Survivorship-injection stress test for the structural-feature OOS IC.

The structural features showed the arc's first significant signal (+0.103,
p=0.005) — but on **survivors only**, and the strongest features are
"dip-is-being-accumulated" signals that survivors flatter (they recover; delisted
names don't). This injects synthetic **delisted** names (normal, then a decline to
pennies) alongside the survivors and re-runs the *same* pooled-permutation test as
the demo. If the significance collapses as the delisting rate rises, the edge was
largely survivorship — exactly what happened to meta-labeling.

    python examples/structural_survivorship.py
    python examples/structural_survivorship.py --n-delisted 12 --perms 200

Honest limitation: the delisted names are *synthetic* (no free delisted data), so
this is a sensitivity estimate, not a substitute for real point-in-time data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from survivorship_stress import synthetic_delisted_ohlcv  # noqa: E402
from vpts import (  # noqa: E402
    CombinatorialPurgedCV,
    DataFetchError,
    build_structural_dataset,
    cpcv_factor_eval,
)
from vpts.ml.models import FactorDataset  # noqa: E402

Built = list[tuple[FactorDataset, CombinatorialPurgedCV]]


def _pooled_ic(built: Built, alpha: float) -> float:
    folds: list[np.ndarray] = []
    for ds, cv in built:
        try:
            folds.append(np.array(cpcv_factor_eval(ds, cv=cv, alpha=alpha).fold_ics, float))
        except ValueError:
            continue
    return float(np.concatenate(folds).mean()) if folds else float("nan")


def _pooled_permutation(built: Built, alpha: float, n_perms: int, seed: int = 0):
    real = _pooled_ic(built, alpha)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perms):
        folds = []
        for ds, cv in built:
            perm = rng.permutation(len(ds))
            shuf = FactorDataset(X=ds.X, y=ds.y[perm], baseline=ds.baseline,
                                 feature_names=ds.feature_names, horizon=ds.horizon,
                                 stride=ds.stride, symbol=ds.symbol)
            try:
                folds.append(np.array(cpcv_factor_eval(shuf, cv=cv, alpha=alpha).fold_ics, float))
            except ValueError:
                continue
        if folds:
            null.append(float(np.concatenate(folds).mean()))
    arr = np.array(null, float)
    p = float((np.sum(arr >= real) + 1) / (arr.size + 1))
    return real, (float(arr.mean()) if arr.size else float("nan")), p


def _build(df, sym, kw) -> tuple[FactorDataset, CombinatorialPurgedCV]:
    ds = build_structural_dataset(df, symbol=sym, interval="1d", **kw)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2,
                               purge=ds.purge_samples, embargo_pct=0.01)
    return ds, cv


def main() -> int:
    ap = argparse.ArgumentParser(description="Structural survivorship-injection stress test.")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-delisted", type=int, default=10)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--survivors", nargs="*", default=[t for t, _ in GITHUB_TICKERS][:8])
    args = ap.parse_args()

    load = github_loader()
    kw = dict(lookback=args.lookback, horizon=args.horizon, stride=args.stride)
    print(f"Structural survivorship stress — {len(args.survivors)} survivors + up to "
          f"{args.n_delisted} synthetic delisted (perms={args.perms})\n")

    surv: Built = []
    for sym in args.survivors:
        try:
            surv.append(_build(load(sym), sym, kw))
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")
    dead: Built = []
    for k in range(args.n_delisted):
        try:
            dead.append(_build(synthetic_delisted_ohlcv(seed=100 + k), f"DEAD{k}", kw))
        except ValueError:
            continue
    print(f"  built {len(surv)} survivors, {len(dead)} delisted\n")
    if not surv:
        print("No survivors built."); return 1

    print(f"{'mix':>26}{'nDead':>7}{'pooledIC':>11}{'nullIC':>9}{'p':>8}")
    S = len(surv)
    for rate in (0.0, 0.05, 0.10, 0.20, 0.30):
        d = min(int(round(rate * S / (1 - rate))) if rate < 1 else len(dead), len(dead))
        built = surv + dead[:d]
        real, null, p = _pooled_permutation(built, args.alpha, args.perms, seed=0)
        tag = f"survivors+{d} delisted" if d else "survivors only"
        flag = " *" if p < 0.05 else ""
        print(f"{tag:>26}{d:>7}{real:>+11.3f}{null:>+9.3f}{p:>8.3f}{flag}")

    print("\nRead: if the pooled IC / p-value collapses at a low delisting rate, the "
          "structural signal was largely survivorship (as meta-labeling proved to be). "
          "Large-cap delistings are rare, so survival at low rates would make it more "
          "credible. (Synthetic delisted — a sensitivity estimate, not point-in-time data.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
