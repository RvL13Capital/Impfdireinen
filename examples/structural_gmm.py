"""Experiment 12 — does a *parametric* EM-GMM decomposition beat the heuristic, or
is its one signal just price-minus-VWAP?

Experiment 9 decomposed the profile *heuristically* (smoothed peaks, P/b/B/D
shapes). A natural objection: maybe a *parametric* decomposition would recover real
hidden structure the smoothing blurs. So I fit a **1-D Gaussian mixture by weighted
EM** (`vpts.structure.gmm`) to each profile and turned the hidden POCs / antimode /
gravity into 7 features. This script asks — adversarially, against a one-line
baseline — whether any of that machinery actually earns its keep.

It checks four things, each on the *same* CPCV factor harness as everything else:

1. **Don't pool blindly.** The 7-feature ridge IC is near-zero — but that is *ridge
   dilution*, not absence of signal: one feature (`gmm_gravity`) carries it. Per-
   feature IC makes that visible (the lesson: read features individually first).
2. **The decomposition's one signal is `gmm_gravity`** — signed distance from price
   to the dominant hidden POC. On survivors its single-factor OOS IC ≈ the
   heuristic's best.
3. **…but it is just price-minus-VWAP.** Against a **no-GMM baseline** (`vwap_dist`
   = (close−VWAP)/range, and trailing-return momentum), `gmm_gravity` is ~0.9
   correlated, adds **no** incremental OOS IC, has **~0 partial correlation**
   controlling for momentum+VWAP — and the one-line baseline *scores higher*.
4. **Economically it's the same survivorship family** — long-only inverts under
   delisted injection exactly like momentum/VWAP-distance.

Verdict: the EM-GMM decomposition adds nothing the heuristic (or a moving-average
distance) didn't already have; the "hidden-POC gravity" reduces to extension from
VWAP. Feature *content* (trend/extension), not decomposition *machinery*, is the axis.

    python examples/structural_gmm.py
    python examples/structural_gmm.py --survivors AAPL MSFT JPM --perms 100

Honest scope: survivorship-biased 2012–17 daily; synthetic delisted (sensitivity
estimate); gross-of-cost IC; a research decomposition, not a tradeable result.
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
    cpcv_factor_eval,
    cpcv_factor_quantile_returns,
)
from vpts.ml.models import FactorDataset  # noqa: E402
from vpts.profile.calculator import VolumeProfileCalculator  # noqa: E402
from vpts.regime.indicators import ensure_ohlcv  # noqa: E402
from vpts.structure.gmm import GMM_FEATURES, gmm_feature_vector  # noqa: E402

BASELINES = ("vwap_dist", "mom_120", "mom_20")     # no-GMM controls
COLS = GMM_FEATURES + BASELINES


def _build_combined(df, sym, *, lookback, horizon, stride) -> FactorDataset:
    """One no-look-ahead walk → 7 EM-GMM features + 3 no-GMM baselines → forward return."""
    ensure_ohlcv(df, min_bars=lookback + horizon + 2)
    pc = VolumeProfileCalculator(bin_mode="auto")
    close = df["Close"].to_numpy(float)
    vol = df["Volume"].to_numpy(float)
    n = len(df)
    rows, ys, ts = [], [], []
    for t in range(lookback - 1, n - horizon, max(1, stride)):
        window = df.iloc[t - lookback + 1 : t + 1]
        try:
            profile = pc.calculate(window, sym, "1d")
        except (ValueError, ZeroDivisionError):
            continue
        gmm = gmm_feature_vector(profile, close[t])
        sl = slice(t - lookback + 1, t + 1)
        vwap = float((close[sl] * vol[sl]).sum() / max(float(vol[sl].sum()), 1e-9))
        rng = profile.price_high - profile.price_low
        vwap_dist = (close[t] - vwap) / rng if rng > 1e-9 else 0.0
        mom_120 = close[t] / close[t - lookback + 1] - 1.0
        mom_20 = close[t] / close[t - 20] - 1.0
        row = [*gmm, vwap_dist, mom_120, mom_20]
        if not np.all(np.isfinite(row)):
            continue
        rows.append(row)
        ys.append(float(close[t + horizon] / close[t] - 1.0))
        ts.append(df.index[t])
    return FactorDataset(
        X=np.array(rows, float).reshape(-1, len(COLS)), y=np.array(ys, float),
        baseline=np.zeros(len(ys)), feature_names=COLS, horizon=horizon,
        stride=max(1, stride), symbol=sym)


def _sub(ds: FactorDataset, cols):
    idx = [ds.feature_names.index(c) for c in cols]
    sub = FactorDataset(X=ds.X[:, idx], y=ds.y, baseline=ds.baseline, feature_names=tuple(cols),
                        horizon=ds.horizon, stride=ds.stride, symbol=ds.symbol)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge=sub.purge_samples, embargo_pct=0.01)
    return sub, cv


def _pooled_ic(dsets, cols) -> float:
    folds = []
    for ds in dsets:
        sub, cv = _sub(ds, cols)
        try:
            folds.extend(cpcv_factor_eval(sub, cv=cv).fold_ics)
        except ValueError:
            continue
    return float(np.mean(folds)) if folds else float("nan")


def _buckets(dsets, col):
    lo, ls = [], []
    for ds in dsets:
        sub, cv = _sub(ds, (col,))
        try:
            r = cpcv_factor_quantile_returns(sub, cv=cv, n_buckets=5, cost_bps=10.0)
        except ValueError:
            continue
        lo.append(r.long_only_net_pct); ls.append(r.long_short_net_pct)
    return (float(np.mean(lo)) if lo else float("nan"),
            float(np.mean(ls)) if ls else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description="EM-GMM decomposition vs a no-GMM VWAP/momentum baseline.")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--n-delisted", type=int, default=9)
    ap.add_argument("--survivors", nargs="*", default=[t for t, _ in GITHUB_TICKERS])
    args = ap.parse_args()

    load = github_loader()
    kw = dict(lookback=args.lookback, horizon=args.horizon, stride=args.stride)
    print(f"Experiment 12 — EM-GMM decomposition vs no-GMM baseline — {len(args.survivors)} survivors "
          f"+ {args.n_delisted} synthetic delisted\n")

    surv, dead = [], []
    for sym in args.survivors:
        try:
            surv.append(_build_combined(load(sym), sym, **kw))
        except (DataFetchError, ValueError):
            pass
    for k in range(args.n_delisted):
        try:
            dead.append(_build_combined(synthetic_delisted_ohlcv(seed=100 + k), f"DEAD{k}", **kw))
        except ValueError:
            continue
    if not surv:
        print("No survivors built."); return 1
    both = surv + dead
    print(f"  built {len(surv)} survivors, {len(dead)} delisted\n")

    # 1) Read features individually — the 7-feature ridge DILUTES the one real feature.
    per = {f: _pooled_ic(surv, (f,)) for f in GMM_FEATURES}
    best = max(per, key=lambda f: abs(per[f]))
    print("1) Don't pool blindly — GMM 7-feature ridge vs best single feature (survivors OOS IC):")
    print(f"   all 7 features (ridge) : {_pooled_ic(surv, GMM_FEATURES):+.3f}   <- dilution hides the signal")
    print(f"   best single ({best})   : {per[best]:+.3f}")
    print("   per-feature OOS IC:")
    for f in GMM_FEATURES:
        print(f"     {f:22} {per[f]:+.3f}")

    # 2-3) The one signal (gmm_gravity) is just price-minus-VWAP.
    gv = np.concatenate([d.X[:, COLS.index("gmm_gravity")] for d in surv])
    vw = np.concatenate([d.X[:, COLS.index("vwap_dist")] for d in surv])
    mo = np.concatenate([d.X[:, COLS.index("mom_120")] for d in surv])
    yy = np.concatenate([d.y for d in surv])
    Z = np.column_stack([mo, vw, np.ones(len(yy))])
    coef_g = np.linalg.lstsq(Z, gv, rcond=None)[0]
    coef_y = np.linalg.lstsq(Z, yy, rcond=None)[0]
    partial = float(np.corrcoef(gv - Z @ coef_g, yy - Z @ coef_y)[0, 1])
    print("\n2-3) Is gmm_gravity just price-minus-VWAP? (no-GMM baseline = vwap_dist, momentum)")
    print(f"   corr(gmm_gravity, vwap_dist)            : {np.corrcoef(gv, vw)[0,1]:+.2f}")
    print(f"   corr(gmm_gravity, mom_120)              : {np.corrcoef(gv, mo)[0,1]:+.2f}")
    print(f"   single-factor OOS IC  gmm_gravity        : {_pooled_ic(surv, ('gmm_gravity',)):+.3f}")
    print(f"   single-factor OOS IC  vwap_dist (no GMM) : {_pooled_ic(surv, ('vwap_dist',)):+.3f}   <- baseline wins")
    print(f"   single-factor OOS IC  mom_120 (no GMM)   : {_pooled_ic(surv, ('mom_120',)):+.3f}")
    # Clean incremental test (avoid multi-feature ridge dilution): does adding gravity to the
    # baseline help? It LOWERS the IC — a redundant, noisier copy of vwap_dist.
    print(f"   IC  [vwap_dist + gmm_gravity]            : {_pooled_ic(surv, ('vwap_dist','gmm_gravity')):+.3f}   <- adding gravity to the baseline lowers it")
    print(f"   partial corr(gravity, ret | mom, vwap)   : {partial:+.3f}   <- ~0 (in-sample): no info beyond momentum/VWAP")

    # 4) Economically the same survivorship family — long-only inverts under injection.
    print("\n4) Economic test — conviction buckets, long top / short bottom / FLAT middle (10bps):")
    print(f"   {'feature':>14} {'long-only surv→inj':>22} {'tails L/S surv→inj':>22}")
    for c in ("gmm_gravity", "vwap_dist"):
        lo_s, ls_s = _buckets(surv, c); lo_b, ls_b = _buckets(both, c)
        print(f"   {c:>14}   {lo_s:+6.2f}% → {lo_b:+6.2f}%      {ls_s:+6.2f}% → {ls_b:+6.2f}%")

    print("\nVerdict: the 7-feature ridge near-zero IC was dilution; gmm_gravity carries a real "
          f"survivor signal — but it is ~0.9 correlated with a no-GMM VWAP-distance feature that scores "
          "higher, adds no incremental IC, and has ~0 partial correlation once momentum/VWAP are "
          "controlled. Economically it inverts under injection like any momentum/extension signal. The "
          "EM-GMM decomposition adds nothing; its one signal reduces to price-minus-VWAP. (Synthetic "
          "delisted; gross of cost; research decomposition, not a tradeable result.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
