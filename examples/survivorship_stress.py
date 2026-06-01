"""Survivorship-injection stress test for the meta-labeling 'edge'.

The 20-name basket is all 2017 *survivors*. A point-in-time universe would also
contain names that were trading in 2012–17 but later **delisted/went bankrupt**.
This script injects synthetic delisted names — normal for a while, then declining
to pennies with elevated volatility — alongside the survivors, and measures how
the meta-edge (per-name AUC vs 0.5, net return, permutation p-value) **shrinks**
as the delisting rate rises. If it collapses at a plausible rate, the edge was
largely survivorship; if it holds, it is more credible.

    python examples/survivorship_stress.py
    python examples/survivorship_stress.py --n-delisted 12 --cost-bps 10

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
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from vpts import (  # noqa: E402
    DataFetchError,
    build_meta_dataset,
    cpcv_meta_eval,
    permutation_test_meta,
)
from vpts.ml.models import MetaDataset  # noqa: E402


def synthetic_delisted_ohlcv(n: int = 1259, seed: int = 0) -> pd.DataFrame:
    """A 'name that died': ~normal, then a sustained decline to pennies.

    Generated from a fixed, strategy-agnostic process (not tuned to the model):
    a flat-ish first phase, then a negative-drift, elevated-volatility decline
    floored near zero, with capitulation volume spikes on big down days.
    """
    rng = np.random.default_rng(seed)
    start = rng.uniform(40.0, 160.0)
    cut = int(rng.uniform(0.2, 0.5) * n)              # when the decline begins
    vol_n, vol_d = rng.uniform(0.012, 0.022), rng.uniform(0.03, 0.06)
    drift_d = -rng.uniform(0.002, 0.005)              # -0.2%..-0.5% / day
    rets = np.empty(n)
    rets[:cut] = rng.normal(0.0, vol_n, cut)
    rets[cut:] = rng.normal(drift_d, vol_d, n - cut)
    close = np.maximum(start * np.exp(np.cumsum(rets)), 0.20)   # penny floor
    bar_vol = np.where(np.arange(n) < cut, vol_n, vol_d)
    high = close * (1.0 + np.abs(rng.normal(0, bar_vol, n)))
    low = close * (1.0 - np.abs(rng.normal(0, bar_vol, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    base = rng.uniform(2e6, 8e6)
    volume = base * (1.0 + 3.0 * np.maximum(0.0, -rets) / (vol_d + 1e-9))
    idx = pd.date_range("2012-09-04", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": np.maximum(high, close),
                         "Low": np.minimum(low, close), "Close": close,
                         "Volume": volume}, index=idx)


def _eval(ds: MetaDataset, cost_bps: float):
    try:
        r = cpcv_meta_eval(ds, threshold=0.55, cost_bps=cost_bps)
        return r.oos_auc_mean, r.meta_return_mean
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Survivorship-injection stress test.")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--n-delisted", type=int, default=10)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--survivors", nargs="*", default=[t for t, _ in GITHUB_TICKERS])
    args = ap.parse_args()

    load = github_loader()
    kw = dict(lookback=args.lookback, horizon=args.horizon, stride=args.stride)

    print(f"Survivorship stress — {len(args.survivors)} survivors + up to "
          f"{args.n_delisted} synthetic delisted, cost={args.cost_bps:.0f}bps\n")

    surv_auc, surv_net, surv_ds = [], [], []
    for sym in args.survivors:
        try:
            ds = build_meta_dataset(load(sym), symbol=sym, interval="1d", **kw)
            ev = _eval(ds, args.cost_bps)
        except (DataFetchError, ValueError):
            ev = None
        if ev:
            surv_auc.append(ev[0]); surv_net.append(ev[1]); surv_ds.append(ds)

    del_auc, del_net, del_ds = [], [], []
    for k in range(args.n_delisted):
        ds = build_meta_dataset(synthetic_delisted_ohlcv(seed=100 + k),
                                symbol=f"DEAD{k}", interval="1d", **kw)
        ev = _eval(ds, args.cost_bps)
        if ev:
            del_auc.append(ev[0]); del_net.append(ev[1]); del_ds.append(ds)

    surv_auc = np.array(surv_auc); surv_net = np.array(surv_net)
    del_auc = np.array(del_auc); del_net = np.array(del_net)
    print(f"Survivors: {surv_auc.size} usable, mean AUC {surv_auc.mean():.3f}  "
          f"net {surv_net.mean() * 100:+.3f}%")
    print(f"Delisted : {del_auc.size} usable, mean AUC {del_auc.mean():.3f}  "
          f"net {del_net.mean() * 100:+.3f}%\n")

    S = surv_auc.size
    print("--- Delisting-rate sweep (each name = one observation) ---")
    print(f"{'rate':>6}{'nDead':>7}{'meanAUC':>10}{'p(AUC>.5)':>12}{'net%':>9}")
    for rate in (0.0, 0.05, 0.10, 0.20, 0.30):
        d = min(int(round(rate * S / (1 - rate))) if rate < 1 else del_auc.size, del_auc.size)
        aucs = np.concatenate([surv_auc, del_auc[:d]])
        nets = np.concatenate([surv_net, del_net[:d]])
        p = stats.ttest_1samp(aucs, 0.5).pvalue if aucs.size > 1 else float("nan")
        print(f"{rate:>6.0%}{d:>7}{aucs.mean():>10.3f}{p:>12.3f}{nets.mean() * 100:>8.3f}%")

    # Permutation: survivors-only vs survivors + ALL delisted.
    def pool(dsets):
        return MetaDataset(
            X=np.vstack([d.X for d in dsets]),
            meta_label=np.concatenate([d.meta_label for d in dsets]),
            side=np.concatenate([d.side for d in dsets]),
            realized_return=np.concatenate([d.realized_return for d in dsets]),
            feature_names=dsets[0].feature_names, horizon=args.horizon,
            stride=args.stride, symbol="POOL")
    print("\n--- Pooled permutation (label shuffle) ---")
    p_surv = permutation_test_meta(pool(surv_ds), n_permutations=args.permutations,
                                   threshold=0.55, cost_bps=args.cost_bps, seed=0)
    print(f"  survivors-only       : AUC {p_surv.real_auc:.3f}  p={p_surv.p_value_auc:.3f}")
    if del_ds:
        p_all = permutation_test_meta(pool(surv_ds + del_ds), n_permutations=args.permutations,
                                      threshold=0.55, cost_bps=args.cost_bps, seed=0)
        print(f"  survivors + delisted : AUC {p_all.real_auc:.3f}  p={p_all.p_value_auc:.3f}")

    print("\nRead: if mean AUC / significance collapses at a low delisting rate, the "
          "edge was largely survivorship. Large-cap delisting is rare (~low rate), so "
          "survival at low rates would make the edge more credible. (Synthetic delisted "
          "— a sensitivity estimate, not real point-in-time data.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
