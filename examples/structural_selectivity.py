"""Push the one thread that resisted survivorship: meta-labeling **selectivity**.

The swing rater showed that *which* long setups are higher-R:R (selectivity) — as
opposed to *whether* to be long (direction) — was the only signal in the arc that
degraded rather than inverting under delisted injection (+0.14%/bet, p=0.005 on
survivors → +0.09%, p=0.10 injected). Before believing it, three adversarial tests:

  1. **Robustness grid** — is the survivors lift stable across horizon / R:R /
     selection fraction, or did one lucky (10, 2:1, top-20%) combo carry it? A real
     effect is broadly positive; a mined one is erratic.
  2. **Feature decomposition** — is the lift carried by REGIME features (genuine) or
     the survivorship-prone DIP features (the mirage at its source)?
  3. **Significance** — permutation p of the lift, survivors vs injected, at full
     power (pass more --tickers).

    python examples/structural_selectivity.py --tickers <many> --perms 300

Honest scope: survivorship-biased 2012–2017 daily; synthetic delisted = sensitivity
estimate; non-overlap 10 bps cost; a research probe, not a tradeable result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from survivorship_stress import synthetic_delisted_ohlcv  # noqa: E402
from vpts import (  # noqa: E402
    DataFetchError,
    build_structural_meta_dataset,
    cpcv_meta_eval,
    permutation_test_meta,
)
from vpts.ml.models import MetaDataset  # noqa: E402

DIP = ("delta_net", "delta_poc", "poc_loc", "cost_basis_migration")
REGIME = ("vacr_z", "poc_slope", "skew", "kurtosis", "n_ledges", "poor_high",
          "is_P", "is_b", "is_B")
COST_BPS = 10.0


def _subset(ds: MetaDataset, names: tuple[str, ...]) -> MetaDataset:
    idx = [ds.feature_names.index(n) for n in names]
    return MetaDataset(X=ds.X[:, idx], meta_label=ds.meta_label, side=ds.side,
                       realized_return=ds.realized_return, feature_names=tuple(names),
                       horizon=ds.horizon, stride=ds.stride, symbol=ds.symbol)


def _pool(dsets: list[MetaDataset], h: int, stride: int) -> MetaDataset:
    return MetaDataset(
        X=np.vstack([d.X for d in dsets]),
        meta_label=np.concatenate([d.meta_label for d in dsets]),
        side=np.concatenate([d.side for d in dsets]),
        realized_return=np.concatenate([d.realized_return for d in dsets]),
        feature_names=dsets[0].feature_names, horizon=h, stride=stride, symbol="POOL")


def _lift(pool: MetaDataset, top: float, names: tuple[str, ...] | None = None) -> float:
    p = _subset(pool, names) if names else pool
    return cpcv_meta_eval(p, select_top=top, cost_bps=COST_BPS).return_improvement_mean * 100.0


class Builder:
    """Builds (and caches) pooled survivor / +delisted meta-datasets per (h, pt, sl)."""

    def __init__(self, surv_dfs: dict, dead_dfs: dict, stride: int):
        self.surv_dfs, self.dead_dfs, self.stride = surv_dfs, dead_dfs, stride
        self._cache: dict = {}

    def _one(self, df: pd.DataFrame, sym: str, h: int, pt: float, sl: float):
        try:
            ds = build_structural_meta_dataset(df, symbol=sym, lookback=120, horizon=h,
                                               stride=self.stride, pt_mult=pt, sl_mult=sl, side=1)
        except (DataFetchError, ValueError):
            return None
        return ds if (len(ds) >= 40 and len(np.unique(ds.meta_label)) == 2) else None

    def pools(self, h: int, pt: float, sl: float) -> tuple[MetaDataset, MetaDataset]:
        key = (h, pt, sl)
        if key not in self._cache:
            surv = [d for d in (self._one(df, s, h, pt, sl) for s, df in self.surv_dfs.items()) if d]
            dead = [d for d in (self._one(df, s, h, pt, sl) for s, df in self.dead_dfs.items()) if d]
            self._cache[key] = (_pool(surv, h, self.stride), _pool(surv + dead, h, self.stride))
        return self._cache[key]


def main() -> int:
    ap = argparse.ArgumentParser(description="Selectivity robustness / decomposition / significance.")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--top", type=float, default=0.2)
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--n-delisted", type=int, default=9)
    ap.add_argument("--tickers", nargs="*", default=[t for t, _ in GITHUB_TICKERS][:20])
    args = ap.parse_args()

    load = github_loader()
    surv_dfs: dict = {}
    for sym in args.tickers:
        try:
            surv_dfs[sym] = load(sym)
        except DataFetchError:
            pass
    dead_dfs = {f"DEAD{k}": synthetic_delisted_ohlcv(seed=100 + k) for k in range(args.n_delisted)}
    b = Builder(surv_dfs, dead_dfs, args.stride)
    print(f"Cached {len(surv_dfs)} survivors + {len(dead_dfs)} delisted; selectivity = best-"
          f"{args.top * 100:.0f}% rated long setups, {COST_BPS:.0f}bps.\n")

    # ---- 1) robustness grid: is the survivors lift stable across params? ---- #
    print("1) Selectivity LIFT (%/trade) across parameters     survivors   +delisted")
    print("   horizon (R:R 2:1, top 20%)")
    for h in (5, 10, 15):
        s, bo = b.pools(h, 2.0, 1.0)
        print(f"     h={h:<3}                                       {_lift(s, args.top):+8.3f}   {_lift(bo, args.top):+8.3f}")
    print("   reward:risk (h=10, top 20%)")
    for pt, sl, rr in ((1.5, 1.0, "1.5:1"), (2.0, 1.0, "2:1"), (3.0, 1.0, "3:1")):
        s, bo = b.pools(10, pt, sl)
        print(f"     {rr:<6}                                    {_lift(s, args.top):+8.3f}   {_lift(bo, args.top):+8.3f}")
    print("   selection fraction (h=10, R:R 2:1)")
    s, bo = b.pools(10, 2.0, 1.0)
    for top in (0.1, 0.2, 0.3):
        print(f"     top {top * 100:.0f}%                                     {_lift(s, top):+8.3f}   {_lift(bo, top):+8.3f}")

    # ---- 2) feature decomposition: REGIME vs DIP carry the selectivity? ---- #
    print("\n2) Which features drive the lift? (h=10, R:R 2:1, top 20%)")
    print(f"   {'group':>8}   survivors   +delisted")
    for tag, names in (("ALL", None), ("REGIME", REGIME), ("DIP", DIP)):
        print(f"   {tag:>8}   {_lift(s, args.top, names):+8.3f}   {_lift(bo, args.top, names):+8.3f}")

    # ---- 3) significance at full power: survivors vs injected ---- #
    print(f"\n3) Permutation significance of the LIFT ({args.perms} shuffles)")
    for tag, pool in (("survivors", s), ("+delisted", bo)):
        pt_ = permutation_test_meta(pool, n_permutations=args.perms, select_top=args.top,
                                    cost_bps=COST_BPS, seed=0)
        sig = "significant" if pt_.p_value_improvement < 0.05 else "NOT significant"
        print(f"   [{tag}]  real {pt_.real_improvement * 100:+.3f}%  vs null "
              f"{pt_.null_improvement_mean * 100:+.3f}%  -> p = {pt_.p_value_improvement:.3f} ({sig})")

    print("\nVerdict: selectivity is real-and-worth-pursuing only if the survivors lift is positive "
          "ACROSS the grid, is carried by REGIME (not just DIP), and stays significant injected. "
          "Erratic signs, DIP-only, or p>0.05 injected => close the thread.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
