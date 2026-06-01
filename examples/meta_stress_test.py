"""Statistical stress test for the meta-labeling 'edge' (is it real or noise?).

Runs the full meta-labeling evaluation across the whole 20-name basket and asks,
rigorously, whether the apparent uplift survives scrutiny:

* **costs** — net of a per-trade round-trip cost,
* **threshold sweep** — does it hold across meta-thresholds, or is 0.55 cherry-picked,
* **AUC significance** — a one-sample t-test of the 20 per-name OOS AUCs vs 0.5
  (each name ≈ one independent observation),
* **permutation test** — pooled label-shuffle null for the AUC.

    python examples/meta_stress_test.py
    python examples/meta_stress_test.py --cost-bps 10 --permutations 300

Honest scope: survivorship-biased universe; this is a falsification test, expected
(given the earlier results) to show the uplift is at the noise threshold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from vpts import (  # noqa: E402
    DataFetchError,
    build_meta_dataset,
    cpcv_meta_eval,
    permutation_test_meta,
)
from vpts.ml.models import MetaDataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Meta-labeling statistical stress test.")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--cost-bps", type=float, default=10.0, help="per-trade round-trip cost")
    ap.add_argument("--permutations", type=int, default=300)
    ap.add_argument("--tickers", nargs="*", default=[t for t, _ in GITHUB_TICKERS])
    args = ap.parse_args()

    load = github_loader()
    thresholds = [0.50, 0.55, 0.60, 0.65]
    print(f"Meta stress test — {len(args.tickers)} names, horizon={args.horizon}, "
          f"cost={args.cost_bps:.0f}bps/trade\n")

    datasets: list[MetaDataset] = []
    per_name_auc: list[float] = []
    per_name_net_ret: list[float] = []
    rows = []
    for sym in args.tickers:
        try:
            df = load(sym)
            ds = build_meta_dataset(df, lookback=args.lookback, horizon=args.horizon,
                                    stride=args.stride, symbol=sym, interval="1d")
            res = cpcv_meta_eval(ds, threshold=0.55, cost_bps=args.cost_bps)
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")
            continue
        datasets.append(ds)
        per_name_auc.append(res.oos_auc_mean)
        per_name_net_ret.append(res.meta_return_mean)
        rows.append((sym, res.base_win_rate, res.oos_auc_mean,
                     res.meta_return_mean, res.return_improvement_mean))

    if not datasets:
        print("No datasets built."); return 1

    print(f"{'name':<6}{'base_wr':>9}{'AUC':>8}{'meta_net%':>11}{'Δ%':>9}")
    for sym, bwr, auc, net, imp in rows:
        print(f"{sym:<6}{bwr * 100:>8.1f}%{auc:>8.3f}{net * 100:>10.3f}%{imp * 100:>8.2f}%")

    auc = np.array(per_name_auc, float)
    net = np.array(per_name_net_ret, float)

    print("\n--- Threshold sweep (pooled mean across names) ---")
    print(f"{'thr':>6}{'mean AUC':>11}{'mean meta_net%':>16}")
    for th in thresholds:
        a, r = [], []
        for ds in datasets:
            try:
                res = cpcv_meta_eval(ds, threshold=th, cost_bps=args.cost_bps)
            except ValueError:
                continue
            a.append(res.oos_auc_mean); r.append(res.meta_return_mean)
        print(f"{th:>6.2f}{np.nanmean(a):>11.3f}{np.nanmean(r) * 100:>15.3f}%")

    print("\n--- Significance (each name ≈ one observation) ---")
    t_auc = stats.ttest_1samp(auc, 0.5)
    t_ret = stats.ttest_1samp(net, 0.0)
    print(f"  AUC vs 0.5      : mean {auc.mean():.3f}  t={t_auc.statistic:+.2f}  "
          f"p={t_auc.pvalue:.3f}  ({'SIGNIFICANT' if t_auc.pvalue < 0.05 else 'n.s.'})")
    print(f"  meta_net vs 0   : mean {net.mean() * 100:+.3f}%  t={t_ret.statistic:+.2f}  "
          f"p={t_ret.pvalue:.3f}  ({'SIGNIFICANT' if t_ret.pvalue < 0.05 else 'n.s.'})")

    print("\n--- Pooled permutation test (label shuffle) ---")
    pooled = MetaDataset(
        X=np.vstack([d.X for d in datasets]),
        meta_label=np.concatenate([d.meta_label for d in datasets]),
        side=np.concatenate([d.side for d in datasets]),
        realized_return=np.concatenate([d.realized_return for d in datasets]),
        feature_names=datasets[0].feature_names,
        horizon=args.horizon, stride=args.stride, symbol="POOLED")
    perm = permutation_test_meta(pooled, n_permutations=args.permutations,
                                 threshold=0.55, cost_bps=args.cost_bps, seed=0)
    print(perm.summary())

    sig = (t_auc.pvalue < 0.05 and auc.mean() > 0.5 and perm.p_value_auc < 0.05
           and net.mean() > 0)
    print(f"\nVERDICT: {'a real, cost-surviving meta-edge' if sig else 'NO significant edge — consistent with noise'}")
    print("(survivorship-biased universe; pooled CV mixes names — a falsification check.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
