"""Phase C — predict MFE/MAE (triple-barrier win) from structural features, with XGBoost.

Reframes the target the way the proposal asks: instead of the raw forward return,
label each bar by whether a long bet's **Maximum Favorable Excursion beat its
Maximum Adverse Excursion** (a volatility-scaled triple barrier), and learn
``P(win)`` from the 13 structural features. Two models on the *same* purged-CPCV
splits:

  * **Logistic** (numpy) — the honest, overfit-resistant baseline; its OOS AUC is
    significance-tested by label permutation.
  * **XGBoost** (optional) — the nonlinear model the proposal names. We report its
    **in-sample vs OOS** AUC next to logistic, so any over-fitting is visible:
    a high in-sample AUC that doesn't carry to OOS is the tell.

    pip install xgboost            # optional; the logistic baseline runs without it
    python examples/structural_mfe_xgb.py --perms 200

Honest scope: survivorship-biased survivors, 2012-2017 daily, gross of cost. A
validity check on whether the MFE/MAE framing + a nonlinear model add OOS skill —
not a tradeable result.
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
    build_structural_meta_dataset,
    permutation_test_meta,
)
from vpts.ml.meta_model import LogisticMetaModel, _auc  # noqa: E402
from vpts.ml.models import MetaDataset  # noqa: E402

try:
    import xgboost as xgb  # native train/DMatrix API — no scikit-learn needed
    _HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_XGB = False

_XGB_PARAMS = {"max_depth": 3, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
               "lambda": 1.0, "min_child_weight": 5, "objective": "binary:logistic",
               "eval_metric": "logloss", "verbosity": 0}


def _logistic_fp(Xtr, ytr, Xte):
    return LogisticMetaModel().fit(Xtr, ytr).predict_proba(Xte)


def _xgb_fp(Xtr, ytr, Xte):
    bst = xgb.train(_XGB_PARAMS, xgb.DMatrix(Xtr, label=ytr), num_boost_round=120)
    return bst.predict(xgb.DMatrix(Xte))


def _cpcv_auc(ds: MetaDataset, fit_predict) -> tuple[float, list[float]]:
    """Return (in-sample AUC, list of OOS fold AUCs) for one name."""
    X, y = ds.X, ds.meta_label
    if len(np.unique(y)) < 2:
        return float("nan"), []
    ins = _auc(y, fit_predict(X, y, X))
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2,
                               purge=ds.purge_samples, embargo_pct=0.01)
    fold: list[float] = []
    for sp in cv.split(len(ds)):
        tr, te = sp.train_idx, sp.test_idx
        if tr.size < 30 or te.size < 10:
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        a = _auc(y[te], fit_predict(X[tr], y[tr], X[te]))
        if np.isfinite(a):
            fold.append(a)
    return ins, fold


def _pool(dsets: list[MetaDataset], horizon: int, stride: int) -> MetaDataset:
    return MetaDataset(
        X=np.vstack([d.X for d in dsets]),
        meta_label=np.concatenate([d.meta_label for d in dsets]),
        side=np.concatenate([d.side for d in dsets]),
        realized_return=np.concatenate([d.realized_return for d in dsets]),
        feature_names=dsets[0].feature_names, horizon=horizon, stride=stride, symbol="POOL")


def main() -> int:
    ap = argparse.ArgumentParser(description="Structural MFE/MAE meta-labeling + XGBoost.")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pt-mult", type=float, default=2.0)
    ap.add_argument("--sl-mult", type=float, default=2.0)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--tickers", nargs="*", default=[t for t, _ in GITHUB_TICKERS][:20])
    args = ap.parse_args()

    load = github_loader()
    kw = dict(lookback=args.lookback, horizon=args.horizon, stride=args.stride,
              pt_mult=args.pt_mult, sl_mult=args.sl_mult, side=1)
    print(f"Structural MFE/MAE (long triple-barrier, pt={args.pt_mult} sl={args.sl_mult}) — "
          f"{len(args.tickers)} names, XGBoost {'available' if _HAS_XGB else 'NOT installed'}\n")

    dsets: list[MetaDataset] = []
    for sym in args.tickers:
        try:
            ds = build_structural_meta_dataset(load(sym), symbol=sym, interval="1d", **kw)
        except (DataFetchError, ValueError):
            continue
        if len(ds) >= 40 and len(np.unique(ds.meta_label)) == 2:
            dsets.append(ds)
    if not dsets:
        print("No usable names."); return 1
    base = float(np.concatenate([d.meta_label for d in dsets]).mean())
    print(f"  {len(dsets)} names, {sum(len(d) for d in dsets)} events, "
          f"base win rate {base * 100:.1f}%\n")

    # ---- model comparison on identical CPCV splits ----
    models = [("logistic", _logistic_fp)]
    if _HAS_XGB:
        models.append(("xgboost", _xgb_fp))
    print(f"  {'model':>9} {'in-sample AUC':>15} {'OOS AUC':>10}   overfit gap")
    oos_by_model = {}
    for name, fp in models:
        ins_list, oos_all = [], []
        for ds in dsets:
            ins, fold = _cpcv_auc(ds, fp)
            if np.isfinite(ins):
                ins_list.append(ins)
            oos_all.extend(fold)
        ins_m = float(np.mean(ins_list)) if ins_list else float("nan")
        oos_m = float(np.mean(oos_all)) if oos_all else float("nan")
        oos_by_model[name] = oos_m
        print(f"  {name:>9} {ins_m:>15.3f} {oos_m:>10.3f}   {ins_m - oos_m:+.3f}")

    # ---- significance of the (logistic) meta-signal: pooled permutation ----
    print(f"\nPooled label-permutation test (logistic, {args.perms} shuffles) …")
    pt = permutation_test_meta(_pool(dsets, args.horizon, args.stride),
                               n_permutations=args.perms, threshold=0.55, cost_bps=0.0, seed=0)
    print(f"  OOS AUC real {pt.real_auc:.3f}  vs null {pt.null_auc_mean:.3f}  "
          f"-> p = {pt.p_value_auc:.3f}  "
          f"({'SIGNIFICANT' if pt.p_value_auc < 0.05 else 'not significant'})")

    # ---- honest verdict ----
    print()
    if _HAS_XGB:
        xgb_gain = oos_by_model.get("xgboost", float("nan")) - oos_by_model.get("logistic", float("nan"))
        if xgb_gain > 0.01:
            print(f"  XGBoost adds {xgb_gain:+.3f} OOS AUC over logistic — nonlinearity helps.")
        else:
            print(f"  XGBoost does NOT beat logistic OOS ({xgb_gain:+.3f}); its higher in-sample "
                  "AUC is over-fitting, not skill.")
    verdict = ("the MFE/MAE meta-signal is significant" if pt.p_value_auc < 0.05
               else "the MFE/MAE meta-signal is NOT significant under permutation")
    print(f"  Verdict: {verdict} (AUC {pt.real_auc:.3f}). Survivorship + cost caveats still apply; "
          "run structural_survivorship.py-style injection before believing any positive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
