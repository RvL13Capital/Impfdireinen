"""Swing-trade setup rater — score the current setup and judge its risk/reward.

This is the product framing: for a **swing** horizon (days–weeks, no HFT/scalping),
rate the setup in front of you and decide whether it is a good entry at a favorable
risk/reward — otherwise stay **flat**. Mechanically it is meta-labeling:

  * the bet is a long (``--side 1``) or short (``--side -1``) swing entry, with a
    **volatility-scaled triple barrier** that *defines the R:R* — take-profit at
    ``pt_mult × vol``, stop at ``sl_mult × vol`` (default 2:1 reward:risk);
  * the **max holding period** (``--max-hold``) is the vertical barrier / time stop:
    if no barrier is touched within it the trade is force-exited. The rater reports
    the *realised* holding-period distribution (most trades exit early at a barrier);
  * the **rater** (numpy logistic) learns ``P(win)`` from the 13 structural
    features and turns it into a 0–100 **setup rating**;
  * acting only on the best-rated setups (flat otherwise) should lift the
    **expectancy** of taken trades over taking every signal.

The honest question is NOT "is the AUC high" — it is "does the *selectivity* raise
expectancy, and does that lift **survive survivorship**?" So every number is shown
on survivors and again with synthetic delisted names injected.

    python examples/structural_swing_rater.py --max-hold 10 --pt-mult 2 --sl-mult 1 --select-top 0.2

Honest scope: survivorship-biased 2012–2017 daily; synthetic delisted = sensitivity
estimate; non-overlap per-trade cost; a research rater, not trading advice.
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
    DataFetchError,
    LogisticMetaModel,
    build_structural_meta_dataset,
    cpcv_meta_eval,
    permutation_test_meta,
)
from vpts.ml.models import MetaDataset  # noqa: E402


def _pool(dsets: list[MetaDataset], horizon: int, stride: int) -> MetaDataset:
    return MetaDataset(
        X=np.vstack([d.X for d in dsets]),
        meta_label=np.concatenate([d.meta_label for d in dsets]),
        side=np.concatenate([d.side for d in dsets]),
        realized_return=np.concatenate([d.realized_return for d in dsets]),
        feature_names=dsets[0].feature_names, horizon=horizon, stride=stride, symbol="POOL",
        holding_bars=np.concatenate([d.holding_bars for d in dsets]))


def _holding_report(dsets: list[MetaDataset], horizon: int) -> None:
    """Surface the MAX holding period and the *realised* holding-period distribution."""
    h = np.concatenate([d.holding_bars for d in dsets])
    win = np.concatenate([d.meta_label for d in dsets]).astype(bool)
    capped = float(np.mean(h >= horizon) * 100.0)
    q25, q50, q75 = (int(x) for x in np.percentile(h, [25, 50, 75]))
    bins = np.bincount(np.clip(h, 1, horizon), minlength=horizon + 1)[1:horizon + 1]
    mx = int(bins.max()) or 1
    spark = "".join("▁▂▃▄▅▆▇█"[min(7, int(7 * b / mx))] for b in bins)
    print(f"\nMax holding period {horizon} bars (time stop) — realised bars-in-trade:")
    print(f"  mean {h.mean():.1f} · median {q50} · p25/p75 {q25}/{q75} bars · "
          f"{capped:.0f}% run to the cap (rest exit early at a barrier)")
    print(f"  winners exit in {h[win].mean():.1f} bars avg · losers {h[~win].mean():.1f} bars avg")
    print(f"  distribution 1→{horizon} bars: {spark}")


def _report(tag: str, pool: MetaDataset, top: float, cost_bps: float, rr: float) -> None:
    r = cpcv_meta_eval(pool, select_top=top, cost_bps=cost_bps)
    breakeven = 1.0 / (1.0 + rr) * 100.0           # win rate needed at this R:R
    print(f"  [{tag}]  {len(pool)} setups, base win {r.base_win_rate * 100:.1f}%  "
          f"(breakeven @ {rr:.0f}:1 = {breakeven:.0f}%)")
    print(f"     take-all expectancy   : {r.primary_return_mean * 100:+.3f}% / trade")
    print(f"     best-{top * 100:.0f}% rated expectancy: {r.meta_return_mean * 100:+.3f}% / trade   "
          f"(win {r.oos_precision_mean * 100:.1f}%, in market {r.avg_fraction_taken * 100:.0f}%)")
    print(f"     selectivity LIFT      : {r.return_improvement_mean * 100:+.3f}% / trade   "
          f"({r.pct_folds_meta_beats_primary:.0f}% of folds beat take-all)")


def _rating_demo(load, sym: str, kw: dict, rr: float) -> None:
    """Train the rater on a name's early history; rate its most recent setups."""
    try:
        ds = build_structural_meta_dataset(load(sym), symbol=sym, interval="1d", **kw)
    except (DataFetchError, ValueError):
        print(f"  ({sym}: unavailable)"); return
    if len(ds) < 40:
        print(f"  ({sym}: too few setups)"); return
    cut = int(len(ds) * 0.7)
    model = LogisticMetaModel().fit(ds.X[:cut], ds.meta_label[:cut])   # train strictly before
    rate_from = cut + ds.purge_samples                                 # purge the horizon overlap
    p = model.predict_proba(ds.X[rate_from:])
    er = (1.0 + rr) * p - 1.0                                          # expected R-multiple = (1+RR)P - 1
    side = "LONG" if ds.side[0] > 0 else "SHORT"
    hold = ds.holding_bars
    cap = ds.horizon
    print(f"\nLive-rating demo — {sym} ({side} swing entries; rater trained on first {cut} setups):")
    print(f"  {'date':>12} {'rating':>7} {'E[R]':>7} {'verdict':>10}  {'held':>7}  actual")
    for i in range(max(0, len(p) - 5), len(p)):           # last 5 rated setups
        j = rate_from + i
        ts = ds.timestamps[j] if ds.timestamps is not None else j
        date = ts.date() if hasattr(ts, "date") else ts
        verdict = "TAKE" if er[i] > 0 else "skip (flat)"
        outcome = "win" if ds.meta_label[j] == 1 else "loss"
        held = f"{int(hold[j])}b{'*' if hold[j] >= cap else ''}"     # * = ran to the max-hold cap
        print(f"  {str(date):>12} {p[i] * 100:>6.0f} {er[i]:>+7.2f} {verdict:>10}  {held:>7}  "
              f"{outcome} ({ds.realized_return[j] * 100:+.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Swing-trade structural setup rater + R:R / survivorship test.")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--max-hold", type=int, default=10, dest="horizon",
                    help="MAX HOLDING PERIOD in bars — the triple-barrier vertical / time stop")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--pt-mult", type=float, default=2.0, help="take-profit barrier (× vol)")
    ap.add_argument("--sl-mult", type=float, default=1.0, help="stop barrier (× vol)")
    ap.add_argument("--side", type=int, default=1, choices=(1, -1), help="1 = rate buys, -1 = rate sells")
    ap.add_argument("--select-top", type=float, default=0.2, help="act on the best-rated fraction (0-1)")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--n-delisted", type=int, default=9)
    ap.add_argument("--tickers", nargs="*", default=[t for t, _ in GITHUB_TICKERS][:20])
    args = ap.parse_args()

    rr = args.pt_mult / args.sl_mult
    load = github_loader()
    kw = dict(lookback=args.lookback, horizon=args.horizon, stride=args.stride,
              pt_mult=args.pt_mult, sl_mult=args.sl_mult, side=args.side)
    label = "BUY" if args.side > 0 else "SELL"
    print(f"Swing {label} setup rater — max-hold {args.horizon} bars (~{args.horizon / 5:.0f}w), "
          f"R:R {rr:.0f}:1 (pt {args.pt_mult}× / sl {args.sl_mult}× vol), "
          f"act on best {args.select_top * 100:.0f}% rated\n")

    surv: list[MetaDataset] = []
    for sym in args.tickers:
        try:
            ds = build_structural_meta_dataset(load(sym), symbol=sym, interval="1d", **kw)
        except (DataFetchError, ValueError):
            continue
        if len(ds) >= 40 and len(np.unique(ds.meta_label)) == 2:
            surv.append(ds)
    if not surv:
        print("No usable names."); return 1
    dead = [build_structural_meta_dataset(synthetic_delisted_ohlcv(seed=100 + k),
                                          symbol=f"DEAD{k}", **kw) for k in range(args.n_delisted)]
    dead = [d for d in dead if len(d) >= 20 and len(np.unique(d.meta_label)) == 2]

    _holding_report(surv, args.horizon)

    print(f"\nDoes rating setups raise expectancy — and does the lift survive survivorship?")
    _report("survivors", _pool(surv, args.horizon, args.stride), args.select_top, args.cost_bps, rr)
    _report("+delisted", _pool(surv + dead, args.horizon, args.stride), args.select_top, args.cost_bps, rr)

    print(f"\nIs the selectivity LIFT significant — on survivors, and still with delisted injected?")
    for tag, pool in (("survivors", _pool(surv, args.horizon, args.stride)),
                      ("+delisted", _pool(surv + dead, args.horizon, args.stride))):
        pt = permutation_test_meta(pool, n_permutations=args.perms, select_top=args.select_top,
                                   cost_bps=args.cost_bps, seed=0)
        sig = "significant" if pt.p_value_improvement < 0.05 else "NOT significant"
        print(f"  [{tag}]  LIFT real {pt.real_improvement * 100:+.3f}% vs null "
              f"{pt.null_improvement_mean * 100:+.3f}%  -> p = {pt.p_value_improvement:.3f} ({sig})")

    _rating_demo(load, args.tickers[0], kw, rr)

    print("\nRead: a useful rater makes rated-setup expectancy > take-all (positive LIFT) AND keeps it "
          "positive with delisted names injected. If the LIFT is positive on survivors but flips "
          "negative injected, the rating is selecting on survival, not on genuine R:R.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
