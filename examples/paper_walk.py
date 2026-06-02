"""Phase 7 — the forward paper-walk runner (survivorship-free evidence, paper only).

Thirteen experiments found **no survivorship-robust edge** in the historical data, and
the binding constraint is the data itself. The one evidence the backtest can't make is
**survivorship-free** evidence — so this collects it forward, in paper.

  * ``--live`` : one honest day. Fetch current data (free yfinance), decide on bars up
    to today, append any actionable call to the JSONL ledger, resolve prior open orders
    against the bars that have since arrived, print the running track record. Drop this
    behind a daily cron ~15 min before the close and let it accumulate untouched.

  * ``--demo`` (default): replay the *mechanism* over the static 2012–17 sample (no
    network) so you can see how the ledger fills. NB this demo data is still
    survivorship-biased — it shows the plumbing, not an edge.

    python examples/paper_walk.py --demo
    python examples/paper_walk.py --live --watchlist AAPL MSFT JPM --ledger ~/.vpts_paper.jsonl

It never places an order or moves money.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from vpts.execution import PaperLedger, run_paper_walk  # noqa: E402


def _print_report(rep: dict, ledger: PaperLedger) -> None:
    print(f"\n[{rep['as_of']}]  +{rep['new']} new order(s)  ·  "
          f"{rep['open']} open · {rep['resolved']} resolved")
    if rep["resolved"]:
        print(f"  track record (resolved; survivorship-free only if run --live): "
              f"profitable {rep['pct_profitable'] * 100:.0f}% (R>0)  ·  avg {rep['avg_R']:+.2f}R  ·  "
              f"total {rep['total_R']:+.1f}R")
        print(f"  exits: {rep['target_hits']} target · {rep['stops']} stop · {rep['timeouts']} time-stop")
    for sym, err in rep["errors"]:
        print(f"  ! {sym}: {err}")


def _live(args) -> int:
    from vpts.data.fetcher import MarketDataFetcher
    fetcher = MarketDataFetcher(cache_ttl=3600)
    load = lambda s: fetcher.fetch(s, period=args.period, interval="1d")  # noqa: E731
    ledger = PaperLedger.load(args.ledger)
    as_of = pd.Timestamp.today().normalize()
    rep = run_paper_walk(load, args.watchlist, as_of, ledger,
                         style=args.style, max_hold=args.max_hold)
    ledger.save(args.ledger)
    print(f"Forward paper-walk (LIVE) — {len(args.watchlist)} names, ledger {args.ledger}")
    _print_report(rep, ledger)
    print("\n(paper only — no order placed, no money moved.)")
    return 0


def _demo(args) -> int:
    load = github_loader()
    syms = args.watchlist or [t for t, _ in GITHUB_TICKERS][:8]
    ref = load(syms[0]).index
    # step the as-of date forward to replay the mechanism (every ~10 trading days, last ~2y)
    steps = [ref[i] for i in range(len(ref) - 520, len(ref) - 25, 10)]
    ledger = PaperLedger()
    print(f"Forward paper-walk (DEMO replay on static 2012–17 data) — {len(syms)} names, "
          f"{len(steps)} steps\n(mechanism demo only — this data is survivorship-biased, not an edge)")
    rep = {}
    for as_of in steps:
        rep = run_paper_walk(load, syms, as_of, ledger, style=args.style, max_hold=args.max_hold)
    _print_report(rep, ledger)   # final accumulated state
    print(f"\nReplayed {len(steps)} days → {len(ledger.orders)} paper orders logged. "
          f"Run with --live behind a daily cron for the real, survivorship-free walk.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Forward paper-walk — survivorship-free evidence (paper only).")
    ap.add_argument("--live", action="store_true", help="fetch current data and log one real day")
    ap.add_argument("--demo", action="store_true", help="replay the mechanism on static data (default)")
    ap.add_argument("--watchlist", nargs="*", default=[])
    ap.add_argument("--ledger", default=str(Path.home() / ".vpts_paper.jsonl"))
    ap.add_argument("--style", default="reversion", choices=("reversion", "breakout"))
    ap.add_argument("--max-hold", type=int, default=20)
    ap.add_argument("--period", default="1y", help="(live) yfinance lookback")
    args = ap.parse_args()
    if not args.watchlist and args.live:
        print("Pass --watchlist for --live."); return 2
    return _live(args) if args.live else _demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
