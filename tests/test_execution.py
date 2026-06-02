"""Tests for vpts.execution — first-touch resolution, fills, time-stop, ledger idempotency.

Resolution is the bug-prone part (a wrong outcome = a wrong track record), so it is
checked on hand-built bars with known answers: long/short win & loss, stop-first ties,
the next-open fill, the time-stop, and "not enough data yet → stays open".

    python tests/test_execution.py
    pytest tests/test_execution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.execution import PaperLedger, PaperOrder, resolve_order, run_paper_walk  # noqa: E402


def _df(bars: list[tuple[float, float, float, float]], start: str = "2024-01-01") -> pd.DataFrame:
    """bars = list of (open, high, low, close); daily DatetimeIndex from *start*."""
    idx = pd.date_range(start, periods=len(bars), freq="B")
    a = np.array(bars, float)
    return pd.DataFrame({"Open": a[:, 0], "High": a[:, 1], "Low": a[:, 2], "Close": a[:, 3],
                         "Volume": np.full(len(bars), 1e6)}, index=idx)


def _order(side: int, stop: float, target: float, max_hold: int = 5) -> PaperOrder:
    return PaperOrder(symbol="T", decision_date="2024-01-01", side=side,
                      stop=stop, target=target, max_hold=max_hold)


def test_long_win_fills_next_open_and_hits_target() -> None:
    # decision bar 0; fill at bar 1 open (=100); bar 2 prints the target.
    df = _df([(100, 101, 99, 100), (100, 102, 98, 101), (101, 110, 100, 108)])
    o = resolve_order(_order(+1, stop=90.0, target=110.0), df)
    assert o.status == "win" and o.fill_price == 100.0
    assert o.fill_date == "2024-01-02" and o.exit_price == 110.0
    assert o.holding_bars == 2                       # fill bar (1) + 1 → bar 2
    assert abs(o.r_multiple - 1.0) < 1e-9            # risk 10, reward 10 → +1R


def test_long_loss_hits_stop() -> None:
    df = _df([(100, 101, 99, 100), (100, 101, 88, 90), (90, 95, 85, 88)])
    o = resolve_order(_order(+1, stop=90.0, target=120.0), df)
    assert o.status == "loss" and o.exit_price == 90.0
    assert abs(o.r_multiple - (-1.0)) < 1e-9         # exit at stop → −1R


def test_short_win_and_loss() -> None:
    # short: fill 100, target 90 (price down = win), stop 110 (price up = loss).
    win = _df([(100, 101, 99, 100), (100, 101, 99, 100), (99, 100, 89, 91)])
    o = resolve_order(_order(-1, stop=110.0, target=90.0), win)
    assert o.status == "win" and o.exit_price == 90.0 and o.r_multiple > 0
    loss = _df([(100, 101, 99, 100), (100, 101, 99, 100), (101, 112, 100, 110)])
    o2 = resolve_order(_order(-1, stop=110.0, target=80.0), loss)
    assert o2.status == "loss" and o2.exit_price == 110.0


def test_stop_wins_same_bar_tie() -> None:
    # one bar straddles both stop and target → conservative: count the stop (loss).
    df = _df([(100, 101, 99, 100), (100, 115, 85, 100), (100, 100, 100, 100)])
    o = resolve_order(_order(+1, stop=90.0, target=110.0), df)
    assert o.status == "loss"


def test_timeout_at_horizon_close() -> None:
    # never touches stop/target within max_hold=2 bars → timeout at the 2nd held bar's close.
    df = _df([(100, 101, 99, 100), (100, 102, 99, 101), (101, 103, 100, 102), (102, 103, 101, 102.5)])
    o = resolve_order(_order(+1, stop=80.0, target=130.0, max_hold=2), df)
    assert o.status == "timeout"
    # fill bar1 (2024-01-02); 2 bars held = bars 1,2 → time-stop at bar2 close (2024-01-03).
    assert o.exit_date == "2024-01-03" and o.exit_price == 102.0
    assert o.holding_bars == 2


def test_stays_open_without_enough_future_bars() -> None:
    # only the fill bar exists, no touch, horizon not elapsed → unresolved.
    df = _df([(100, 101, 99, 100), (100, 101, 99, 100)])
    o = resolve_order(_order(+1, stop=80.0, target=130.0, max_hold=5), df)
    assert o.status == "open" and o.r_multiple is None


def test_ledger_idempotent_upsert_and_roundtrip(tmp_path) -> None:
    led = PaperLedger()
    o = _order(+1, 90.0, 110.0)
    assert led.upsert(o) is True
    assert led.upsert(_order(+1, 90.0, 110.0)) is False     # same (symbol, decision_date)
    p = tmp_path / "ledger.jsonl"
    led.save(p)
    again = PaperLedger.load(p)
    assert len(again.orders) == 1 and again.orders[0].symbol == "T"


def test_run_paper_walk_no_lookahead_and_records(tmp_path) -> None:
    # A constructed uptrend; decision uses only bars <= as_of, then logs/resolves.
    rng = np.random.default_rng(0)
    n = 200
    close = 50 + np.cumsum(rng.normal(0.05, 0.4, n))
    df = pd.DataFrame({"Open": close, "High": close + 0.6, "Low": close - 0.6,
                       "Close": close, "Volume": 1e6}, index=pd.date_range("2023-01-01", periods=n, freq="B"))
    as_of = df.index[150]
    led = PaperLedger()
    rep = run_paper_walk(lambda s: df, ["T"], as_of, led, max_hold=10)
    # decisions only ever dated on/before as_of (no look-ahead)
    assert all(pd.Timestamp(o.decision_date) <= as_of for o in led.orders)
    assert rep["as_of"] == as_of.date().isoformat()
    assert isinstance(rep["resolved"], int)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
