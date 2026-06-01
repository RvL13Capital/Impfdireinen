"""Phase 6 test-suite — Backtester.

Fully offline and deterministic. Emphasis on the two correctness guarantees that
matter most for a backtest: **no look-ahead** and **cost realism**.

    python tests/test_phase6.py
    pytest tests/test_phase6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
    Backtester,
    CostModel,
    SignalAction,
    SignalGenerator,
    TradeSignal,
)


# --------------------------------------------------------------------------- #
def oscillating_df(n: int = 260, seed: int = 7) -> pd.DataFrame:
    """Mean-reverting oscillation so reversion long/short signals recur."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 6 * np.sin(t / 22.0) + 3 * np.sin(t / 7.0) + rng.normal(0, 0.6, n)
    ranges = 1.0 + 0.5 * np.abs(np.sin(t / 22.0))
    high = close + ranges / 2 + np.abs(rng.normal(0, 0.2, n))
    low = close - ranges / 2 - np.abs(rng.normal(0, 0.2, n))
    open_ = np.concatenate([[close[0]], close[:-1]]) + rng.normal(0, 0.15, n)
    vol = (3000 + 1500 * np.cos(t / 22.0) + rng.integers(-400, 400, n)).astype(float)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def flat_df(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": 100.0, "High": 100.2, "Low": 99.8, "Close": 100.0, "Volume": 1000.0},
        index=idx,
    )


def _bt(**kw) -> Backtester:
    kw.setdefault("lookback", 80)
    kw.setdefault("signal_generator",
                  SignalGenerator(style="reversion", min_quality=30, min_abs_bias=10))
    return Backtester(**kw)


# --------------------------------------------------------------------------- #
def test_runs_and_conserves_equity() -> None:
    res = _bt().run(oscillating_df(), symbol="OSC", interval="1d")
    assert res.n_trades > 0
    # Equity conservation: final == initial + sum of realised PnL.
    assert np.isclose(res.final_equity,
                      res.initial_equity + sum(t.pnl for t in res.trades))
    assert np.isclose(res.equity_curve.iloc[-1], res.final_equity)


def test_no_lookahead_entry_is_next_open() -> None:
    """Every entry fills at the (cost-adjusted) OPEN of its entry bar — proof the
    engine acts only on the *next* bar after a decision, never on future data."""
    df = oscillating_df()
    res = _bt(cost_model=CostModel(slippage_bps=5)).run(df)
    adverse = 5 / 1e4
    assert res.n_trades > 0
    for tr in res.trades:
        if tr.exit_reason == "end" and tr.entry_time is None:
            continue
        o = float(df["Open"].loc[tr.entry_time])
        side = 1 if tr.side == "long" else -1
        assert np.isclose(tr.entry_price, o * (1 + side * adverse), rtol=1e-9)


def test_costs_reduce_returns() -> None:
    df = oscillating_df()
    no_cost = _bt(cost_model=CostModel(slippage_bps=0.0)).run(df)
    high_cost = _bt(cost_model=CostModel(slippage_bps=50.0, commission_bps=5.0)).run(df)
    assert high_cost.final_equity < no_cost.final_equity


def test_stats_are_internally_consistent() -> None:
    res = _bt().run(oscillating_df())
    assert res.n_trades == len(res.trades)
    assert 0.0 <= res.win_rate <= 1.0
    assert res.profit_factor >= 0.0
    assert res.max_drawdown_pct <= 0.0
    assert np.isclose(res.expectancy, np.mean([t.pnl for t in res.trades]))
    assert np.isfinite(res.sharpe)


def test_flat_data_produces_no_trades() -> None:
    res = _bt().run(flat_df())
    assert res.n_trades == 0
    assert np.isclose(res.final_equity, res.initial_equity)
    assert res.max_drawdown_pct == 0.0
    assert res.profit_factor == 0.0
    assert res.win_rate == 0.0


def test_time_stop_caps_holding() -> None:
    k = 3
    res = _bt(max_hold_bars=k).run(oscillating_df())
    assert res.n_trades > 0
    # No trade may be held longer than the time stop …
    assert all(t.bars_held <= k for t in res.trades)
    # … and the time stop must actually fire on this data.
    assert any(t.exit_reason == "time" for t in res.trades)


def test_dataframes_and_serialisation() -> None:
    res = _bt().run(oscillating_df(), symbol="OSC", interval="1d")
    blotter = res.trades_dataframe()
    assert len(blotter) == res.n_trades
    assert {"side", "entry_price", "exit_price", "pnl", "exit_reason"} <= set(blotter.columns)
    json.dumps(res.as_dict())
    assert "Backtest" in res.summary()


def test_determinism() -> None:
    df = oscillating_df()
    a, b = _bt().run(df), _bt().run(df)
    assert a.as_dict() == b.as_dict()
    pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)


def test_equity_curve_figure_builds() -> None:
    try:
        import plotly.graph_objects as go
        from vpts.dashboard import charts
    except Exception:  # pragma: no cover - plotly not installed
        print("      (skipped: plotly not available)")
        return
    res = _bt().run(oscillating_df(), symbol="OSC", interval="1d")
    fig = charts.equity_curve_figure(res)
    assert isinstance(fig, go.Figure)
    assert any(isinstance(t, go.Scatter) for t in fig.data)
    assert isinstance(fig.to_dict(), dict)


def test_min_bars_guard() -> None:
    short = oscillating_df(n=50)
    try:
        Backtester(lookback=80).run(short)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for too-few bars")


def test_invalid_config() -> None:
    for cls, kwargs in (
        (Backtester, {"lookback": 10}),
        (Backtester, {"initial_equity": 0.0}),
        (Backtester, {"recompute_stride": 0}),
        (Backtester, {"max_hold_bars": 0}),
        (CostModel, {"slippage_bps": -1.0}),
    ):
        try:
            cls(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {cls.__name__} {kwargs}")


# --------------------------------------------------------------------------- #
# Regression tests for fixed bugs
# --------------------------------------------------------------------------- #
def _pending(action: SignalAction, stop: float, target: float) -> TradeSignal:
    return TradeSignal(action=action, style="reversion", price=stop, rationale="t",
                       stop=stop, targets=(target,))


def test_no_entry_when_open_gaps_through_stop() -> None:
    """Bug #1: a next-bar open that gaps past the stop must NOT open a position
    (and must not fabricate a same-bar fill at the stale stop)."""
    bt = Backtester()
    long_sig = _pending(SignalAction.LONG, stop=100.0, target=110.0)
    # Open gaps to 95 (below the long's stop) -> setup invalid -> no position.
    assert bt._try_open(long_sig, 95.0, 10_000.0, None, 5) is None
    # Normal open above the stop -> opens with entry strictly above the stop.
    pos = bt._try_open(long_sig, 101.0, 10_000.0, None, 5)
    assert pos is not None and pos.entry_price > pos.stop and pos.size > 0

    short_sig = _pending(SignalAction.SHORT, stop=100.0, target=90.0)
    # Open gaps to 105 (above the short's stop) -> no position.
    assert bt._try_open(short_sig, 105.0, 10_000.0, None, 5) is None
    pos2 = bt._try_open(short_sig, 99.0, 10_000.0, None, 5)
    assert pos2 is not None and pos2.entry_price < pos2.stop


def test_position_size_capped_at_cash_no_leverage() -> None:
    """Bug #7: a very tight stop must not lever the account — notional <= cash."""
    bt = Backtester()  # risk_fraction 0.01
    cash = 10_000.0
    sig = _pending(SignalAction.LONG, stop=99.99, target=110.0)  # ~0.06 risk/unit
    pos = bt._try_open(sig, 100.0, cash, None, 5)
    assert pos is not None
    assert pos.size * pos.entry_price <= cash + 1e-6        # no leverage
    assert pos.size == float(np.floor(cash / pos.entry_price))  # cash cap is binding


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} Phase-6 tests …\n")
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"  ✗ {t.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__} (error)\n      {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed, {failed} failed.")

    print("\n" + "=" * 56)
    print("Sample backtest (oscillating synthetic, reversion):")
    print("=" * 56)
    print(_bt().run(oscillating_df(), symbol="OSC", interval="1d").summary())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
