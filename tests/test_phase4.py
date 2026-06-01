"""Phase 4 test-suite — Signal Generator.

Fully offline and deterministic.

    python tests/test_phase4.py     # pretty PASS/FAIL report + sample signals
    pytest tests/test_phase4.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
    ConfluenceScorer,
    QuietPhaseDetector,
    SignalAction,
    SignalGenerator,
    VolumePatternDetector,
    VolumeProfileCalculator,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def regime_df(seed: int = 5) -> pd.DataFrame:
    """Warm-up → quiet coil → bearish expansion (used for SHORT / NO_TRADE)."""
    rng = np.random.default_rng(seed)
    n_w, n_q, n_e = 60, 45, 45
    cw = 100 + np.cumsum(rng.normal(0, 0.4, n_w))
    cq = cw[-1] + np.cumsum(rng.normal(0.0, 0.06, n_q))
    ce = cq[-1] + np.cumsum(rng.normal(0.3, 1.0, n_e))
    close = np.concatenate([cw, cq, ce])
    ranges = np.concatenate([np.full(n_w, 1.0), np.full(n_q, 0.2), np.full(n_e, 1.8)])
    high, low = close + ranges / 2, close - ranges / 2
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = np.concatenate([
        rng.integers(2500, 3500, n_w).astype(float),
        rng.integers(700, 1100, n_q).astype(float),
        rng.integers(3000, 5000, n_e).astype(float),
    ])
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def long_df(seed: int = 3) -> pd.DataFrame:
    """A bullish reversion setup: a quiet pullback to a support shelf below value."""
    rng = np.random.default_rng(seed)
    body = 108 + rng.normal(0, 1.3, 70)
    shelf = 100 + rng.normal(0, 0.30, 40)
    pull = np.concatenate([np.linspace(107, 100.8, 12), 100.7 + rng.normal(0, 0.10, 33)])
    close = np.concatenate([body, shelf, pull])
    ranges = np.concatenate([np.full(70, 2.2), np.full(40, 0.9), np.full(45, 0.7)])
    high, low = close + ranges / 2, close - ranges / 2
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = np.concatenate([
        rng.integers(4000, 6000, 70).astype(float),
        rng.integers(4000, 6000, 40).astype(float),
        rng.integers(900, 1400, 45).astype(float),
    ])
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Actionable signals
# --------------------------------------------------------------------------- #
def test_long_reversion_is_actionable_with_valid_geometry() -> None:
    sig = SignalGenerator(style="reversion", min_quality=35, min_abs_bias=12).analyze(
        long_df(), symbol="L", interval="1d"
    )
    assert sig.action == SignalAction.LONG
    assert sig.is_actionable and sig.direction == 1
    assert sig.stop < sig.entry < sig.targets[0]
    assert all(t > sig.entry for t in sig.targets)
    assert sig.risk_reward_ratio >= 1.5
    assert sig.suggested_size > 0


def test_short_reversion_is_actionable_with_valid_geometry() -> None:
    sig = SignalGenerator(style="reversion", min_quality=30, min_abs_bias=15).analyze(
        regime_df(), symbol="S", interval="1d"
    )
    assert sig.action == SignalAction.SHORT
    assert sig.stop > sig.entry > sig.targets[0]
    assert all(t < sig.entry for t in sig.targets)
    assert sig.risk_reward_ratio >= 1.5


def test_breakout_short_geometry() -> None:
    sig = SignalGenerator(style="breakout", min_quality=30, min_abs_bias=15).analyze(
        regime_df(), symbol="S", interval="1d"
    )
    assert sig.action == SignalAction.SHORT
    assert sig.stop > sig.entry            # stop back inside the coil
    assert all(t < sig.entry for t in sig.targets)
    assert sig.risk_reward_ratio >= 1.5


# --------------------------------------------------------------------------- #
# Gating / NO_TRADE paths
# --------------------------------------------------------------------------- #
def test_no_trade_when_conviction_too_low() -> None:
    # The coil's bias (~+12) is below the default min_abs_bias (15).
    sig = SignalGenerator(style="reversion", min_quality=35).analyze(
        regime_df().iloc[:105]
    )
    assert sig.action == SignalAction.NO_TRADE
    assert not sig.is_actionable
    assert sig.entry is None and sig.targets == ()
    assert "conviction" in sig.rationale


def test_no_trade_when_quality_too_low() -> None:
    sig = SignalGenerator(min_quality=95).analyze(regime_df())
    assert sig.action == SignalAction.NO_TRADE
    assert "quality" in sig.rationale


def test_no_trade_when_rr_below_minimum() -> None:
    sig = SignalGenerator(style="reversion", min_quality=35, min_abs_bias=12,
                          min_rr=100.0).analyze(long_df())
    assert sig.action == SignalAction.NO_TRADE
    assert "risk:reward" in sig.rationale


def test_require_quiet_blocks_active_market() -> None:
    sig = SignalGenerator(style="reversion", min_quality=30, min_abs_bias=15,
                          require_quiet=True).analyze(regime_df())
    assert sig.action == SignalAction.NO_TRADE
    assert "quiet" in sig.rationale


# --------------------------------------------------------------------------- #
# Risk model / helpers
# --------------------------------------------------------------------------- #
def test_position_sizing_matches_fixed_fractional() -> None:
    gen = SignalGenerator(style="reversion", min_quality=35, min_abs_bias=12,
                          risk_fraction=0.01)
    sig = gen.analyze(long_df(), account_equity=50_000.0)
    assert sig.is_actionable
    assert np.isclose(sig.risk_amount, 500.0)            # 1% of 50k
    expected = math.floor(sig.risk_amount / sig.risk_per_unit)
    assert sig.suggested_size == expected
    assert sig.account_equity == 50_000.0


def test_risk_reward_and_size_are_exposed() -> None:
    sig = SignalGenerator(style="reversion", min_quality=35, min_abs_bias=12).analyze(long_df())
    assert isinstance(sig.risk_reward_ratio, float) and sig.risk_reward_ratio > 0
    assert isinstance(sig.suggested_size, float) and sig.suggested_size >= 0
    assert sig.final_target == sig.targets[-1]


def test_explain_and_serialisation() -> None:
    sig = SignalGenerator(style="reversion", min_quality=35, min_abs_bias=12).analyze(
        long_df(), symbol="L", interval="1d"
    )
    text = sig.explain()
    assert "TRADE SIGNAL" in text and "Entry" in text and "R:R" in text
    assert str(sig)  # one-liner non-empty
    json.dumps(sig.as_dict())  # JSON-serialisable

    nt = SignalGenerator(min_quality=99).analyze(regime_df())
    assert "NO TRADE" in nt.explain()


def test_analyze_matches_from_score() -> None:
    df = long_df()
    profile = VolumeProfileCalculator().calculate(df)
    quiet = QuietPhaseDetector().detect(df)
    patterns = VolumePatternDetector().detect(df, profile=profile)
    score = ConfluenceScorer().score(df, profile, quiet, patterns)
    gen = SignalGenerator(style="reversion", min_quality=35, min_abs_bias=12)
    a = gen.from_score(score, profile)
    b = gen.analyze(df)
    assert a.as_dict() == b.as_dict()


def test_determinism() -> None:
    df = long_df()
    gen = SignalGenerator(style="reversion", min_quality=35, min_abs_bias=12)
    assert gen.analyze(df).as_dict() == gen.analyze(df).as_dict()


def test_invalid_config() -> None:
    for kwargs in (
        {"style": "bogus"},
        {"min_rr": 0.0},
        {"risk_fraction": 0.0},
        {"risk_fraction": 1.0},
        {"account_equity": 0.0},
        {"min_quality": 150.0},
    ):
        try:
            SignalGenerator(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_no_trade_when_risk_budget_too_small_to_size() -> None:
    """Bug #3: when the risk budget can't afford even one unit, the signal must
    be NO_TRADE — not an actionable signal with suggested_size == 0."""
    sig = SignalGenerator(style="reversion", min_quality=35, min_abs_bias=12).analyze(
        long_df(), account_equity=10.0  # $10 account -> $0.10 risk budget
    )
    assert sig.action == SignalAction.NO_TRADE
    assert "budget" in sig.rationale
    assert not sig.is_actionable
    assert sig.suggested_size is None


# --------------------------------------------------------------------------- #
# Manual runner
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} Phase-4 tests …\n")
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
    print("Sample signals:")
    print("=" * 56)
    print(SignalGenerator(style="reversion", min_quality=35, min_abs_bias=12).analyze(
        long_df(), symbol="LONGDEMO", interval="1d").explain())
    print()
    print(SignalGenerator(style="reversion", min_quality=30).analyze(
        regime_df(), symbol="SHORTDEMO", interval="1d").explain())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
