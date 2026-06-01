"""Phase 2 test-suite — Quiet-Phase Detector, Volume Patterns & indicators.

Runs **fully offline** on deterministic synthetic data (no network). Each
fixture is built with a deliberate structure (warm-up → quiet coil → expansion)
so the regime detectors have something unambiguous to find.

    python tests/test_phase2.py     # pretty PASS/FAIL report + sample read-outs
    pytest tests/test_phase2.py     # standard pytest collection
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
    QuietPhaseDetector,
    VolumePatternDetector,
    VolumePatternType,
    VolumeProfileCalculator,
)
from vpts.regime import indicators as ind  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #
def make_regime_data(seed: int = 5) -> pd.DataFrame:
    """Warm-up (normal) → quiet coil (dry, tight) → expansion (with a climax).

    Designed so a quiet phase, a volume dry-up, accumulation, divergence and a
    climax are all present.
    """
    rng = np.random.default_rng(seed)
    n_w, n_q, n_e = 60, 45, 45
    cw = 100 + np.cumsum(rng.normal(0, 0.4, n_w))
    cq = cw[-1] + np.cumsum(rng.normal(0.0, 0.06, n_q))     # flat coil
    ce = cq[-1] + np.cumsum(rng.normal(0.3, 1.0, n_e))      # breakout up
    close = np.concatenate([cw, cq, ce])
    ranges = np.concatenate([np.full(n_w, 1.0), np.full(n_q, 0.2), np.full(n_e, 1.8)])
    high = close + ranges / 2
    low = close - ranges / 2
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = np.concatenate([
        rng.integers(2500, 3500, n_w).astype(float),       # normal
        rng.integers(700, 1100, n_q).astype(float),        # dried up
        rng.integers(3000, 5000, n_e).astype(float),       # expansion
    ])
    ci = n_w + n_q + 20                                     # climax bar
    vol[ci] *= 6.0
    high[ci] = close[ci] + 4.5
    low[ci] = close[ci] - 4.5
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def make_trend_data(price_up: bool, volume_up: bool, n: int = 50) -> pd.DataFrame:
    """A clean linear price/volume trend for divergence tests."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.linspace(100, 120 if price_up else 80, n)
    vol = np.linspace(1000, 4000 if volume_up else 200, n)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Indicator tests
# --------------------------------------------------------------------------- #
def test_true_range_first_bar_is_high_low() -> None:
    high = pd.Series([10.0, 11.0])
    low = pd.Series([9.0, 9.5])
    close = pd.Series([9.5, 10.5])
    tr = ind.true_range(high, low, close)
    assert np.isclose(tr.iloc[0], 1.0)
    assert np.isclose(tr.iloc[1], 1.5)  # max(1.5, |11-9.5|, |9.5-9.5|)


def test_atr_wilder_and_sma() -> None:
    high = pd.Series([10.0, 11.0])
    low = pd.Series([9.0, 9.5])
    close = pd.Series([9.5, 10.5])
    wilder = ind.atr(high, low, close, period=2, method="wilder")
    sma = ind.atr(high, low, close, period=2, method="sma")
    assert np.isclose(wilder.iloc[1], 1.25)  # 0.5*1.5 + 0.5*1.0
    assert np.isclose(sma.iloc[1], 1.25)


def test_rolling_percentile_rank() -> None:
    s = pd.Series([3.0, 1.0, 2.0])
    rank = ind.rolling_percentile_rank(s, window=3, min_periods=1)
    assert np.isclose(rank.iloc[0], 1.0)        # only value
    assert np.isclose(rank.iloc[1], 0.5)        # 1 of {3,1} <= 1
    assert np.isclose(rank.iloc[2], 2 / 3)      # 2 of {3,1,2} <= 2


def test_rolling_slope_sign() -> None:
    up = ind.rolling_slope(pd.Series([0.0, 1.0, 2.0, 3.0]), window=4)
    flat = ind.rolling_slope(pd.Series([5.0, 5.0, 5.0]), window=3)
    assert np.isclose(up.iloc[-1], 1.0)
    assert np.isclose(flat.iloc[-1], 0.0)


def test_bollinger_bandwidth_zero_on_constant() -> None:
    bw = ind.bollinger_bandwidth(pd.Series([10.0] * 5), window=3)
    assert np.isclose(bw.iloc[-1], 0.0)


def test_ensure_ohlcv_guards() -> None:
    good = make_trend_data(True, True)
    ind.ensure_ohlcv(good, min_bars=10)  # should not raise
    for bad_kwargs in (
        dict(df=good.drop(columns=["Volume"]), min_bars=10),
        dict(df=good.iloc[:5], min_bars=30),
    ):
        try:
            ind.ensure_ohlcv(**bad_kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError from ensure_ohlcv")


# --------------------------------------------------------------------------- #
# Quiet-phase tests
# --------------------------------------------------------------------------- #
def test_quiet_phase_scores_coil_higher_than_expansion() -> None:
    df = make_regime_data()
    res = QuietPhaseDetector().detect(df, symbol="SYN", interval="1d")
    coil = res.frame["quiet_score"].iloc[70:104].mean()
    expansion = res.frame["quiet_score"].iloc[-30:].mean()
    assert coil > expansion + 20         # clear separation
    assert res.frame["is_quiet"].iloc[70:104].mean() > 0.7   # coil mostly quiet
    assert not res.latest.is_quiet        # series ends in expansion
    assert len(res.quiet_segments()) >= 1


def test_quiet_phase_result_shape_and_helpers() -> None:
    df = make_regime_data()
    res = QuietPhaseDetector().detect(df)
    for col in ("atr", "quiet_score", "is_quiet", "bars_in_state",
                "score_volatility", "score_volume", "score_range"):
        assert col in res.frame.columns
    assert 0.0 <= res.latest.quiet_score <= 100.0
    assert isinstance(res.summary(), str) and "Quiet-Phase" in res.summary()
    # bars_in_state never exceeds the longest quiet segment.
    longest = max((s["length"] for s in res.quiet_segments()), default=0)
    assert res.frame["bars_in_state"].max() <= longest


def test_quiet_phase_is_deterministic() -> None:
    df = make_regime_data()
    a = QuietPhaseDetector().detect(df).frame["quiet_score"]
    b = QuietPhaseDetector().detect(df).frame["quiet_score"]
    pd.testing.assert_series_equal(a, b)


def test_quiet_phase_invalid_config() -> None:
    for kwargs in (
        {"quiet_threshold": 0.0},
        {"quiet_threshold": 100.0},
        {"weights": (0.0, 0.0, 0.0)},
        {"atr_period": 1},
    ):
        try:
            QuietPhaseDetector(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {kwargs}")


# --------------------------------------------------------------------------- #
# Volume-pattern tests
# --------------------------------------------------------------------------- #
def test_patterns_rich_fixture_finds_all_families() -> None:
    df = make_regime_data()
    profile = VolumeProfileCalculator(num_bins=80).calculate(df)
    res = VolumePatternDetector().detect(df, profile=profile, symbol="SYN")

    assert len(res.of_type(VolumePatternType.DRY_UP)) >= 1
    assert len(res.of_type(VolumePatternType.ACCUMULATION)) >= 1
    assert len(res.of_type(VolumePatternType.CLIMAX)) >= 1
    # Profile anchoring: at least one event sits at a named level.
    assert any(p.at_level for p in res.patterns)
    # Helpers.
    assert res.latest is not None
    assert len(res.recent(3)) <= 3
    assert "Volume Patterns" in res.summary()
    # Every dry-up run respects the minimum-run setting.
    for p in res.of_type(VolumePatternType.DRY_UP):
        assert p.n_bars >= VolumePatternDetector().dryup_min_run


def test_climax_anchors_to_poc() -> None:
    rng = np.random.default_rng(0)
    n = 80
    close = 100 + rng.normal(0, 0.1, n)
    high = close + 0.4
    low = close - 0.4
    vol = np.full(n, 1000.0)
    ci = 70
    close[ci] = 100.0
    high[ci] = 104.5
    low[ci] = 95.5            # wide bar
    vol[ci] = 9000.0          # climax volume
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )
    profile = VolumeProfileCalculator(num_bins=50).calculate(df)
    res = VolumePatternDetector().detect(df, profile=profile)
    climaxes = res.of_type(VolumePatternType.CLIMAX)
    assert climaxes, "expected a climax event"
    assert any(p.at_level and p.at_level.startswith("POC") for p in climaxes)


def test_bearish_divergence_detected() -> None:
    df = make_trend_data(price_up=True, volume_up=False)
    res = VolumePatternDetector(min_bars=30).detect(df)
    divs = res.of_type(VolumePatternType.DIVERGENCE)
    assert divs, "expected a divergence"
    assert all(p.direction == "bearish" for p in divs)


def test_no_divergence_when_volume_confirms() -> None:
    df = make_trend_data(price_up=True, volume_up=True)
    res = VolumePatternDetector(min_bars=30).detect(df)
    assert len(res.of_type(VolumePatternType.DIVERGENCE)) == 0


def test_patterns_without_profile_have_no_levels() -> None:
    df = make_regime_data()
    res = VolumePatternDetector().detect(df)  # no profile
    assert all(p.at_level is None for p in res.patterns)


def test_patterns_min_bars_guard() -> None:
    df = make_regime_data().iloc[:10]
    try:
        VolumePatternDetector().detect(df)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for too-few bars")


def test_patterns_invalid_config() -> None:
    for kwargs in (
        {"dryup_factor": 1.5},
        {"climax_volume_factor": 0.9},
        {"divergence_min_run": 0},
        {"accumulation_flat_atr": 0.0},
    ):
        try:
            VolumePatternDetector(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {kwargs}")


# --------------------------------------------------------------------------- #
# Manual runner
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} Phase-2 tests …\n")
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
    print("Sample read-outs on the synthetic regime fixture:")
    print("=" * 56)
    df = make_regime_data()
    profile = VolumeProfileCalculator(num_bins=80).calculate(df, symbol="SYN")
    print(QuietPhaseDetector().detect(df, symbol="SYN", interval="1d").summary())
    print()
    print(VolumePatternDetector().detect(df, profile=profile, symbol="SYN",
                                         interval="1d").summary())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
