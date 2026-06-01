"""Phase 1 test-suite — Volume Profile Calculator + MarketDataFetcher.

Runs **fully offline** on deterministic synthetic data (no network, no
yfinance needed), so it is safe for CI and Google Colab alike.

Run it either way::

    python tests/test_phase1.py        # pretty PASS/FAIL report + sample profile
    pytest tests/test_phase1.py        # standard pytest collection
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repo root importable whether run via pytest or directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
    MarketDataFetcher,
    VolumeProfile,
    VolumeProfileCalculator,
)

# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #
def make_clustered_ohlcv(
    n_cluster: int = 140, n_spread: int = 120, seed: int = 42
) -> pd.DataFrame:
    """Build OHLCV with volume piled up around price ~105.

    Cluster bars trade tightly around 105 with heavy volume; spread bars roam
    [100, 110] with light volume. The Point of Control should therefore land
    very close to 105. Uses a per-call seeded RNG so the data is fully
    deterministic and independent of test execution order.
    """
    rng = np.random.default_rng(seed)
    rows = []
    # Heavy cluster centred on 105.
    for _ in range(n_cluster):
        c = rng.normal(105.0, 0.5)
        rows.append((c - 0.5, c + 0.5, c, 8000.0))
    # Light, broad background.
    for _ in range(n_spread):
        c = rng.uniform(100.5, 109.5)
        rows.append((c - 0.75, c + 0.75, c, 1200.0))

    arr = np.array(rows, dtype=float)
    rng.shuffle(arr)
    idx = pd.date_range("2024-01-01", periods=len(arr), freq="D")
    return pd.DataFrame(
        {
            "Open": arr[:, 2],
            "High": arr[:, 1],
            "Low": arr[:, 0],
            "Close": arr[:, 2],
            "Volume": arr[:, 3],
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Calculator tests
# --------------------------------------------------------------------------- #
def test_core_levels_are_consistent() -> None:
    df = make_clustered_ohlcv()
    profile = VolumeProfileCalculator(num_bins=100).calculate(df, symbol="SYN")

    assert isinstance(profile, VolumeProfile)
    # POC should sit in the heavy cluster around 105.
    assert 104.0 <= profile.poc <= 106.0, profile.poc
    # Ordering & containment.
    assert profile.val <= profile.poc <= profile.vah
    assert profile.vah > profile.val
    assert profile.price_low <= profile.val
    assert profile.vah <= profile.price_high


def test_value_area_reaches_target() -> None:
    df = make_clustered_ohlcv()
    profile = VolumeProfileCalculator(num_bins=100, value_area_pct=0.70).calculate(df)
    # Expansion stops only once the target is met, so actual >= target.
    assert profile.value_area_pct_actual >= 0.70 - 1e-9
    assert profile.value_area_pct_actual <= 1.0 + 1e-9


def test_volume_is_conserved_uniform() -> None:
    df = make_clustered_ohlcv()
    profile = VolumeProfileCalculator(num_bins=100, distribution="uniform").calculate(df)
    # "uniform" must conserve total volume exactly (no leakage outside range).
    assert np.isclose(profile.total_volume, df["Volume"].sum(), rtol=1e-9)
    assert np.isclose(profile.volume_distribution.sum(), df["Volume"].sum(), rtol=1e-9)


def test_volume_is_conserved_typical() -> None:
    df = make_clustered_ohlcv()
    profile = VolumeProfileCalculator(num_bins=100, distribution="typical").calculate(df)
    assert np.isclose(profile.total_volume, df["Volume"].sum(), rtol=1e-9)
    assert 104.0 <= profile.poc <= 106.0


def test_poc_is_reported_as_hvn() -> None:
    df = make_clustered_ohlcv()
    profile = VolumeProfileCalculator(num_bins=100).calculate(df)
    assert profile.hvn, "expected at least one HVN"
    hvn_prices = [n.price for n in profile.hvn]
    assert any(abs(p - profile.poc) <= profile.bin_size for p in hvn_prices)
    # HVNs are ranked strongest-first.
    vols = [n.volume for n in profile.hvn]
    assert vols == sorted(vols, reverse=True)


def test_to_dataframe_flags() -> None:
    df = make_clustered_ohlcv()
    profile = VolumeProfileCalculator(num_bins=100).calculate(df)
    table = profile.to_dataframe()

    assert list(table.columns) == [
        "volume",
        "volume_pct",
        "in_value_area",
        "is_poc",
        "is_hvn",
        "is_lvn",
    ]
    assert int(table["is_poc"].sum()) == 1
    assert table["in_value_area"].sum() > 0
    assert np.isclose(table["volume_pct"].sum(), 100.0)


def test_lookup_helpers() -> None:
    df = make_clustered_ohlcv()
    profile = VolumeProfileCalculator(num_bins=100).calculate(df)

    assert profile.is_in_value_area(profile.poc) is True
    assert profile.location(profile.vah + 10) == "above_value"
    assert profile.location(profile.val - 10) == "below_value"
    assert profile.location(profile.poc) == "in_value"

    node = profile.nearest_node(profile.poc, kind="HVN")
    assert node is not None and node.kind == "HVN"


def test_degenerate_single_price() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "Open": 50.0,
            "High": 50.0,
            "Low": 50.0,
            "Close": 50.0,
            "Volume": 1000.0,
        },
        index=idx,
    )
    profile = VolumeProfileCalculator(num_bins=50).calculate(df)
    assert abs(profile.poc - 50.0) < 1e-3
    assert np.isclose(profile.total_volume, 30_000.0)


def test_invalid_bars_dropped_and_conserved() -> None:
    df = make_clustered_ohlcv()
    df.iloc[0, df.columns.get_loc("High")] = np.nan  # non-finite bar
    df.iloc[1, df.columns.get_loc("Volume")] = 0.0  # zero-volume bar
    df.iloc[2, df.columns.get_loc("High")] = 0.0  # High < Low bar
    df.iloc[2, df.columns.get_loc("Low")] = 100.0

    profile = VolumeProfileCalculator(num_bins=100).calculate(df)
    # Volume of the three dropped bars must not be counted.
    good = df.iloc[3:]
    assert np.isclose(profile.total_volume, good["Volume"].sum(), rtol=1e-9)
    assert profile.n_bars == len(good)


def test_no_volume_raises() -> None:
    df = make_clustered_ohlcv()
    df["Volume"] = 0.0
    try:
        VolumeProfileCalculator().calculate(df)
    except ValueError as exc:
        assert "volume" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for zero-volume data")


def test_missing_column_raises() -> None:
    df = make_clustered_ohlcv().drop(columns=["Volume"])
    try:
        VolumeProfileCalculator().calculate(df)
    except ValueError as exc:
        assert "Volume" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing column")


def test_bin_size_overrides_num_bins() -> None:
    df = make_clustered_ohlcv()
    profile = VolumeProfileCalculator(bin_size=0.25).calculate(df)
    assert np.isclose(profile.bin_size, 0.25)
    # Roughly (range / 0.25) bins.
    assert profile.num_bins >= int((profile.price_high - profile.price_low) / 0.25)


def test_value_area_target_is_configurable() -> None:
    df = make_clustered_ohlcv()
    p70 = VolumeProfileCalculator(num_bins=100, value_area_pct=0.70).calculate(df)
    p90 = VolumeProfileCalculator(num_bins=100, value_area_pct=0.90).calculate(df)
    assert p70.value_area_pct_actual >= 0.70 - 1e-9
    assert p90.value_area_pct_actual >= 0.90 - 1e-9
    assert p90.value_area_pct_target == 0.90
    # A higher target must produce an equal-or-wider value area.
    assert p90.value_area_width >= p70.value_area_width


# --- Auto-binning (ATR / range based) ------------------------------------- #
def _ramp_ohlcv(bar_h: float = 0.1, n: int = 120) -> pd.DataFrame:
    """Quiet series: price drifts 100 -> 110 in tiny steps (small ATR)."""
    centers = np.linspace(100.05, 109.95, n)
    arr = np.array([(c - bar_h / 2, c + bar_h / 2, c, 1000.0) for c in centers])
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": arr[:, 2], "High": arr[:, 1], "Low": arr[:, 0],
         "Close": arr[:, 2], "Volume": arr[:, 3]}, index=idx,
    )


def _zigzag_ohlcv(bar_h: float = 0.1, n: int = 120) -> pd.DataFrame:
    """Volatile series: price slams between 100 and 110 each bar (large ATR),
    spanning the *same* [100, 110] range as the quiet series."""
    centers = np.where(np.arange(n) % 2 == 0, 100.05, 109.95)
    arr = np.array([(c - bar_h / 2, c + bar_h / 2, c, 1000.0) for c in centers])
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": arr[:, 2], "High": arr[:, 1], "Low": arr[:, 0],
         "Close": arr[:, 2], "Volume": arr[:, 3]}, index=idx,
    )


def test_auto_bin_within_bounds_and_metadata() -> None:
    df = make_clustered_ohlcv()
    profile = VolumeProfileCalculator(
        bin_mode="auto", min_bins=20, max_bins=500
    ).calculate(df)
    assert 20 <= profile.num_bins <= 500
    assert profile.extra["bin_mode"] == "auto"
    assert profile.extra["atr"] > 0
    assert "target_bin_width" in profile.extra
    # Same range, so VA levels should still be sensible.
    assert profile.val <= profile.poc <= profile.vah


def test_auto_bin_quiet_phase_gets_finer_resolution() -> None:
    """Core promise: for an identical price range, lower volatility (quiet)
    yields *more* bins than high volatility."""
    quiet = VolumeProfileCalculator(bin_mode="auto", atr_bin_fraction=0.25).calculate(
        _ramp_ohlcv()
    )
    volatile = VolumeProfileCalculator(
        bin_mode="auto", atr_bin_fraction=0.25
    ).calculate(_zigzag_ohlcv())
    # Both cover ~[100, 110]; the quiet series must be resolved more finely.
    assert quiet.num_bins > volatile.num_bins
    assert quiet.extra["atr"] < volatile.extra["atr"]


def test_auto_bin_fraction_controls_resolution() -> None:
    df = make_clustered_ohlcv()
    fine = VolumeProfileCalculator(bin_mode="auto", atr_bin_fraction=0.10).calculate(df)
    coarse = VolumeProfileCalculator(bin_mode="auto", atr_bin_fraction=0.50).calculate(df)
    assert fine.num_bins >= coarse.num_bins


def test_auto_bin_conserves_volume() -> None:
    df = make_clustered_ohlcv()
    profile = VolumeProfileCalculator(bin_mode="auto").calculate(df)
    assert np.isclose(profile.total_volume, df["Volume"].sum(), rtol=1e-9)
    assert 104.0 <= profile.poc <= 106.0


def test_auto_bin_degenerate_single_price() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {"Open": 50.0, "High": 50.0, "Low": 50.0, "Close": 50.0, "Volume": 1000.0},
        index=idx,
    )
    profile = VolumeProfileCalculator(bin_mode="auto").calculate(df)
    assert abs(profile.poc - 50.0) < 1e-3  # must not crash on zero ATR / range


def test_compute_atr_known_values() -> None:
    high = np.array([10.0, 11.0, 12.0])
    low = np.array([9.0, 10.0, 11.0])
    close = np.array([9.5, 10.5, 11.5])
    atr = VolumeProfileCalculator._compute_atr
    assert np.isclose(atr(high, low, close, 3), (1.0 + 1.5 + 1.5) / 3.0)
    assert np.isclose(atr(high, low, close, 2), 1.5)  # mean of last two TRs


def test_invalid_bin_config_raises() -> None:
    for kwargs in (
        {"bin_mode": "bogus"},
        {"atr_bin_fraction": 0.0},
        {"min_bins": 50, "max_bins": 10},
        {"atr_period": 0},
    ):
        try:
            VolumeProfileCalculator(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {kwargs}")


# --------------------------------------------------------------------------- #
# Fetcher tests (pure helpers — no network)
# --------------------------------------------------------------------------- #
def test_clamp_period_intraday_limits() -> None:
    f = MarketDataFetcher
    assert f._clamp_period("1y", "1m") == "7d"  # 1m -> max 7 days
    assert f._clamp_period("6mo", "5m") == "1mo"  # 5m -> max 60 days
    assert f._clamp_period("max", "15m") == "1mo"  # 15m -> max 60 days
    assert f._clamp_period("5d", "1m") == "5d"  # already within limit
    assert f._clamp_period("1y", "1d") == "1y"  # daily: no cap
    assert f._clamp_period("2y", "1h") == "2y"  # 1h -> max 730 days (~2y ok)


def test_unsupported_interval_raises() -> None:
    try:
        MarketDataFetcher().fetch("AAPL", interval="3m")  # not a Yahoo interval
    except ValueError as exc:
        assert "interval" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for bad interval")


def test_cache_path_sanitises_symbol() -> None:
    path = MarketDataFetcher(cache_dir="/tmp/vpts_test")._cache_path(
        "^GDAXI", "6mo", "1d", None, None
    )
    assert "^" not in path.name
    assert path.suffix == ".pkl"
    # Deterministic for identical inputs.
    path2 = MarketDataFetcher(cache_dir="/tmp/vpts_test")._cache_path(
        "^GDAXI", "6mo", "1d", None, None
    )
    assert path.name == path2.name


# --------------------------------------------------------------------------- #
# Manual runner (pretty report)
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)  # quiet expected clamp warnings
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} Phase-1 tests …\n")
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

    # Show a sample profile so the output is tangible.
    print("\n" + "=" * 56)
    print("Sample profile on synthetic clustered data:")
    print("=" * 56)
    data = make_clustered_ohlcv()
    print(VolumeProfileCalculator(num_bins=100).calculate(
        data, symbol="SYNTHETIC", interval="1d").summary())

    print("\n" + "=" * 56)
    print("Same data, auto-binning (bin_mode='auto', ATR-scaled):")
    print("=" * 56)
    print(VolumeProfileCalculator(bin_mode="auto").calculate(
        data, symbol="SYNTHETIC", interval="1d").summary())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
