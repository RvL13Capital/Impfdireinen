"""Tests for vpts.structure — synthetic delta, profile shape, footprints, decay.

Math is checked on hand-constructed inputs with known answers; the dataset
builder is checked for shape, finiteness and no-look-ahead, then fed to the
real CPCV factor harness.

    python tests/test_structure.py
    pytest tests/test_structure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
    STRUCTURAL_FEATURES,
    VolumeProfileCalculator,
    build_structural_dataset,
    build_structural_meta_dataset,
    classify_shape,
    close_location_value,
    cpcv_factor_eval,
    cpcv_meta_eval,
    decayed_poc,
    detect_ledges,
    poor_high,
    synthetic_delta_stats,
)
from vpts.ml.models import FactorCVResult, FactorDataset, MetaCVResult, MetaDataset  # noqa: E402
from vpts.profile.models import VolumeProfile  # noqa: E402
from vpts.structure.analytics import (  # noqa: E402
    SHAPE_B,
    SHAPE_D,
    SHAPE_P,
    SHAPE_b,
    double_distribution,
    poc_location,
    value_area_compression_ratio,
    weighted_moments,
)


def _profile_from_hist(hist: np.ndarray, lo: float = 0.0, hi: float = None) -> VolumeProfile:
    """Build a real VolumeProfile around a given histogram (for shape/footprint tests)."""
    hist = np.asarray(hist, float)
    n = hist.size
    hi = float(n - 1) if hi is None else hi
    edges = np.linspace(lo, hi, n + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_i = int(np.argmax(hist))
    total = float(hist.sum())
    return VolumeProfile(
        poc=float(centers[poc_i]), vah=float(centers[min(poc_i + 1, n - 1)]),
        val=float(centers[max(poc_i - 1, 0)]), poc_volume=float(hist[poc_i]),
        value_area_volume=total * 0.7, value_area_pct_target=0.7, value_area_pct_actual=0.7,
        hvn=(), lvn=(), bin_edges=edges, bin_centers=centers, volume_distribution=hist,
        total_volume=total, bin_size=float(edges[1] - edges[0]), num_bins=n,
        price_low=lo, price_high=hi, distribution_method="uniform", n_bars=100)


def _ohlcv(n: int = 420, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 60 + np.cumsum(rng.normal(0.02, 0.5, n)) + 5 * np.sin(t / 30)
    close = np.maximum(close, 5)
    rngb = 0.5 + 0.3 * np.abs(np.sin(t / 15))
    high, low = close + rngb, close - rngb
    op = np.concatenate([[close[0]], close[:-1]])
    vol = (2e6 + 5e5 * np.cos(t / 20) + rng.integers(-2e5, 2e5, n)).astype(float)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close,
                         "Volume": vol}, index=idx)


# --------------------------------------------------------------------------- #
# Synthetic delta
# --------------------------------------------------------------------------- #
def test_clv_bounds_and_sign() -> None:
    clv = close_location_value(np.array([10., 10, 10, 10]),
                               np.array([8., 8, 8, 10]),     # last bar zero-range
                               np.array([10., 8, 9, 10]))
    assert np.allclose(clv, [1.0, -1.0, 0.0, 0.0])           # high, low, mid, zero-range→0
    assert clv.min() >= -1 and clv.max() <= 1


def test_synthetic_delta_dominance() -> None:
    # All bars close on their high -> net delta ~ +1 and positive at the POC.
    n = 60
    high = np.full(n, 50.0); low = np.full(n, 49.0); close = high.copy()
    vol = np.full(n, 1e6)
    df = pd.DataFrame({"Open": low, "High": high, "Low": low, "Close": close, "Volume": vol})
    prof = VolumeProfileCalculator(num_bins=20).calculate(df)
    net, poc = synthetic_delta_stats(high, low, close, vol, prof)
    assert net > 0.9 and poc > 0.9
    # Flip: all close on the low -> net ~ -1.
    net2, _ = synthetic_delta_stats(high, low, low.copy(), vol, prof)
    assert net2 < -0.9


# --------------------------------------------------------------------------- #
# Profile shape
# --------------------------------------------------------------------------- #
def test_weighted_moments_skew_sign() -> None:
    centers = np.linspace(1, 10, 10)
    top = np.array([1, 1, 1, 1, 1, 2, 5, 12, 20, 15.])       # mass high -> long LOW tail
    assert weighted_moments(centers, top)[2] < -0.5
    assert weighted_moments(centers, top[::-1].copy())[2] > 0.5
    sym = np.array([1, 3, 8, 16, 22, 22, 16, 8, 3, 1.])      # symmetric
    assert abs(weighted_moments(centers, sym)[2]) < 0.2


def test_classify_shape() -> None:
    # P: top-heavy (long low tail, POC near high).
    p_hist = np.array([1, 1, 1, 1, 2, 3, 6, 12, 22, 30.])
    pp = _profile_from_hist(p_hist)
    assert poc_location(pp) > 0.7 and classify_shape(pp) == SHAPE_P
    # b: bottom-heavy (mirror).
    assert classify_shape(_profile_from_hist(p_hist[::-1].copy())) == SHAPE_b
    # D: symmetric single bell.
    d_hist = np.array([1, 3, 8, 16, 22, 22, 16, 8, 3, 1.])
    assert classify_shape(_profile_from_hist(d_hist)) == SHAPE_D
    # B: clean double distribution.
    b_hist = np.array([1, 8, 10, 8, 2, 0.5, 0.5, 2, 8, 10, 8, 1.])
    assert classify_shape(_profile_from_hist(b_hist)) == SHAPE_B


def test_double_distribution() -> None:
    bimodal = _profile_from_hist(np.array([1, 8, 10, 8, 2, 0.5, 0.5, 2, 8, 10, 8, 1.]))
    assert double_distribution(bimodal)[0] is True
    unimodal = _profile_from_hist(np.array([1, 3, 6, 10, 14, 10, 6, 3, 1.]))
    assert double_distribution(unimodal)[0] is False


# --------------------------------------------------------------------------- #
# Footprints
# --------------------------------------------------------------------------- #
def test_detect_ledges() -> None:
    rng = np.random.default_rng(0)
    hist = 5.0 + rng.normal(0, 0.4, 50)                      # gentle background
    hist[30] = 40.0                                          # a cliff/wall at bin 30
    ledges = detect_ledges(_profile_from_hist(hist), z=3.0)
    assert any(abs(i - 30) <= 1 for i in ledges)             # detected near the wall
    flat = _profile_from_hist(np.full(40, 7.0))
    assert detect_ledges(flat) == []                         # no cliffs in a flat profile


def test_poor_high() -> None:
    n = 60
    blunt = np.full(n, 10.0)                                 # never tapers -> poor high
    assert poor_high(_profile_from_hist(blunt)) is True
    taper = np.concatenate([np.full(n - 4, 10.0), [2, 1, 0.2, 0.0]])  # tapers at top
    assert poor_high(_profile_from_hist(taper)) is False


# --------------------------------------------------------------------------- #
# Time-decayed gravity
# --------------------------------------------------------------------------- #
def test_decayed_poc_migration() -> None:
    # First half trades ~90, recent half ~100 -> decayed POC migrates UP vs lifetime.
    n = 80
    lvl = np.concatenate([np.full(n // 2, 90.0), np.full(n // 2, 100.0)])
    high, low, close = lvl + 0.5, lvl - 0.5, lvl.copy()
    vol = np.full(n, 1e6)
    df = pd.DataFrame({"Open": lvl, "High": high, "Low": low, "Close": close, "Volume": vol})
    prof = VolumeProfileCalculator(num_bins=60).calculate(df)
    up = decayed_poc(high, low, close, vol, prof, halflife=8.0)
    assert up >= prof.poc and up > 95.0                      # recency pulls POC toward 100
    # Reverse the order -> decayed POC migrates DOWN.
    down = decayed_poc(high[::-1].copy(), low[::-1].copy(), close[::-1].copy(), vol, prof, 8.0)
    assert down < 95.0


def test_vacr_ratio() -> None:
    prof = _profile_from_hist(np.array([1, 2, 8, 20, 8, 2, 1.]), lo=98, hi=104)
    vacr = value_area_compression_ratio(prof, ref_price=101.0)
    assert vacr > 0 and np.isfinite(vacr)


# --------------------------------------------------------------------------- #
# Dataset builder + harness
# --------------------------------------------------------------------------- #
def test_build_structural_dataset_shape_and_no_lookahead() -> None:
    df = _ohlcv(440)
    ds = build_structural_dataset(df, lookback=120, horizon=20, stride=5,
                                  vacr_window=15, poc_window=5, symbol="SYN")
    assert isinstance(ds, FactorDataset) and len(ds) > 0
    assert ds.feature_names == STRUCTURAL_FEATURES and ds.X.shape[1] == 13
    assert ds.X.shape[0] == len(ds.y) == len(ds.baseline)
    assert np.isfinite(ds.X).all() and np.isfinite(ds.y).all()
    assert ds.purge_samples == int(np.ceil(20 / 5))
    # No look-ahead: last sample leaves a full forward horizon.
    assert ds.timestamps is not None
    assert ds.timestamps[-1] <= df.index[len(df) - 1 - ds.horizon]


def test_structural_dataset_feeds_cpcv() -> None:
    ds = build_structural_dataset(_ohlcv(460, seed=7), lookback=120, horizon=20,
                                  stride=5, vacr_window=15, symbol="SYN")
    res = cpcv_factor_eval(ds, alpha=1.0)
    assert isinstance(res, FactorCVResult)
    assert res.n_paths > 0 and np.isfinite(res.oos_ic_mean)
    assert len(res.mean_weights) == 13


def test_build_structural_meta_dataset() -> None:
    df = _ohlcv(460, seed=5)
    ds = build_structural_meta_dataset(df, lookback=120, horizon=20, stride=5,
                                       vacr_window=15, side=1, symbol="SYN")
    assert isinstance(ds, MetaDataset)
    assert ds.feature_names == STRUCTURAL_FEATURES and ds.X.shape[1] == 13
    assert ds.X.shape[0] == len(ds.meta_label) == len(ds.side) == len(ds.realized_return)
    assert set(np.unique(ds.meta_label).tolist()) <= {0, 1}     # binary win/loss
    assert set(np.unique(ds.side).tolist()) == {1}              # fixed long side
    assert np.isfinite(ds.X).all()
    # No look-ahead: last event leaves a full barrier horizon.
    assert ds.timestamps is not None
    assert ds.timestamps[-1] <= df.index[len(df) - 1 - ds.horizon]
    # Max holding period: realised holding in [1, horizon]; convenience props populated.
    assert ds.holding_bars is not None and len(ds.holding_bars) == len(ds)
    assert int(ds.holding_bars.min()) >= 1 and int(ds.holding_bars.max()) <= ds.horizon
    assert 1.0 <= ds.mean_holding <= ds.horizon
    assert 0.0 <= ds.pct_capped <= 100.0
    # pct_capped is exactly the share of trades that ran to the time stop.
    assert abs(ds.pct_capped - float((ds.holding_bars >= ds.horizon).mean() * 100.0)) < 1e-9
    # Feeds the meta harness.
    res = cpcv_meta_eval(ds, threshold=0.55)
    assert isinstance(res, MetaCVResult) and np.isfinite(res.oos_auc_mean)


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} structure tests …\n")
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
