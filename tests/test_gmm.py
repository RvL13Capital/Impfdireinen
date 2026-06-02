"""Tests for vpts.structure.gmm — weighted EM, BIC model selection, features, no-look-ahead.

The EM is checked on hand-built mixtures with known means; the feature vector is
checked for bounds/finiteness on bimodal vs unimodal profiles; the dataset builder
is checked for shape and no-look-ahead, then fed to the real CPCV factor harness.

    python tests/test_gmm.py
    pytest tests/test_gmm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import cpcv_factor_eval  # noqa: E402
from vpts.ml.models import FactorCVResult, FactorDataset  # noqa: E402
from vpts.profile.models import VolumeProfile  # noqa: E402
from vpts.structure.gmm import (  # noqa: E402
    GMM_FEATURES,
    build_gmm_dataset,
    fit_gmm_1d,
    gmm_feature_vector,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _gauss(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def _profile_from_hist(hist: np.ndarray, lo: float = 0.0, hi: float = 100.0) -> VolumeProfile:
    hist = np.asarray(hist, float)
    n = hist.size
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
# EM fit
# --------------------------------------------------------------------------- #
def test_em_recovers_bimodal_means() -> None:
    # Two well-separated lumps at u=0.25 and u=0.75 (twice the weight on the upper).
    u = np.linspace(0.0, 1.0, 101)
    w = _gauss(u, 0.25, 0.04) + 2.0 * _gauss(u, 0.75, 0.04)
    fit = fit_gmm_1d(u, w, 2, n_eff=100.0)
    assert fit.k == 2 and fit.means[0] < fit.means[1]          # sorted ascending
    assert abs(fit.means[0] - 0.25) < 0.04                     # lower hidden POC
    assert abs(fit.means[1] - 0.75) < 0.04                     # upper hidden POC
    assert fit.weights[1] > fit.weights[0]                     # upper node is heavier
    assert abs(float(fit.weights.sum()) - 1.0) < 1e-9


def test_bic_prefers_two_for_bimodal_one_for_unimodal() -> None:
    u = np.linspace(0.0, 1.0, 101)
    bi = _gauss(u, 0.25, 0.04) + _gauss(u, 0.75, 0.04)
    uni = _gauss(u, 0.5, 0.05)
    bic_bi = {k: fit_gmm_1d(u, bi, k, n_eff=100.0).bic for k in (1, 2)}
    bic_uni = {k: fit_gmm_1d(u, uni, k, n_eff=100.0).bic for k in (1, 2)}
    assert bic_bi[2] < bic_bi[1]      # two modes win on the bimodal histogram
    assert bic_uni[1] < bic_uni[2]    # one mode wins on the unimodal histogram


def test_em_degenerate_inputs() -> None:
    # All volume in one bin, or asking for more comps than non-zero bins → k=1, finite.
    u = np.linspace(0.0, 1.0, 20)
    w = np.zeros(20); w[7] = 1.0
    fit = fit_gmm_1d(u, w, 2, n_eff=100.0)
    assert fit.k == 1 and np.isfinite(fit.bic) and np.isfinite(fit.means[0])


# --------------------------------------------------------------------------- #
# Feature vector
# --------------------------------------------------------------------------- #
def test_feature_vector_bimodal_vs_unimodal() -> None:
    u = np.linspace(0.0, 1.0, 80)
    bi = _profile_from_hist(_gauss(u, 0.25, 0.04) + _gauss(u, 0.75, 0.04))
    uni = _profile_from_hist(_gauss(u, 0.5, 0.05))

    fb = dict(zip(GMM_FEATURES, gmm_feature_vector(bi, close_price=50.0)))
    fu = dict(zip(GMM_FEATURES, gmm_feature_vector(uni, close_price=50.0)))

    assert np.all(np.isfinite(list(fb.values()))) and np.all(np.isfinite(list(fu.values())))
    # Bimodal: two modes, real separation, a real density valley.
    assert fb["gmm_n_modes"] >= 2.0
    assert fb["gmm_separation"] > fu["gmm_separation"]
    assert fb["gmm_antimode_depth"] > 0.1
    # Bounds / encodings hold for both.
    for f in (fb, fu):
        assert f["gmm_price_zone"] in (-1.0, 0.0, 1.0)
        assert -1.0 <= f["gmm_weight_imbalance"] <= 1.0
        assert 0.0 <= f["gmm_antimode_depth"] <= 1.0
        assert f["gmm_n_modes"] in (1.0, 2.0, 3.0)


def test_feature_vector_neutral_on_tiny_profile() -> None:
    # Too few bins → graceful unimodal-neutral row (no crash, all finite).
    tiny = _profile_from_hist(np.array([1.0, 2.0]))
    vec = gmm_feature_vector(tiny, close_price=50.0)
    assert vec.shape == (len(GMM_FEATURES),) and np.all(np.isfinite(vec))
    assert vec[0] == 1.0 and vec[1] == 0.0     # one mode, no separation


# --------------------------------------------------------------------------- #
# Dataset builder
# --------------------------------------------------------------------------- #
def test_build_gmm_dataset_shape_and_no_lookahead() -> None:
    df = _ohlcv(420, seed=3)
    horizon = 20
    ds = build_gmm_dataset(df, lookback=120, horizon=horizon, stride=3, symbol="SYN")
    assert isinstance(ds, FactorDataset)
    assert ds.X.shape[1] == len(GMM_FEATURES) and len(ds) == ds.X.shape[0]
    assert np.all(np.isfinite(ds.X)) and np.all(np.isfinite(ds.y))
    # No look-ahead: the last event leaves a full forward horizon.
    assert ds.timestamps is not None
    assert ds.timestamps[-1] <= df.index[len(df) - 1 - horizon]
    # Feeds the real CPCV factor harness.
    res = cpcv_factor_eval(ds)
    assert isinstance(res, FactorCVResult) and np.isfinite(res.oos_ic_mean)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
