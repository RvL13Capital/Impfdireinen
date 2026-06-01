"""Tests for vpts.ml — learned factor weights + CPCV evaluation.

Model/eval tests use synthetic feature matrices (fast); one smoke test exercises
the real pipeline-based dataset builder on synthetic OHLCV.

    python tests/test_ml.py
    pytest tests/test_ml.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
    CROSS_SECTIONAL_FEATURES,
    ENRICHED_FEATURES,
    FactorCVResult,
    RidgeFactorModel,
    build_cross_sectional_panel,
    build_enriched_factor_dataset,
    build_factor_dataset,
    cpcv_factor_eval,
    cross_sectional_ic_eval,
    permutation_test_cross_sectional,
    permutation_test_factor,
)
from vpts.ml.models import CrossSectionalICResult, CrossSectionalPanel, FactorDataset  # noqa: E402

_NAMES = ("value_area", "key_level", "quiet", "patterns")
_XS_NAMES = ("mom_21", "mom_252_21", "vol_60", "vol_trend")


def _panel(n_dates: int = 120, n_names: int = 12, signal: float = 0.0,
           seed: int = 0) -> CrossSectionalPanel:
    """Synthetic cross-sectional panel: feature 0 drives the per-date rank iff signal>0."""
    rng = np.random.default_rng(seed)
    X, y, did = [], [], []
    for d in range(n_dates):
        Xi = rng.uniform(-0.5, 0.5, size=(n_names, 4))
        y.append(signal * Xi[:, 0] + 0.05 * rng.normal(size=n_names))
        X.append(Xi)
        did += [d] * n_names
    return CrossSectionalPanel(
        X=np.vstack(X), y=np.concatenate(y), date_id=np.array(did, dtype=int),
        feature_names=_XS_NAMES, horizon=20, rebalance=5, n_dates=n_dates,
        symbols=tuple(f"N{i}" for i in range(n_names)))


def _frames(n_names: int = 8, n: int = 420, seed: int = 11) -> dict:
    """A small universe of synthetic OHLCV frames sharing one business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=n, freq="B")
    out = {}
    for i in range(n_names):
        ret = rng.normal(0.0002 * (i - n_names / 2), 0.02, n)
        close = 40 * np.exp(np.cumsum(ret))
        vol = (2e6 + rng.integers(-4e5, 4e5, n)).astype(float)
        out[f"N{i:02d}"] = pd.DataFrame(
            {"Open": close, "High": close * 1.01, "Low": close * 0.99,
             "Close": close, "Volume": vol}, index=idx)
    return out


def _dataset(n: int = 300, signal: float = 0.0, seed: int = 0) -> FactorDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = signal * X[:, 0] + 0.1 * rng.normal(size=n)   # feature 0 predicts y iff signal>0
    return FactorDataset(X=X, y=y, baseline=X[:, 0].copy(), feature_names=_NAMES,
                         horizon=1, stride=1, symbol="SYN")


def _ohlcv(n: int = 220, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 6 * np.sin(t / 22.0) + np.cumsum(rng.normal(0, 0.15, n))
    close = np.maximum(close, 5)
    rngs = 1.0 + 0.5 * np.abs(np.sin(t / 22.0))
    high, low = close + rngs / 2 + 0.2, close - rngs / 2 - 0.2
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = (3e6 + 1e6 * np.cos(t / 22.0) + rng.integers(-3e5, 3e5, n)).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": vol}, index=idx)


# --------------------------------------------------------------------------- #
# Ridge model
# --------------------------------------------------------------------------- #
def test_ridge_recovers_linear_relationship() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    true = np.array([2.0, -1.0, 0.5])
    m = RidgeFactorModel(alpha=1e-6).fit(X, X @ true)
    assert np.corrcoef(m.predict(X), X @ true)[0, 1] > 0.99
    assert np.sign(m.weights_[0]) == 1 and np.sign(m.weights_[1]) == -1
    assert abs(m.weights_[0]) > abs(m.weights_[2])     # importance ordering recovered


def test_ridge_guards() -> None:
    try:
        RidgeFactorModel(alpha=-1.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for alpha < 0")
    try:
        RidgeFactorModel().predict(np.zeros((2, 4)))
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError predicting before fit")


# --------------------------------------------------------------------------- #
# CPCV factor evaluation
# --------------------------------------------------------------------------- #
def test_cpcv_detects_real_signal() -> None:
    res = cpcv_factor_eval(_dataset(n=360, signal=0.6), alpha=1.0)
    assert isinstance(res, FactorCVResult)
    assert res.oos_ic_mean > 0.3                       # the planted signal shows up OOS
    assert res.pct_folds_positive_ic > 80
    # value_area (feature 0) is the dominant learned weight.
    assert int(np.argmax(np.abs(res.mean_weights))) == 0


def test_cpcv_random_features_have_no_edge() -> None:
    res = cpcv_factor_eval(_dataset(n=360, signal=0.0, seed=5), alpha=1.0)
    assert abs(res.oos_ic_mean) < 0.15                 # ~0 OOS on pure noise
    assert "Factor-weight CPCV" in res.summary()
    json.dumps(res.as_dict())


def test_cpcv_factor_eval_too_small_raises() -> None:
    try:
        cpcv_factor_eval(_dataset(n=12, signal=0.5))  # every fold starved of train rows
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for too-few folds")


# --------------------------------------------------------------------------- #
# Dataset builder (real pipeline, synthetic OHLCV)
# --------------------------------------------------------------------------- #
def test_build_factor_dataset_shape_and_no_lookahead() -> None:
    df = _ohlcv(220)
    ds = build_factor_dataset(df, lookback=60, horizon=10, stride=5, symbol="SYN")
    assert len(ds) > 0
    assert ds.X.shape[1] == 4 and ds.X.shape[0] == len(ds.y) == len(ds.baseline)
    assert ds.feature_names == _NAMES
    assert np.isfinite(ds.X).all() and np.isfinite(ds.y).all()
    assert ds.purge_samples == int(np.ceil(10 / 5))
    # No look-ahead: the last sampled bar must leave room for its forward label.
    assert ds.timestamps is not None
    assert ds.timestamps[-1] <= df.index[len(df) - 1 - ds.horizon]


# --------------------------------------------------------------------------- #
# Enriched feature set (momentum / volatility / microstructure)
# --------------------------------------------------------------------------- #
def test_build_enriched_dataset_shape_and_no_lookahead() -> None:
    df = _ohlcv(320)
    ds = build_enriched_factor_dataset(df, lookback=120, horizon=10, stride=5, symbol="SYN")
    assert len(ds) > 0
    assert ds.feature_names == ENRICHED_FEATURES
    assert ds.X.shape[1] == len(ENRICHED_FEATURES) == 11
    assert ds.X.shape[0] == len(ds.y) == len(ds.baseline)
    # Every momentum/vol feature is fully warmed up — no NaNs leak through.
    assert np.isfinite(ds.X).all() and np.isfinite(ds.y).all()
    assert ds.purge_samples == int(np.ceil(10 / 5))
    # No look-ahead: the last sampled bar still leaves room for its forward label,
    # and the 120-bar momentum warm-up pushes the first sample past bar 120.
    assert ds.timestamps is not None
    assert ds.timestamps[-1] <= df.index[len(df) - 1 - ds.horizon]
    assert ds.timestamps[0] >= df.index[119]


# --------------------------------------------------------------------------- #
# Factor permutation (label-shuffle) significance test
# --------------------------------------------------------------------------- #
def test_permutation_test_factor_significance() -> None:
    # A planted signal must clear the label-shuffled null …
    hit = permutation_test_factor(_dataset(n=360, signal=0.6), n_permutations=80, seed=1)
    assert hit.real_ic > 0.3
    assert hit.real_ic > hit.null_ic_mean
    assert hit.p_value < 0.05
    assert "SIGNIFICANT" in hit.summary()
    json.dumps(hit.as_dict())
    # … and pure noise must NOT (the real IC sits inside the null distribution).
    noise = permutation_test_factor(_dataset(n=360, signal=0.0, seed=7),
                                    n_permutations=80, seed=1)
    assert noise.p_value > 0.10


# --------------------------------------------------------------------------- #
# Cross-sectional rank factors
# --------------------------------------------------------------------------- #
def test_build_cross_sectional_panel_shape_and_no_lookahead() -> None:
    frames = _frames(n_names=8, n=420)
    panel = build_cross_sectional_panel(frames, horizon=20, rebalance=5, min_names=5)
    assert panel.feature_names == CROSS_SECTIONAL_FEATURES == _XS_NAMES
    assert panel.X.shape[1] == 4 and panel.X.shape[0] == len(panel.y) == len(panel.date_id)
    assert panel.n_names == 8 and panel.purge_dates == int(np.ceil(20 / 5))
    # Ranks are centred & bounded; every cell finite (warm-up rows were dropped).
    assert panel.X.min() >= -0.5 - 1e-9 and panel.X.max() <= 0.5 + 1e-9
    assert np.isfinite(panel.X).all() and np.isfinite(panel.y).all()
    # Each date carries a full cross-section (>= min_names) and date_ids are contiguous.
    counts = np.bincount(panel.date_id)
    assert counts.min() >= 5 and panel.n_dates == counts.size
    # No look-ahead: the last sampled date leaves a full forward-return horizon.
    common = frames["N00"].index
    assert panel.dates is not None and panel.dates[-1] <= common[len(common) - 1 - 20]


def test_cross_sectional_detects_planted_signal() -> None:
    res = cross_sectional_ic_eval(_panel(n_dates=120, signal=0.5, seed=1), alpha=1.0)
    assert isinstance(res, CrossSectionalICResult)
    assert res.combined_oos_ic_mean > 0.3                 # planted rank signal shows up OOS
    assert res.pct_dates_positive_ic > 80
    assert int(np.argmax(np.abs(res.mean_weights))) == 0  # the driving factor is learned
    assert "Cross-sectional rank IC" in res.summary()
    json.dumps(res.as_dict())


def test_cross_sectional_noise_has_no_edge() -> None:
    res = cross_sectional_ic_eval(_panel(n_dates=120, signal=0.0, seed=3), alpha=1.0)
    assert abs(res.combined_oos_ic_mean) < 0.1            # ~0 OOS on pure noise


def test_cross_sectional_eval_too_small_raises() -> None:
    try:
        cross_sectional_ic_eval(_panel(n_dates=4, n_names=6, signal=0.5))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for too-few dates")


def test_permutation_test_cross_sectional_significance() -> None:
    hit = permutation_test_cross_sectional(_panel(n_dates=120, signal=0.5, seed=1),
                                           n_permutations=60, seed=2)
    assert hit.real_ic > 0.3 and hit.real_ic > hit.null_ic_mean
    assert hit.p_value < 0.05
    noise = permutation_test_cross_sectional(_panel(n_dates=120, signal=0.0, seed=3),
                                             n_permutations=60, seed=2)
    assert noise.p_value > 0.10


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} ml tests …\n")
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
