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
    ENRICHED_FEATURES,
    FactorCVResult,
    RidgeFactorModel,
    build_enriched_factor_dataset,
    build_factor_dataset,
    cpcv_factor_eval,
    permutation_test_factor,
)
from vpts.ml.models import FactorDataset  # noqa: E402

_NAMES = ("value_area", "key_level", "quiet", "patterns")


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
